// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Environment-agnostic core of the RFC-8 collection readers and writers.
 *
 * Parsing, serialization, location resolution and document dispatch live
 * here, mirroring the Python port's `ngff_zarr.rfc8.serialize` / `resolve` /
 * `collection` modules; the store- and file-access differences between the
 * Node/Deno and browser entry points are injected through
 * {@link CollectionIo}.
 */

import type { NgffMultiscales } from "../types/multiscales.ts";
import { OmeNodeDocumentSchema } from "../schemas/rfc8.ts";
import { validateCollection } from "../utils/rfc8_validation.ts";
import {
  isCollectionNode,
  NgffCollection,
  type OmeNode,
  type OmePath,
  PATH_TYPE_JSON,
  PATH_TYPE_ZARR,
  type Reference,
} from "../types/rfc8.ts";
import { NgffVersion } from "../types/supported_versions.ts";

/**
 * The node keys this package models; everything else round-trips through
 * `extra`. `version` is the root-only key handled by the (de)serializer.
 */
const NODE_FIELDS = new Set([
  "type",
  "id",
  "name",
  "attributes",
  "nodes",
  "path",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Serialize a node tree to the object stored as the `ome` value.
 *
 * `version` stamps the root node (the RFC-8 root-only key) and is never
 * passed down to children. `attributes` is deep-copied verbatim, unknown and
 * prefixed keys included, and so are the `extra` keys, emitted after the
 * modeled ones in their captured order. Absent optional fields are omitted
 * rather than written as JSON `null`.
 */
export function nodeToOmeDict(
  node: OmeNode,
  version?: string,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (version !== undefined) {
    out.version = version;
  }
  out.type = node.type;
  if (node.id !== undefined) {
    out.id = node.id;
  }
  out.name = node.name;
  if (node.attributes !== undefined) {
    out.attributes = structuredClone(node.attributes);
  }
  if (node.nodes !== undefined) {
    out.nodes = node.nodes.map((child) => nodeToOmeDict(child));
  }
  if (node.path !== undefined) {
    out.path = { type: node.path.type, path: node.path.path };
  }
  for (const [key, value] of Object.entries(node.extra ?? {})) {
    out[key] = structuredClone(value);
  }
  return out;
}

function requireString(value: unknown, key: string, location: string): string {
  if (typeof value !== "string" || value === "") {
    throw new Error(
      `An RFC-8 node must declare a non-empty string '${key}'; got ` +
        `${JSON.stringify(value)} at ${location}.`,
    );
  }
  return value;
}

function nodeFromDict(
  data: Record<string, unknown>,
  location: string,
): OmeNode {
  const nodeType = requireString(data.type, "type", location);
  const name = requireString(data.name, "name", location);

  const nodeId = data.id;
  if (nodeId !== undefined && typeof nodeId !== "string") {
    throw new Error(
      `An RFC-8 node 'id' must be a string; got ${JSON.stringify(nodeId)} ` +
        `at ${location}.`,
    );
  }

  let attributes: Record<string, unknown> | undefined;
  if (data.attributes !== undefined) {
    if (!isRecord(data.attributes)) {
      throw new Error(
        `An RFC-8 node 'attributes' must be an object; got ` +
          `${JSON.stringify(data.attributes)} at ${location}.`,
      );
    }
    attributes = structuredClone(data.attributes);
  }

  let nodes: OmeNode[] | undefined;
  if (data.nodes !== undefined) {
    if (!Array.isArray(data.nodes)) {
      throw new Error(
        `An RFC-8 node 'nodes' must be an array of nodes; got ` +
          `${JSON.stringify(data.nodes)} at ${location}.`,
      );
    }
    nodes = data.nodes.map((child, index) => {
      const childLocation = `${location}.nodes[${index}]`;
      if (!isRecord(child)) {
        throw new Error(
          `An RFC-8 node 'nodes' entry must be an object; got ` +
            `${JSON.stringify(child)} at ${childLocation}.`,
        );
      }
      return nodeFromDict(child, childLocation);
    });
  }

  let path: OmePath | undefined;
  if (data.path !== undefined) {
    if (!isRecord(data.path)) {
      throw new Error(
        `An RFC-8 node 'path' must be an object with 'type' and 'path'; ` +
          `got ${JSON.stringify(data.path)} at ${location}.`,
      );
    }
    path = {
      type: requireString(data.path.type, "path.type", location),
      path: requireString(data.path.path, "path.path", location),
    };
  }

  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (!NODE_FIELDS.has(key)) {
      extra[key] = structuredClone(value);
    }
  }

  return {
    type: nodeType,
    name,
    ...(nodeId !== undefined && { id: nodeId }),
    ...(attributes !== undefined && { attributes }),
    ...(nodes !== undefined && { nodes }),
    ...(path !== undefined && { path }),
    ...(Object.keys(extra).length > 0 && { extra }),
  };
}

/**
 * Parse an `ome` value into the root node and the declared version.
 *
 * The root's `version` key is returned separately rather than modeled on
 * the node: RFC-8 puts it on the root only, and the serializer stamps it
 * back. A non-string `version` is reported rather than carried.
 */
export function nodeFromOmeDict(
  data: unknown,
): { node: OmeNode; version?: string | undefined } {
  if (!isRecord(data)) {
    throw new Error(
      `An RFC-8 'ome' value must be an object; got ${JSON.stringify(data)}.`,
    );
  }
  const version = data.version;
  if (version !== undefined && typeof version !== "string") {
    throw new Error(
      `An RFC-8 'version' must be a string; got ${JSON.stringify(version)}.`,
    );
  }
  const rootData = Object.fromEntries(
    Object.entries(data).filter(([key]) => key !== "version"),
  );
  return {
    node: nodeFromDict(rootData, "ome"),
    ...(version !== undefined && { version }),
  };
}

const HTTP_SCHEMES = ["http://", "https://"];

function isHttp(location: string): boolean {
  return HTTP_SCHEMES.some((scheme) => location.startsWith(scheme));
}

function joinLocalPath(base: string, relative: string): string {
  const segments = [...base.split("/"), ...relative.split("/")];
  const out: string[] = [];
  for (const [index, segment] of segments.entries()) {
    if (segment === "" && index > 0) {
      continue;
    }
    if (segment === ".") {
      continue;
    }
    if (segment === "..") {
      if (out.length > 1 || (out.length === 1 && out[0] !== "")) {
        out.pop();
      }
      continue;
    }
    out.push(segment);
  }
  return out.join("/");
}

/**
 * Resolve a path's location string against the document base.
 *
 * `base` is the directory-like location of the document that carries the
 * path: the Zarr group directory or URL for a document stored in group
 * attributes, the parent directory or URL of a standalone JSON file. A
 * relative path (IETF RFC 1808) is resolved against it; `file://` paths
 * (IETF RFC 8089) and `http(s)://` URLs stand alone.
 */
export function resolveLocation(path: OmePath, base?: string): string {
  const location = path.path;
  if (isHttp(location)) {
    return location;
  }
  if (location.startsWith("file://")) {
    const parsed = new URL(location);
    // RFC 8089 permits a Windows drive in the authority position
    // (`file://C:/...`); rejoin it with the path before decoding.
    return decodeURIComponent(`${parsed.host}${parsed.pathname}`);
  }
  if (base === undefined) {
    return location;
  }
  if (isHttp(base)) {
    return new URL(location, `${base.replace(/\/+$/, "")}/`).href;
  }
  return joinLocalPath(base, location);
}

/**
 * The base future relative paths in the resolved document count from.
 *
 * A `"zarr"` path's document lives in the group's own attributes, so the
 * group location is the base; a `"json"` path's document is the file, so
 * its parent is.
 */
export function documentBase(path: OmePath, location: string): string {
  if (path.type === PATH_TYPE_ZARR) {
    return location;
  }
  if (isHttp(location)) {
    return new URL(".", location).href.replace(/\/+$/, "");
  }
  const separator = location.lastIndexOf("/");
  return separator <= 0 ? location : location.slice(0, separator);
}

/** The environment-specific access a collection loader needs. */
export interface CollectionIo {
  /** Root group attributes of the Zarr store at `location`, or undefined. */
  readZarrAttrs(
    location: string,
  ): Promise<Record<string, unknown> | undefined>;
  /** The parsed JSON document at `location`. */
  readJsonDocument(location: string): Promise<unknown>;
  /** A multiscales image store at `location`, via the OME-Zarr reader. */
  readImage(location: string): Promise<NgffMultiscales>;
}

/**
 * Load the document an RFC-8 path locates.
 *
 * Returns the resolved location and the raw document: the Zarr group's
 * attribute mapping or the JSON file's root object. A prefixed extension
 * path type is opaque to this package and reported as such.
 */
export async function resolveOmeDocument(
  path: OmePath,
  base: string | undefined,
  io: CollectionIo,
): Promise<{ location: string; document: Record<string, unknown> }> {
  if (path.type !== PATH_TYPE_ZARR && path.type !== PATH_TYPE_JSON) {
    throw new Error(
      `Path type '${path.type}' is not resolvable by ngff-zarr; only ` +
        `'zarr' and 'json' paths are. A prefixed extension type is opaque ` +
        `to implementations that do not recognize it.`,
    );
  }
  const location = resolveLocation(path, base);
  if (path.type === PATH_TYPE_ZARR) {
    const document = await io.readZarrAttrs(location);
    if (document === undefined) {
      throw new Error(
        `No Zarr group with attributes found at '${location}' (resolved ` +
          `from path '${path.path}').`,
      );
    }
    return { location, document };
  }
  const document = await io.readJsonDocument(location);
  if (!isRecord(document)) {
    throw new Error(
      `The JSON document at '${location}' (resolved from path ` +
        `'${path.path}') is not an object.`,
    );
  }
  return { location, document };
}

/** What loading a node can produce; mirrors the Python `LoadedNode`. */
export type LoadedNode = NgffCollection | NgffMultiscales | OmeNode;

function dispatchDocument(
  path: OmePath,
  location: string,
  document: Record<string, unknown>,
  fallbackVersion: string,
  io: CollectionIo,
): Promise<LoadedNode> | LoadedNode {
  const ome = document.ome;
  const nodeShaped = isRecord(ome) && "type" in ome && !("multiscales" in ome);
  if (nodeShaped) {
    const { node, version } = nodeFromOmeDict(ome);
    if (isCollectionNode(node)) {
      return new NgffCollection({
        root: node,
        base: documentBase(path, location),
        version: version ?? fallbackVersion,
      });
    }
    return node;
  }
  const hasMultiscales = "multiscales" in document ||
    (isRecord(ome) && "multiscales" in ome);
  if (hasMultiscales) {
    if (path.type !== PATH_TYPE_ZARR) {
      throw new Error(
        `The document at '${location}' holds multiscales image metadata, ` +
          `which only a 'zarr' path can designate; got path type ` +
          `'${path.type}'.`,
      );
    }
    return io.readImage(location);
  }
  throw new Error(
    `The document at '${location}' is neither an RFC-8 node document nor ` +
      `a multiscales image.`,
  );
}

async function loadDocument(
  collection: NgffCollection,
  path: OmePath,
  io: CollectionIo,
): Promise<LoadedNode> {
  const cacheKey = `${path.type}\u0000${
    resolveLocation(path, collection.base)
  }`;
  const cached = collection.cache.get(cacheKey);
  if (cached !== undefined) {
    return cached as LoadedNode;
  }
  const { location, document } = await resolveOmeDocument(
    path,
    collection.base,
    io,
  );
  const loaded = await dispatchDocument(
    path,
    location,
    document,
    collection.version,
    io,
  );
  collection.cache.set(cacheKey, loaded);
  return loaded;
}

/**
 * Materialize what `target` designates, dereferencing its path.
 *
 * `target` is a node of `collection` (or its `id` / `name` as a string), or
 * a reference. A dereferenced collection document nests as another
 * {@link NgffCollection} with its base moved to the referenced location; a
 * multiscales image store (any OME-Zarr version up to 0.9.dev1) reads
 * through the OME-Zarr reader; any other RFC-8 node document returns its
 * parsed root node. An inline node (no `path`) stays in this document.
 * Dereferenced documents are cached per resolved location.
 */
export async function loadCollectionNodeWith(
  collection: NgffCollection,
  target: OmeNode | Reference | string,
  io: CollectionIo,
): Promise<LoadedNode> {
  if (typeof target === "string") {
    target = collection.node(target);
  }
  if (!("type" in target)) {
    const reference = target as Reference;
    if (reference.path === undefined) {
      return loadCollectionNodeWith(
        collection,
        collection.node(reference.id),
        io,
      );
    }
    const loaded = await loadDocument(collection, reference.path, io);
    if (loaded instanceof NgffCollection) {
      for (const node of loaded.walk()) {
        if (node.id === reference.id) {
          return loadCollectionNodeWith(loaded, node, io);
        }
      }
    }
    return loaded;
  }
  const node = target as OmeNode;
  if (node.path !== undefined) {
    return loadDocument(collection, node.path, io);
  }
  if (isCollectionNode(node)) {
    return new NgffCollection({
      root: node,
      base: collection.base,
      version: collection.version,
    });
  }
  return node;
}

/**
 * The `ome` node document in `attrs`, or a pointed error naming the reader
 * that fits what the input actually is.
 */
export function nodeDocumentOrThrow(
  attrs: Record<string, unknown>,
  where: string,
): Record<string, unknown> {
  const ome = attrs.ome;
  const isImage = "multiscales" in attrs ||
    (isRecord(ome) && "multiscales" in ome);
  if (isImage) {
    throw new Error(
      `The input at ${where} is a multiscales image store, not an RFC-8 ` +
        `node document. Read it with fromOmeZarr().`,
    );
  }
  if (!isRecord(ome) || !("type" in ome)) {
    throw new Error(
      `The input at ${where} carries no RFC-8 node document: expected an ` +
        `'ome' object with a 'type' key (OME-Zarr 0.9.dev3).`,
    );
  }
  return ome;
}

/** Root attribute keys the OME-Zarr writers own, across every version. */
export const OME_ROOT_KEYS: ReadonlySet<string> = new Set([
  "multiscales",
  "omero",
  "ome",
]);

/**
 * Parse the root attributes of a collection store: the node document under
 * `ome` plus the non-OME root attribute keys. With `validate` the raw
 * document runs the RFC-8 structural rules, then the node schema.
 */
export function parseCollectionRootAttrs(
  attrs: Record<string, unknown>,
  where: string,
  validate: boolean,
): {
  node: OmeNode;
  version?: string | undefined;
  rootAttributes?: Record<string, unknown> | undefined;
} {
  const ome = nodeDocumentOrThrow(attrs, where);
  if (validate) {
    const declared = typeof ome.version === "string" ? ome.version : undefined;
    validateCollection(ome, declared);
    OmeNodeDocumentSchema.parse(ome);
  }
  const rootAttributes = Object.fromEntries(
    Object.entries(attrs).filter(([key]) => !OME_ROOT_KEYS.has(key)),
  );
  return {
    ...nodeFromOmeDict(ome),
    ...(Object.keys(rootAttributes).length > 0 && { rootAttributes }),
  };
}

/**
 * Parse a standalone JSON document: the node document under the root `ome`
 * key. With `validate` the raw document runs the RFC-8 structural rules,
 * then the node schema.
 */
export function parseCollectionJsonDocument(
  document: unknown,
  where: string,
  validate: boolean,
): { node: OmeNode; version?: string | undefined } {
  if (!isRecord(document)) {
    throw new Error(`The JSON document at ${where} is not an object.`);
  }
  const ome = nodeDocumentOrThrow(document, where);
  if (validate) {
    const declared = typeof ome.version === "string" ? ome.version : undefined;
    validateCollection(ome, declared);
    OmeNodeDocumentSchema.parse(ome);
  }
  return nodeFromOmeDict(ome);
}

/** Reject a collection write at any version but 0.9.dev3. */
export function checkCollectionVersion(version: string): string {
  if (version !== NgffVersion.V09dev3) {
    throw new Error(
      `Unsupported version for an RFC-8 node document: '${version}'. Only ` +
        `'0.9.dev3' stores the node model.`,
    );
  }
  return version;
}

/**
 * Serialize an RFC-8 node document to its standalone JSON form,
 * `{"ome": ...}`.
 */
export function toCollectionJsonDocument(
  root: OmeNode,
  version: string,
): Record<string, unknown> {
  return { ome: nodeToOmeDict(root, checkCollectionVersion(version)) };
}
