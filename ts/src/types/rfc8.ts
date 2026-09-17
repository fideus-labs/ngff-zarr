// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * The RFC-8 node model: typed paths, references, nodes and collections.
 *
 * RFC-8 ("Collections and Extensibility",
 * https://ngff.openmicroscopy.org/rfc/8/index.html) replaces the
 * `multiscales` wrapper with a typed node document stored under the `ome`
 * key of a Zarr group's attributes or of a standalone JSON file. This module
 * mirrors the Python port's `ngff_zarr.rfc8.model`.
 *
 * The model is deliberately open-world: a node of a type this package has no
 * dedicated handling for (a prefixed extension type, or `"multiscale"` /
 * `"singlescale"` until issue #714's later stages land) is carried as a
 * generic {@link OmeNode} and round-trips unchanged, and `attributes` is
 * opaque JSON whose prefixed extension keys (e.g. `mobie:grid`) are
 * preserved verbatim, never interpreted.
 */

import { NgffVersion } from "./supported_versions.ts";

/** The RFC-8 id production, shared by node ids and reference ids. */
export const RFC8_ID_PATTERN = /^[a-zA-Z0-9\-_.]+$/;

/**
 * Path type for a node stored in a Zarr group: implementations append
 * `zarr.json` to the path to reach the metadata document.
 */
export const PATH_TYPE_ZARR = "zarr";

/**
 * Path type for a node stored in a standalone JSON file: the path is the
 * metadata document itself.
 */
export const PATH_TYPE_JSON = "json";

/**
 * Whether a value is a prefixed extension identifier (`prefix:name`).
 *
 * Unprefixed identifiers are reserved for the core specification; custom
 * extensions introduce prefixed ones, where the prefix names the user or
 * organization maintaining the extension.
 */
export function isPrefixedIdentifier(value: string): boolean {
  const separator = value.indexOf(":");
  return separator > 0 && separator < value.length - 1;
}

/**
 * RFC-8 `Path`: a typed locator for out-of-document metadata.
 *
 * `type` is {@link PATH_TYPE_ZARR}, {@link PATH_TYPE_JSON}, or a prefixed
 * extension type (e.g. `"myorg:zip"`). `path` is a relative path
 * (IETF RFC 1808, resolved against the document that holds it), an absolute
 * `file://` path (IETF RFC 8089), or an `http(s)://` URL (IETF RFC 1738).
 *
 * Named `OmePath` like the Python port's class (whose bare `Path` would
 * collide with `pathlib`), so the two ports read alike.
 */
export interface OmePath {
  type: string;
  path: string;
}

/**
 * RFC-8 `Reference` to a node or coordinate system by `id`.
 *
 * `path` locates the document that declares the referenced `id`; it is
 * required for external references and omitted for in-document ones.
 */
export interface Reference {
  id: string;
  path?: OmePath | undefined;
}

/**
 * RFC-8 `Node`: the base of every 0.9.dev3 metadata document.
 *
 * `type` and `name` are required by the RFC; `id` (unique within the JSON
 * document), `attributes`, and the container fields `nodes` / `path` are
 * optional. Exactly one of `nodes` and `path` must be present on the node
 * types that define them; that is a validation rule (`validateCollection`),
 * not a parse error, so an invalid document can be parsed, inspected and
 * repaired.
 *
 * `extra` captures unrecognized top-level node keys on read and is
 * serialized back on write: RFC-8 extensions must survive a round trip. No
 * node carries a `version` field; the serializer stamps the root's.
 *
 * Exported as `OmeNode` rather than the RFC's bare `Node`, which would
 * collide with the DOM `Node` type in browser consumers.
 */
export interface OmeNode {
  type: string;
  name: string;
  id?: string | undefined;
  attributes?: Record<string, unknown> | undefined;
  nodes?: OmeNode[] | undefined;
  path?: OmePath | undefined;
  extra?: Record<string, unknown> | undefined;
}

/** Whether a node is a collection, whichever object carries it. */
export function isCollectionNode(node: OmeNode): boolean {
  return node.type === "collection";
}

/**
 * An RFC-8 node document plus the context needed to resolve it.
 *
 * `root` is typically a collection node but may be any node: RFC-8 lets
 * every node type head a document. `base` anchors relative path resolution:
 * the Zarr group directory or URL the document's `zarr.json` lives in, or a
 * standalone JSON file's parent. `undefined` (a document built in memory)
 * leaves relative paths unresolvable.
 *
 * `rootAttributes` carries the non-OME root attribute keys of the group the
 * document was read from, mirroring `NgffMultiscales.rootAttributes`.
 */
export class NgffCollection {
  root: OmeNode;
  base?: string | undefined;
  version: string;
  rootAttributes?: Record<string, unknown> | undefined;
  /** Dereferenced documents, cached per resolved location. */
  readonly cache = new Map<string, unknown>();

  constructor(options: {
    root: OmeNode;
    base?: string | undefined;
    version?: string | undefined;
    rootAttributes?: Record<string, unknown> | undefined;
  }) {
    this.root = options.root;
    this.base = options.base;
    this.version = options.version ?? NgffVersion.V09dev3;
    this.rootAttributes = options.rootAttributes;
  }

  /** Every node in the document, depth-first, root first. */
  *walk(): Generator<OmeNode> {
    function* walkNode(node: OmeNode): Generator<OmeNode> {
      yield node;
      for (const child of node.nodes ?? []) {
        yield* walkNode(child);
      }
    }
    yield* walkNode(this.root);
  }

  /**
   * The document's node with the given `id`, else with the `name`.
   *
   * Ids are unique within a document, so they are looked up first; names
   * are only unique among siblings, and the first match in depth-first
   * order wins.
   */
  node(idOrName: string): OmeNode {
    for (const node of this.walk()) {
      if (node.id === idOrName) {
        return node;
      }
    }
    for (const node of this.walk()) {
      if (node.name === idOrName) {
        return node;
      }
    }
    throw new Error(`No node with id or name '${idOrName}' in this document.`);
  }
}
