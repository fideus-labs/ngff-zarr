// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Reading RFC-8 node documents (OME-Zarr 0.9.dev3) in Node.js/Deno.
 *
 * The environment-agnostic parsing, resolution and dispatch live in
 * `collection_common.ts`; this module supplies local-file and network
 * access. The browser build uses `from_collection_zarr-browser.ts`.
 */

import * as zarr from "zarrita";

import { NgffCollection, type OmeNode, type Reference } from "../types/rfc8.ts";
import {
  type CollectionIo,
  loadCollectionNodeWith as loadWithIo,
  type LoadedNode,
  parseCollectionJsonDocument,
  parseCollectionRootAttrs,
} from "./collection_common.ts";
import { fromOmeZarr, type MemoryStore } from "./from_ngff_zarr.ts";

function isHttp(location: string): boolean {
  return location.startsWith("http://") || location.startsWith("https://");
}

async function resolveReadStore(
  store: string | MemoryStore | zarr.Readable,
): Promise<zarr.Readable> {
  if (typeof store !== "string") {
    return store as zarr.Readable;
  }
  if (isHttp(store)) {
    return new zarr.FetchStore(store);
  }
  if (typeof window !== "undefined") {
    throw new Error(
      "Local file paths are not supported in browser environments. Use a " +
        "MemoryStore or an http(s) URL instead.",
    );
  }
  const { FileSystemStore } = await import("@zarrita/storage");
  const normalizedPath = store.replace(/^\/([A-Za-z]:)/, "$1");
  return new FileSystemStore(normalizedPath) as unknown as zarr.Readable;
}

async function readZarrAttrs(
  location: string,
): Promise<Record<string, unknown> | undefined> {
  try {
    const store = await resolveReadStore(location);
    const root = await zarr.open(store, { kind: "group" });
    return root.attrs as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

async function readJsonDocument(location: string): Promise<unknown> {
  if (isHttp(location)) {
    const response = await fetch(location);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch '${location}': HTTP ${response.status}`,
      );
    }
    return await response.json();
  }
  const { readFile } = await import("node:fs/promises");
  return JSON.parse(await readFile(location, "utf-8"));
}

/** The Node.js/Deno environment access for the collection loader. */
export const nodeCollectionIo: CollectionIo = {
  readZarrAttrs,
  readJsonDocument,
  readImage: (location) => fromOmeZarr(location),
};

export interface FromCollectionOptions {
  /** Run the RFC-8 structural rules and schema against the raw document. */
  validate?: boolean;
}

/**
 * Read an RFC-8 node document from a Zarr store's root group.
 *
 * The store is a local directory path, an http(s) URL, or a key-to-bytes
 * store whose root group attributes carry the document under `ome`. The
 * root may be any node type; a collection is typical. The returned
 * collection's resolution base is the store, so relative paths in the
 * document dereference from it.
 */
export async function fromCollectionZarr(
  store: string | MemoryStore | zarr.Readable,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  const base = typeof store === "string"
    ? store.replace(/\/+$/, "")
    : undefined;
  const resolvedStore = await resolveReadStore(store);
  const root = await zarr.open(resolvedStore, { kind: "group" });
  const attrs = root.attrs as Record<string, unknown>;
  const { node, version, rootAttributes } = parseCollectionRootAttrs(
    attrs,
    `'${store}'`,
    options.validate ?? false,
  );
  return new NgffCollection({ root: node, base, version, rootAttributes });
}

/**
 * Read an RFC-8 node document from standalone JSON.
 *
 * `source` is the path or http(s) URL of a JSON file whose root object
 * carries the document under `ome`, or such a parsed object itself. For an
 * in-memory object the collection has no resolution base, so its relative
 * paths cannot be dereferenced.
 */
export async function fromCollectionJson(
  source: string | Record<string, unknown>,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  let document: unknown;
  let base: string | undefined;
  let where: string;
  if (typeof source === "string") {
    document = await readJsonDocument(source);
    where = `'${source}'`;
    if (isHttp(source)) {
      base = new URL(".", source).href.replace(/\/+$/, "");
    } else {
      const separator = source.lastIndexOf("/");
      base = separator <= 0 ? "." : source.slice(0, separator);
    }
  } else {
    document = source;
    where = "the given object";
  }
  const { node, version } = parseCollectionJsonDocument(
    document,
    where,
    options.validate ?? false,
  );
  return new NgffCollection({ root: node, base, version });
}

/**
 * Materialize what `target` designates within `collection`, dereferencing
 * its path with local-file and network access. See `loadCollectionNode` in
 * `collection_common.ts` for the dispatch rules.
 */
export function loadCollectionNode(
  collection: NgffCollection,
  target: OmeNode | Reference | string,
): Promise<LoadedNode> {
  return loadWithIo(collection, target, nodeCollectionIo);
}
