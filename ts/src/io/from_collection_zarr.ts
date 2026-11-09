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

import type { NgffCollection, OmeNode, Reference } from "../types/rfc8.ts";
import {
  type CollectionIo,
  fromCollectionJsonWith,
  type FromCollectionOptions,
  fromCollectionZarrWith,
  isHttp,
  loadCollectionNodeWith,
  type LoadedNode,
} from "./collection_common.ts";
import { fromOmeZarr, type MemoryStore } from "./from_ngff_zarr.ts";

async function resolveReadStore(
  store: string | MemoryStore | zarr.Readable,
): Promise<zarr.Readable> {
  if (typeof store !== "string") {
    return store as zarr.Readable;
  }
  if (isHttp(store)) {
    return new zarr.FetchStore(store);
  }
  const { FileSystemStore } = await import("@zarrita/storage");
  const normalizedPath = store.replace(/^\/([A-Za-z]:)/, "$1");
  return new FileSystemStore(normalizedPath) as unknown as zarr.Readable;
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
const nodeCollectionIo: CollectionIo = {
  resolveReadStore,
  readJsonDocument,
  readImage: (location) => fromOmeZarr(location),
};

/** Read an RFC-8 node document from a Zarr store's root group. */
export function fromCollectionZarr(
  store: string | MemoryStore | zarr.Readable,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  return fromCollectionZarrWith(store, options, nodeCollectionIo);
}

/** Read an RFC-8 node document from standalone JSON. */
export function fromCollectionJson(
  source: string | Record<string, unknown>,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  return fromCollectionJsonWith(source, options, nodeCollectionIo);
}

/**
 * Materialize what `target` designates within `collection`, dereferencing
 * its path with local-file and network access. See `loadCollectionNodeWith`
 * in `collection_common.ts` for the dispatch rules.
 */
export function loadCollectionNode(
  collection: NgffCollection,
  target: OmeNode | Reference | string,
): Promise<LoadedNode> {
  return loadCollectionNodeWith(collection, target, nodeCollectionIo);
}
