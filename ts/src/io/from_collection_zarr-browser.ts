// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Reading RFC-8 node documents (OME-Zarr 0.9.dev3) in the browser.
 *
 * The environment-agnostic parsing, resolution and dispatch live in
 * `collection_common.ts`; this module supplies fetch-based access only
 * (http(s) URLs and in-memory stores; no local file system).
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
import { fromOmeZarr, type MemoryStore } from "./from_ngff_zarr-browser.ts";

function resolveReadStore(
  store: string | MemoryStore | zarr.Readable,
): zarr.Readable {
  if (typeof store !== "string") {
    return store as zarr.Readable;
  }
  if (isHttp(store)) {
    return new zarr.FetchStore(store);
  }
  throw new Error(
    "Local file paths are not supported in browser environments. Use a " +
      "MemoryStore or an http(s) URL instead.",
  );
}

async function readJsonDocument(location: string): Promise<unknown> {
  if (!isHttp(location)) {
    throw new Error(
      "Local file paths are not supported in browser environments; " +
        `cannot read '${location}'.`,
    );
  }
  const response = await fetch(location);
  if (!response.ok) {
    throw new Error(`Failed to fetch '${location}': HTTP ${response.status}`);
  }
  return await response.json();
}

/** The browser environment access for the collection loader. */
const browserCollectionIo: CollectionIo = {
  resolveReadStore,
  readJsonDocument,
  readImage: (location) => fromOmeZarr(location),
};

/**
 * Read an RFC-8 node document from a Zarr store's root group: an http(s)
 * URL or an in-memory store.
 */
export function fromCollectionZarr(
  store: string | MemoryStore | zarr.Readable,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  return fromCollectionZarrWith(store, options, browserCollectionIo);
}

/**
 * Read an RFC-8 node document from standalone JSON: an http(s) URL or an
 * already-parsed object.
 */
export function fromCollectionJson(
  source: string | Record<string, unknown>,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  return fromCollectionJsonWith(source, options, browserCollectionIo);
}

/**
 * Materialize what `target` designates within `collection`, dereferencing
 * its path over fetch. See `loadCollectionNodeWith` in
 * `collection_common.ts` for the dispatch rules.
 */
export function loadCollectionNode(
  collection: NgffCollection,
  target: OmeNode | Reference | string,
): Promise<LoadedNode> {
  return loadCollectionNodeWith(collection, target, browserCollectionIo);
}
