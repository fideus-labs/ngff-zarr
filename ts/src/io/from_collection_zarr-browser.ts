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

import { NgffCollection, type OmeNode, type Reference } from "../types/rfc8.ts";
import {
  type CollectionIo,
  loadCollectionNodeWith as loadWithIo,
  type LoadedNode,
  parseCollectionJsonDocument,
  parseCollectionRootAttrs,
} from "./collection_common.ts";
import { fromOmeZarr, type MemoryStore } from "./from_ngff_zarr-browser.ts";

function isHttp(location: string): boolean {
  return location.startsWith("http://") || location.startsWith("https://");
}

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
export const browserCollectionIo: CollectionIo = {
  readZarrAttrs: async (location) => {
    try {
      const root = await zarr.open(resolveReadStore(location), {
        kind: "group",
      });
      return root.attrs as Record<string, unknown>;
    } catch {
      return undefined;
    }
  },
  readJsonDocument,
  readImage: (location) => fromOmeZarr(location),
};

export interface FromCollectionOptions {
  /** Run the RFC-8 structural rules and schema against the raw document. */
  validate?: boolean;
}

/**
 * Read an RFC-8 node document from a Zarr store's root group. See the
 * Node.js/Deno `fromCollectionZarr` for the semantics; this variant reads
 * http(s) URLs and in-memory stores only.
 */
export async function fromCollectionZarr(
  store: string | MemoryStore | zarr.Readable,
  options: FromCollectionOptions = {},
): Promise<NgffCollection> {
  const base = typeof store === "string"
    ? store.replace(/\/+$/, "")
    : undefined;
  const root = await zarr.open(resolveReadStore(store), { kind: "group" });
  const attrs = root.attrs as Record<string, unknown>;
  const { node, version, rootAttributes } = parseCollectionRootAttrs(
    attrs,
    `'${store}'`,
    options.validate ?? false,
  );
  return new NgffCollection({ root: node, base, version, rootAttributes });
}

/**
 * Read an RFC-8 node document from standalone JSON: an http(s) URL or an
 * already-parsed object.
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
    base = new URL(".", source).href.replace(/\/+$/, "");
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
 * its path over fetch. See `loadCollectionNode` in `collection_common.ts`
 * for the dispatch rules.
 */
export function loadCollectionNode(
  collection: NgffCollection,
  target: OmeNode | Reference | string,
): Promise<LoadedNode> {
  return loadWithIo(collection, target, browserCollectionIo);
}
