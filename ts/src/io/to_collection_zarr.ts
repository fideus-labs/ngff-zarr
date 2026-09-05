// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Writing RFC-8 node documents (OME-Zarr 0.9.dev3) in Node.js/Deno.
 *
 * Writing is metadata-only: a collection document references image stores,
 * it does not contain their arrays, so the referenced stores are written
 * separately (`toOmeZarr`) at their own versions.
 */

import * as zarr from "zarrita";

import { NgffCollection, type OmeNode } from "../types/rfc8.ts";
import { validateCollection } from "../utils/rfc8_validation.ts";
import {
  checkCollectionVersion,
  nodeToOmeDict,
  OME_ROOT_KEYS,
  toCollectionJsonDocument,
} from "./collection_common.ts";
import type { MemoryStore } from "./from_ngff_zarr.ts";

export interface ToCollectionOptions {
  /** Version to stamp on the root node. Only `"0.9.dev3"` is accepted. */
  version?: string;
  /** Run the RFC-8 structural rules on the outgoing document. */
  validate?: boolean;
  /** Root attribute keys written beside the OME key. */
  rootAttributes?: Record<string, unknown> | undefined;
}

function collectionRoot(collection: NgffCollection | OmeNode): OmeNode {
  return collection instanceof NgffCollection ? collection.root : collection;
}

async function resolveWriteStore(
  store: string | MemoryStore,
): Promise<MemoryStore | unknown> {
  if (store instanceof Map) {
    return store;
  }
  if (
    typeof store === "string" &&
    (store.startsWith("http://") || store.startsWith("https://"))
  ) {
    throw new Error(
      "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use a " +
        "local file path instead.",
    );
  }
  if (typeof window !== "undefined") {
    throw new Error(
      "Local file paths are not supported in browser environments. Use a " +
        "MemoryStore instead.",
    );
  }
  const { FileSystemStore } = await import("@zarrita/storage");
  const normalizedPath = (store as string).replace(/^\/([A-Za-z]:)/, "$1");
  return new FileSystemStore(normalizedPath);
}

/**
 * Write an RFC-8 node document as a Zarr store's root group.
 *
 * Writes metadata only: a Zarr format 3 group whose `attributes.ome` is the
 * serialized document. Unlike `toOmeZarr` nothing beneath the root is ever
 * removed, so a collection root can hold the very stores its document
 * references. Non-OME attributes of an existing root survive;
 * `rootAttributes` (or, when omitted, the collection's own
 * `rootAttributes`) are written over them, and the OME key over both.
 */
export async function toCollectionZarr(
  store: string | MemoryStore,
  collection: NgffCollection | OmeNode,
  options: ToCollectionOptions = {},
): Promise<void> {
  const version = checkCollectionVersion(options.version ?? "0.9.dev3");
  const root = collectionRoot(collection);
  if (options.validate ?? true) {
    validateCollection(nodeToOmeDict(root), version);
  }
  const rootAttributes = options.rootAttributes ??
    (collection instanceof NgffCollection
      ? collection.rootAttributes
      : undefined);
  const owned = Object.keys(rootAttributes ?? {}).filter((key) =>
    OME_ROOT_KEYS.has(key)
  );
  if (owned.length > 0) {
    throw new Error(
      `rootAttributes cannot set ${JSON.stringify(owned.sort())}: the ` +
        `writer derives the OME metadata from the collection.`,
    );
  }
  const resolvedStore = await resolveWriteStore(store);
  const attributes: Record<string, unknown> = {};
  try {
    const existing = await zarr.open(resolvedStore as zarr.Readable, {
      kind: "group",
    });
    for (
      const [key, value] of Object.entries(
        existing.attrs as Record<string, unknown>,
      )
    ) {
      if (!OME_ROOT_KEYS.has(key)) {
        attributes[key] = value;
      }
    }
  } catch {
    // No existing root group: nothing to preserve.
  }
  Object.assign(attributes, rootAttributes ?? {});
  attributes.ome = nodeToOmeDict(root, version);
  const location = zarr.root(resolvedStore as MemoryStore);
  await zarr.create(location, { attributes });
}

export interface ToCollectionJsonOptions {
  /** Version to stamp on the root node. Only `"0.9.dev3"` is accepted. */
  version?: string;
  /** Run the RFC-8 structural rules on the outgoing document. */
  validate?: boolean;
  /** Also write the document to this local file path. */
  path?: string;
}

/**
 * Serialize an RFC-8 node document to standalone JSON.
 *
 * Returns the `{"ome": ...}` document; when `path` is given it is also
 * written there as UTF-8 JSON.
 */
export async function toCollectionJson(
  collection: NgffCollection | OmeNode,
  options: ToCollectionJsonOptions = {},
): Promise<Record<string, unknown>> {
  const version = options.version ?? "0.9.dev3";
  const root = collectionRoot(collection);
  if (options.validate ?? true) {
    validateCollection(nodeToOmeDict(root), checkCollectionVersion(version));
  }
  const document = toCollectionJsonDocument(root, version);
  if (options.path !== undefined) {
    if (typeof window !== "undefined") {
      throw new Error(
        "Local file paths are not supported in browser environments. Use " +
          "the returned document instead.",
      );
    }
    const { writeFile } = await import("node:fs/promises");
    await writeFile(
      options.path,
      `${JSON.stringify(document, null, 2)}\n`,
      "utf-8",
    );
  }
  return document;
}
