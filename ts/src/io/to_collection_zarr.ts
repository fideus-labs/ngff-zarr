// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Writing RFC-8 node documents (OME-Zarr 0.9.dev3) in Node.js/Deno.
 *
 * Writing is metadata-only: a collection document references image stores,
 * it does not contain their arrays, so the referenced stores are written
 * separately (`toOmeZarr`) at their own versions.
 */

import type * as zarr from "zarrita";

import type { NgffCollection, OmeNode } from "../types/rfc8.ts";
import {
  toCollectionJsonDocument,
  type ToCollectionJsonOptions as ToCollectionJsonDocumentOptions,
  type ToCollectionOptions,
  writeCollectionRoot,
} from "./collection_common.ts";
import type { MemoryStore } from "./from_ngff_zarr.ts";

export interface ToCollectionJsonOptions
  extends ToCollectionJsonDocumentOptions {
  /** Also write the document to this local file path. */
  path?: string;
}

async function resolveWriteStore(
  store: string | MemoryStore,
): Promise<MemoryStore | zarr.Readable> {
  if (store instanceof Map) {
    return store;
  }
  if (store.startsWith("http://") || store.startsWith("https://")) {
    throw new Error(
      "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use a " +
        "local file path instead.",
    );
  }
  const { FileSystemStore } = await import("@zarrita/storage");
  const normalizedPath = store.replace(/^\/([A-Za-z]:)/, "$1");
  return new FileSystemStore(normalizedPath) as unknown as zarr.Readable;
}

/**
 * Write an RFC-8 node document as a Zarr store's root group: a local
 * directory path or an in-memory store. See `writeCollectionRoot` for the
 * semantics.
 */
export async function toCollectionZarr(
  store: string | MemoryStore,
  collection: NgffCollection | OmeNode,
  options: ToCollectionOptions = {},
): Promise<void> {
  await writeCollectionRoot(
    await resolveWriteStore(store),
    collection,
    options,
  );
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
  const document = toCollectionJsonDocument(collection, options);
  if (options.path !== undefined) {
    const { writeFile } = await import("node:fs/promises");
    await writeFile(
      options.path,
      `${JSON.stringify(document, null, 2)}\n`,
      "utf-8",
    );
  }
  return document;
}
