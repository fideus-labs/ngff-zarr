// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Writing RFC-8 node documents (OME-Zarr 0.9.dev3) in the browser: into an
 * in-memory store, or to a standalone JSON document the caller persists.
 */

import type { NgffCollection, OmeNode } from "../types/rfc8.ts";
import {
  toCollectionJsonDocument,
  type ToCollectionJsonOptions,
  type ToCollectionOptions,
  writeCollectionRoot,
} from "./collection_common.ts";
import type { MemoryStore } from "./from_ngff_zarr-browser.ts";

/**
 * Write an RFC-8 node document as the root group of an in-memory store.
 * See `writeCollectionRoot` for the semantics.
 */
export function toCollectionZarr(
  store: MemoryStore,
  collection: NgffCollection | OmeNode,
  options: ToCollectionOptions = {},
): Promise<void> {
  return writeCollectionRoot(store, collection, options);
}

/** Serialize an RFC-8 node document to its standalone JSON form. */
export function toCollectionJson(
  collection: NgffCollection | OmeNode,
  options: ToCollectionJsonOptions = {},
): Record<string, unknown> {
  return toCollectionJsonDocument(collection, options);
}
