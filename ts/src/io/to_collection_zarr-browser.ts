// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Writing RFC-8 node documents (OME-Zarr 0.9.dev3) in the browser: into an
 * in-memory store, or to a standalone JSON document the caller persists.
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
import type { MemoryStore } from "./from_ngff_zarr-browser.ts";

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

/**
 * Write an RFC-8 node document as the root group of an in-memory store.
 * See the Node.js/Deno `toCollectionZarr` for the semantics.
 */
export async function toCollectionZarr(
  store: MemoryStore,
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
  const attributes: Record<string, unknown> = { ...(rootAttributes ?? {}) };
  attributes.ome = nodeToOmeDict(root, version);
  const location = zarr.root(store);
  await zarr.create(location, { attributes });
}

export interface ToCollectionJsonOptions {
  /** Version to stamp on the root node. Only `"0.9.dev3"` is accepted. */
  version?: string;
  /** Run the RFC-8 structural rules on the outgoing document. */
  validate?: boolean;
}

/** Serialize an RFC-8 node document to its standalone JSON form. */
export function toCollectionJson(
  collection: NgffCollection | OmeNode,
  options: ToCollectionJsonOptions = {},
): Record<string, unknown> {
  const version = options.version ?? "0.9.dev3";
  const root = collectionRoot(collection);
  if (options.validate ?? true) {
    validateCollection(nodeToOmeDict(root), checkCollectionVersion(version));
  }
  return toCollectionJsonDocument(root, version);
}
