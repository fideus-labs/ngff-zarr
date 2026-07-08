// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible version of upgrade_ome_zarr that doesn't import
// @zarrita/storage (which pulls in Node.js-specific modules like node:fs).
import { fromOmeZarr, type MemoryStore } from "./from_ngff_zarr-browser.ts";
import { toOmeZarr } from "./to_ngff_zarr-browser.ts";
import {
  type UpgradeInput,
  upgradeOmeZarrImpl,
  type UpgradeOmeZarrOptions,
} from "./upgrade_ome_zarr_common.ts";

export type { UpgradeInput, UpgradeOmeZarrOptions };

/**
 * Resolve an in-place upgrade `input` to a writable store, browser edition.
 *
 * Only `Map` (MemoryStore) is writable in the browser. Local file paths require
 * a filesystem store the browser build intentionally omits, and FetchStore /
 * HTTP(S) URLs are read-only; all are rejected with guidance to pass `output`.
 */
function resolveWritableStore(input: UpgradeInput): Promise<MemoryStore> {
  if (input instanceof Map) {
    return Promise.resolve(input);
  }
  const message = typeof input === "string" &&
      !(input.startsWith("http://") || input.startsWith("https://"))
    ? "Local file paths are not supported in browser environments. Use a " +
      "MemoryStore for an in-place upgrade, or pass `output`."
    : "In-place upgrade requires a writable MemoryStore, but a read-only " +
      "store was given. Pass `output` to write the upgraded store to a new " +
      "location.";
  return Promise.reject(new Error(message));
}

/**
 * Browser-compatible version of {@link upgradeOmeZarr}.
 *
 * Offers the same two modes as the Node version:
 *
 * **In-place metadata-only rewrite** (same Zarr format, 0.5 <-> 0.6): only the
 * root group's `zarr.json` is rewritten and every array chunk is left
 * byte-for-byte untouched (the fix for issue #219). An in-place upgrade across
 * the Zarr v2 <-> v3 boundary (0.4 <-> 0.5/0.6) throws with guidance to pass
 * `output`.
 *
 * **Write-to-new-store conversion** (distinct `output`): the source is read
 * lazily and re-written at the requested version through the standard pipeline;
 * every supported transition works and the source store is never mutated.
 *
 * In the browser only `MemoryStore` inputs/outputs are supported — local file
 * paths require Node/Deno. For an in-place upgrade the `input` must therefore be
 * a `MemoryStore`; otherwise pass a `MemoryStore` `output`.
 *
 * @see https://github.com/fideus-labs/ngff-zarr/issues/219
 */
export function upgradeOmeZarr(
  input: UpgradeInput,
  options: UpgradeOmeZarrOptions = {},
): Promise<void> {
  return upgradeOmeZarrImpl(input, options, {
    fromOmeZarr,
    toOmeZarr,
    resolveWritableStore,
  });
}
