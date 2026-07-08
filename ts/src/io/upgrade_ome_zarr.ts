// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";

import { fromOmeZarr, type MemoryStore } from "./from_ngff_zarr.ts";
import { toOmeZarr } from "./to_ngff_zarr.ts";
import {
  type UpgradeInput,
  upgradeOmeZarrImpl,
  type UpgradeOmeZarrOptions,
} from "./upgrade_ome_zarr_common.ts";

export type { UpgradeInput, UpgradeOmeZarrOptions };

/**
 * Resolve an in-place upgrade `input` to a writable store.
 *
 * `Map` (MemoryStore) and local file paths (via `@zarrita/storage`'s
 * `FileSystemStore`) are writable. Read-only inputs — `FetchStore`, HTTP(S)
 * URLs, or a bare `Readable` — are rejected with guidance to pass `output`,
 * since an in-place upgrade must be able to rewrite the store's root metadata.
 */
async function resolveWritableStore(
  input: UpgradeInput,
): Promise<MemoryStore> {
  if (input instanceof Map) {
    return input;
  }
  if (input instanceof zarr.FetchStore) {
    throw new Error(
      "In-place upgrade requires a writable store, but a read-only FetchStore " +
        "was given. Pass `output` to write the upgraded store to a new location.",
    );
  }
  if (typeof input === "string") {
    if (input.startsWith("http://") || input.startsWith("https://")) {
      throw new Error(
        "In-place upgrade requires a writable store, but a read-only HTTP(S) " +
          "URL was given. Pass `output` to write the upgraded store to a new " +
          "location.",
      );
    }
    if (typeof window !== "undefined") {
      throw new Error(
        "Local file paths are not supported in browser environments. Use a " +
          "MemoryStore for an in-place upgrade, or pass `output`.",
      );
    }
    const { FileSystemStore } = await import("@zarrita/storage");
    // Normalize the path for cross-platform compatibility (mirrors the reader).
    const normalizedPath = input.replace(/^\/([A-Za-z]:)/, "$1");
    return new FileSystemStore(normalizedPath) as unknown as MemoryStore;
  }
  // A generic Readable store (e.g. a TiffStore) with no write capability.
  throw new Error(
    "In-place upgrade requires a writable store (a local path or MemoryStore). " +
      "Pass `output` to write the upgraded store to a new location.",
  );
}

/**
 * Upgrade an OME-Zarr store to a different specification version.
 *
 * Mirrors the Python `upgrade_ome_zarr`, offering two modes:
 *
 * **In-place metadata-only rewrite.** When `output` is omitted (or resolves to
 * the same store as `input`) and the source and target share the same
 * underlying Zarr format — 0.5 <-> 0.6, both Zarr v3 — only the root group's
 * `zarr.json` is rewritten. Every array chunk on disk is left byte-for-byte
 * untouched. This is the fix for issue #219, where a naive "read, erase,
 * re-write" upgrade destroyed the array data it was meant to preserve. An
 * in-place upgrade that would cross the Zarr v2 <-> v3 boundary
 * (OME-Zarr 0.4 <-> 0.5/0.6) throws with guidance to pass `output` instead.
 *
 * **Write-to-new-store conversion.** When `output` is a store distinct from
 * `input`, the source is read lazily and re-written to `output` at the
 * requested version through the standard, tested write pipeline. Every
 * supported transition (0.4 <-> 0.5 <-> 0.6) works in this mode, and the source
 * store is never mutated.
 *
 * In-place upgrade of a local **path** store is Node/Deno-only (the browser
 * build ships no filesystem store); `MemoryStore` inputs work everywhere,
 * consistent with the other I/O functions.
 *
 * @param input Source store or path to read.
 * @param options Upgrade options; `version` defaults to `"0.6"`.
 * @returns A promise that resolves once the upgraded metadata has been written
 *   to `output` (or, for an in-place upgrade, back to `input`).
 *
 * @example
 * ```ts
 * // Upgrade a 0.5 store to 0.6 in place, keeping all array chunks.
 * await upgradeOmeZarr("image.ome.zarr", { version: "0.6" });
 *
 * // Write an upgraded copy to a new store.
 * await upgradeOmeZarr("image_v04.ome.zarr", {
 *   output: "image_v06.ome.zarr",
 *   version: "0.6",
 * });
 * ```
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
