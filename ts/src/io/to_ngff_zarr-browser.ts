// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible version of to_ngff_zarr that doesn't import @zarrita/storage
// (which contains Node.js-specific modules like node:fs, node:buffer, node:path)
import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import { NgffScene } from "../types/scene.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import {
  isV06Transform,
  transformationToOzx,
} from "./transformation_common.ts";
import { sceneToOzx } from "./scene_common.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import type { ChunksPerShard } from "../utils/sharding.ts";
import { writableStoreBelow } from "../utils/store_below.ts";
import type { MemoryStore } from "./from_ngff_zarr-browser.ts";
import { memoryStoreToZip } from "./rfc9_zip.ts";
import { arrayWriter } from "./array_writer.ts";
import { writeToStore } from "./to_ngff_zarr_common.ts";
import {
  DEFAULT_OZX_CHUNKS_PER_SHARD,
  gateOzxVersion,
  type OzxVersion,
  type ProgressCallback,
  storeToZipData,
  type StoreToZipOptions,
  writeNgffMultiscalesToMemoryStore,
} from "./to_ngff_zarr_ozx_common.ts";

export type { StoreToZipOptions } from "./to_ngff_zarr_ozx_common.ts";

export { isOzxPath } from "./rfc9_zip.ts";

export interface ToOmeZarrOptions {
  overwrite?: boolean;
  /**
   * OME-Zarr version to write. Defaults to "0.5", as the Node.js
   * toOmeZarr does, and to "0.6" for a scene or a transformation, which
   * that version introduced. Version "0.6" writes RFC 5 coordinate systems
   * and transformations.
   */
  version?: "0.4" | "0.5" | "0.6" | "0.9.dev1";
  /**
   * Store each block of chunks as one Zarr v3 shard: how many chunks a shard
   * spans along each axis -- one count for every axis, one per axis in order,
   * or per axis name (an axis left out spans 1). Requires version 0.5 or
   * later. Omitted, the arrays are not sharded.
   */
  chunksPerShard?: ChunksPerShard;
  /**
   * Custom codec pipeline for array compression. When omitted the default
   * ``blosc(zstd)`` pipeline from {@link defaultCodecs} is used. Use
   * {@link codecFromName} or {@link bytesOnlyCodecs} to build common pipelines.
   */
  codecs?: ZarrCodec[];
  /**
   * Write consolidated metadata into the root `zarr.json` once the arrays are
   * in place (default `true`). Set `false` for a store destined for a backend
   * that rejects it, such as Icechunk, which runs its own consolidation and
   * treats a consolidated block as interference.
   */
  consolidateMetadata?: boolean;
  /**
   * Called after each chunk is written, or each shard when sharded, with the
   * writes completed so far and the total: across every level of an image,
   * and across every image of a scene.
   */
  onProgress?: ProgressCallback;
  /**
   * Write below this path of the store rather than at its root, as if the
   * group there were the root of a store of its own: its metadata is
   * consolidated in its own `zarr.json`, and the groups above it that the
   * store lacks are created. This is how a displacement field, or another
   * node a transformation references by `path`, is staged in a MemoryStore
   * before the scene or the transformation is written with
   * `{ overwrite: false }`; {@link storeToZip} then packs the store into an
   * `.ozx` archive.
   */
  path?: string;
}

/** @deprecated Use {@link ToOmeZarrOptions} instead. */
export type ToNgffZarrOptions = ToOmeZarrOptions;

/** Options for writing to .ozx (RFC-9) format. */
export interface ToOmeZarrOzxOptions {
  /**
   * Optional progress callback invoked after each chunk is written.
   * Reports cumulative progress across all scale levels.
   *
   * @param completedChunks - Number of chunks written so far
   * @param totalChunks - Total number of chunks to write across all levels
   */
  onProgress?:
    | ((completedChunks: number, totalChunks: number) => void)
    | undefined;
  /**
   * Write consolidated metadata into the root `zarr.json` (default `true`).
   * A zipped store is where consolidation pays off most for a reader that
   * understands the block, so leave it on unless the destination rejects it.
   */
  consolidateMetadata?: boolean | undefined;
  /**
   * OME-Zarr version to write the archive at (default `0.5`, and `0.6` for a
   * scene or a transformation). Any version stored in Zarr v3 can be zipped:
   * `0.5`, `0.6` or `0.9.dev1`; a scene or a transformation needs `0.6` or
   * later. The version is recorded in the archive's ZIP comment as well as
   * in its root metadata.
   */
  version?: OzxVersion | undefined;
  /**
   * Store each block of chunks as one Zarr v3 shard: how many chunks a shard
   * spans along each axis -- one count for every axis, one per axis in order,
   * or per axis name (an axis left out spans 1). Omitted, a shard spans 2
   * chunks along every axis, as the Python writer's `.ozx` default. A stored
   * (uncompressed) archive entry can be range-read, so a reader fetches
   * single chunks out of a shard.
   */
  chunksPerShard?: ChunksPerShard | undefined;
  /**
   * Codecs for every array (default: the blosc(zstd) pipeline from
   * `defaultCodecs`); the sharding codec wraps them.
   */
  codecs?: ZarrCodec[] | undefined;
}

/** @deprecated Use {@link ToOmeZarrOzxOptions} instead. */
export type ToNgffZarrOzxOptions = ToOmeZarrOzxOptions;

/**
 * Browser-compatible version of toOmeZarr.
 * Only supports MemoryStore (Map) for writing.
 * Does NOT support local file paths or HTTP URLs (use the full version in Node.js).
 *
 * `multiscales` may also be an {@link NgffScene}: its metadata lands in the
 * root group's `ome.scene` and each of its images is written below its path,
 * after the scene is checked against the spec, at version 0.6 unless given.
 * Or a transformation (a {@link V06Transform}), written as the store's only
 * transformation, at version 0.6 unless given. {@link toOmeZarrOzx} zips
 * any of the three into an `.ozx` archive. One whose transformations
 * reference a stored node, such as a displacement field, is staged in the
 * MemoryStore instead -- the node written below its `path` first, then the
 * scene or transformation with `{ overwrite: false }` -- and the store
 * packed with {@link storeToZip}.
 */
export async function toOmeZarr(
  store: string | MemoryStore | zarr.FetchStore,
  multiscales: NgffMultiscales | NgffScene | V06Transform,
  options: ToOmeZarrOptions = {},
): Promise<void> {
  if (!(store instanceof Map)) {
    if (isV06Transform(multiscales)) {
      throw new Error(
        "A transformation is written to a MemoryStore in the browser.",
      );
    }
    if (multiscales instanceof NgffScene) {
      throw new Error(
        "A scene is written to a MemoryStore in the browser, or zipped into " +
          "an .ozx archive with toOmeZarrOzx().",
      );
    }
    if (store instanceof zarr.FetchStore) {
      throw new Error(
        "FetchStore is read-only and cannot be used for writing. Use MemoryStore instead.",
      );
    }
    if (store.startsWith("http://") || store.startsWith("https://")) {
      throw new Error(
        "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use MemoryStore instead.",
      );
    }
    // Local file paths are not supported in browser environments
    throw new Error(
      "Local file paths are not supported in browser environments. Use MemoryStore instead.",
    );
  }
  const { path, ...rest } = options;
  const target =
    path === undefined
      ? (store as unknown as zarr.Mutable)
      : await writableStoreBelow(store as unknown as zarr.Mutable, path);
  await writeToStore(target, multiscales, arrayWriter, rest);
}

/**
 * The contents of a MemoryStore packed into an RFC-9 `.ozx` archive: what
 * a store staged with {@link toOmeZarr} holds, such as a scene and the
 * displacement field one of its transformations references, written in
 * several calls. The archive records the version the store's root document
 * declares unless `options.version` is given; the root `zarr.json` leads it.
 */
export function storeToZip(
  store: MemoryStore,
  options: StoreToZipOptions = {},
): Uint8Array {
  return storeToZipData(store, options);
}

/** @deprecated Use {@link toOmeZarr} instead. */
export const toNgffZarr = toOmeZarr;

/**
 * Create OME-Zarr .ozx ZIP data (RFC-9) - Browser version.
 *
 * This function creates an OME-Zarr hierarchy in memory and returns
 * the ZIP data as a Uint8Array. This is the browser-compatible version
 * that returns the raw ZIP data for download or further processing.
 *
 * `multiscales` may also be an {@link NgffScene}, zipped with its images
 * below it, or a transformation (a {@link V06Transform}) on its own; either
 * is written at version 0.6 unless given. Arrays are sharded 2 chunks a
 * shard along every axis unless `chunksPerShard` says otherwise, and
 * `onProgress` counts the writes of every image of a scene together. A
 * scene or a transformation that references a stored node by `path`, such
 * as a `displacements` field, cannot be zipped in one piece and is refused.
 *
 * @param multiscales - NgffMultiscales, NgffScene, or V06Transform to write
 * @param options - Options for writing
 * @returns ZIP file data as Uint8Array
 *
 * @see https://ngff.openmicroscopy.org/rfc/9/index.html
 */
export async function toOmeZarrOzx(
  multiscales: NgffMultiscales | NgffScene | V06Transform,
  options: ToOmeZarrOzxOptions = {},
): Promise<Uint8Array> {
  if (isV06Transform(multiscales)) {
    return await transformationToOzx(multiscales, {
      version: options.version,
      consolidateMetadata: options.consolidateMetadata,
    });
  }
  if (multiscales instanceof NgffScene) {
    return await sceneToOzx(multiscales, arrayWriter, options);
  }
  const version = gateOzxVersion(options.version);

  // Create a memory store to hold the zarr data
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();

  // Use the shared write function
  const chunksPerShard = options.chunksPerShard ?? DEFAULT_OZX_CHUNKS_PER_SHARD;
  await writeNgffMultiscalesToMemoryStore(
    memoryStore,
    multiscales,
    arrayWriter({ codecs: options.codecs, chunksPerShard }),
    options.onProgress ?? null,
    options.consolidateMetadata ?? true,
    version,
    chunksPerShard,
  );

  // Convert the memory store to ZIP data; the ZIP comment records the
  // version the root metadata was written at.
  const zipData = memoryStoreToZip(memoryStore, { version });

  return zipData;
}

/** @deprecated Use {@link toOmeZarrOzx} instead. */
export const toNgffZarrOzx = toOmeZarrOzx;
