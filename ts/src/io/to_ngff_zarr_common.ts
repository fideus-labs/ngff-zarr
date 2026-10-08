// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * What the Node and browser `toOmeZarr` do once they hold a writable store:
 * write an image, a scene, or a transformation to it. Each writer resolves
 * its own targets -- a directory path in Node, a `MemoryStore` in both, the
 * group below a `path` of either -- and hands over its array writer. No
 * Node.js dependency.
 */
import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import { NgffScene } from "../types/scene.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import {
  type ConsolidatableStore,
  consolidateMetadata,
} from "../utils/consolidate_metadata.ts";
import {
  type ChunksPerShard,
  ensureRangeReads,
  gateShardingVersion,
} from "../utils/sharding.ts";
import { writeSceneToStore } from "./scene_common.ts";
import {
  isV06Transform,
  writeTransformation,
} from "./transformation_common.ts";
import {
  type ArrayWriterFactory,
  type ProgressCallback,
  writeMultiscalesGroup,
} from "./to_ngff_zarr_ozx_common.ts";

/** The options of `toOmeZarr` that apply once the store is resolved. */
export interface StoreWriteOptions {
  overwrite?: boolean;
  version?: "0.4" | "0.5" | "0.6" | "0.9.dev1";
  chunksPerShard?: ChunksPerShard;
  codecs?: ZarrCodec[];
  consolidateMetadata?: boolean;
  onProgress?: ProgressCallback;
}

/**
 * Write `data` to `store` as `toOmeZarr` does: a transformation as the
 * store's only transformation and a scene with its images, each at version
 * 0.6 unless given, or a multiscales image at version 0.5 unless given, its
 * metadata consolidated at the store's root. `arrayWriter` writes the arrays
 * the way the calling writer does.
 */
export async function writeToStore(
  store: zarr.Mutable,
  data: NgffMultiscales | NgffScene | V06Transform,
  arrayWriter: ArrayWriterFactory,
  options: StoreWriteOptions = {},
): Promise<void> {
  if (isV06Transform(data)) {
    await writeTransformation(store, data, {
      version: options.version ?? "0.6",
      ...(options.overwrite !== undefined && { overwrite: options.overwrite }),
    });
    return;
  }
  if (data instanceof NgffScene) {
    await writeSceneToStore(store, data, arrayWriter, {
      ...options,
      version: options.version ?? "0.6",
    });
    return;
  }

  const version = options.version ?? "0.5";
  gateShardingVersion(version, options.chunksPerShard);
  try {
    // A sharded array needs range reads, which an in-memory `Map` lacks.
    const root = zarr.root(
      options.chunksPerShard === undefined ? store : ensureRangeReads(store),
    );

    // The version-specific root document (v0.6 RFC-5 coordinate systems,
    // v0.5 `ome`-wrapped axes, or bare v0.4 multiscales), its matrix arrays,
    // and every level, as the `.ozx` writers write them.
    const nodePaths = await writeMultiscalesGroup(
      root,
      data,
      arrayWriter({
        codecs: options.codecs,
        chunksPerShard: options.chunksPerShard,
      }),
      version,
      options.onProgress ?? null,
      options.chunksPerShard,
    );

    // Consolidate last: the block inlines the array documents, so it has to be
    // written after them. The root document was replaced wholesale, so a
    // store that was consolidated before this call and is written with
    // `consolidateMetadata: false` is left unconsolidated rather than stale --
    // the Zarr v3 behavior the Python writer relies on too.
    if (options.consolidateMetadata ?? true) {
      await consolidateMetadata(
        store as unknown as ConsolidatableStore,
        nodePaths,
      );
    }
  } catch (error) {
    throw new Error(
      `Failed to write OME-Zarr: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}
