// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Zarr v3 sharding for the OME-Zarr writers.
 *
 * A sharded array stores each block of inner chunks as one store object, a
 * *shard*, with an index locating the chunks inside it: the chunk shape a
 * viewer reads stays small without the store filling up with tiny objects.
 *
 * The layout follows the Python writer (`_configure_sharding` and the shard
 * clamp in `ngff_zarr/to_ngff_zarr.py`), so the two write the same grid:
 * `chunksPerShard` multiplies the chunk shape per axis, and a shard is held
 * to the fewest whole inner chunks that cover its axis, which keeps the inner
 * chunk dividing it.
 *
 * `zarrSet` (fizarrita) encodes shards; zarrita's own `set` cannot. It writes
 * a shard whole, and two writes into one shard at once race, so a writer
 * plans its writes on {@link ArrayLayout.chunkShape} -- one per shard.
 */

import * as zarr from "zarrita";

import type { ZarrCodec } from "./codecs.ts";

/**
 * How many chunks a shard spans along each axis: one count for every axis,
 * one per axis in order, or per axis name (an axis left out spans 1).
 */
export type ChunksPerShard = number | number[] | Record<string, number>;

/** The chunk grid and codecs an array is created with. */
export interface ArrayLayout {
  /**
   * The array's chunk grid: the shard shape when sharded, else the chunk
   * shape. It is the unit of store writes.
   */
  chunkShape: number[];
  /**
   * The array's codecs: when sharded, the chunk codecs inside a
   * `sharding_indexed` codec.
   */
  codecs: ZarrCodec[];
}

/** The shard index codecs zarr-python and zarrs write by default. */
const SHARD_INDEX_CODECS = [
  { name: "bytes", configuration: { endian: "little" } },
  { name: "crc32c" },
];

/**
 * Refuse sharding at a version stored without it: OME-Zarr 0.4 is Zarr v2,
 * which has no sharding. The same check, and message, as the Python writer.
 */
export function gateShardingVersion(
  version: string,
  chunksPerShard: ChunksPerShard | undefined,
): void {
  if (chunksPerShard !== undefined && version === "0.4") {
    throw new Error(
      "Sharding is only supported for OME-Zarr version 0.5 and later",
    );
  }
}

function shardMultipliers(
  chunksPerShard: ChunksPerShard,
  dims: readonly string[],
): number[] {
  let multipliers: number[];
  if (typeof chunksPerShard === "number") {
    multipliers = dims.map(() => chunksPerShard);
  } else if (Array.isArray(chunksPerShard)) {
    if (chunksPerShard.length !== dims.length) {
      throw new Error(
        `chunksPerShard must have length ${dims.length}, one entry per ` +
          `axis, got ${chunksPerShard.length}`,
      );
    }
    multipliers = chunksPerShard;
  } else {
    // A name the image lacks is a typo that would leave its axis unsharded.
    const unknown = Object.keys(chunksPerShard).filter(
      (dim) => !dims.includes(dim),
    );
    if (unknown.length > 0) {
      throw new Error(
        `chunksPerShard names axes the image does not have: ` +
          `${unknown.join(", ")} (axes: ${dims.join(", ")})`,
      );
    }
    multipliers = dims.map((dim) => chunksPerShard[dim] ?? 1);
  }
  for (const multiplier of multipliers) {
    if (!Number.isInteger(multiplier) || multiplier < 1) {
      throw new Error(
        `chunksPerShard entries must be positive integers, got ${multiplier}`,
      );
    }
  }
  return multipliers;
}

/**
 * The chunk grid an array is written on: its shards when `chunksPerShard` is
 * given, else `chunkShape` itself.
 *
 * @param shape - The array shape
 * @param dims - The axis names, for a per-name `chunksPerShard`
 * @param chunkShape - The (inner) chunk shape
 * @param chunksPerShard - Chunks per shard, or `undefined` for no sharding
 */
export function outerChunkShape(
  shape: readonly number[],
  dims: readonly string[],
  chunkShape: number[],
  chunksPerShard: ChunksPerShard | undefined,
): number[] {
  if (chunksPerShard === undefined) {
    return chunkShape;
  }
  const multipliers = shardMultipliers(chunksPerShard, dims);
  // A shard may run past its axis, as a chunk may, but by less than a chunk:
  // the grid is never larger than the array needs.
  return chunkShape.map((chunk, i) =>
    Math.min(
      chunk * multipliers[i],
      Math.max(1, Math.ceil(shape[i] / chunk)) * chunk,
    )
  );
}

/**
 * The chunk grid and codecs to create an array with, sharded when
 * `chunksPerShard` is given.
 *
 * @param shape - The array shape
 * @param dims - The axis names, for a per-name `chunksPerShard`
 * @param chunkShape - The (inner) chunk shape
 * @param codecs - The chunk codecs
 * @param chunksPerShard - Chunks per shard, or `undefined` for no sharding
 */
export function arrayLayout(
  shape: readonly number[],
  dims: readonly string[],
  chunkShape: number[],
  codecs: ZarrCodec[],
  chunksPerShard: ChunksPerShard | undefined,
): ArrayLayout {
  if (chunksPerShard === undefined) {
    return { chunkShape, codecs };
  }
  return {
    chunkShape: outerChunkShape(shape, dims, chunkShape, chunksPerShard),
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: chunkShape,
          codecs,
          index_codecs: SHARD_INDEX_CODECS,
          index_location: "end",
        },
      },
    ],
  };
}

/** Range reads sliced out of a store's whole-object `get`. */
const withRangeReads = zarr.defineStoreExtension((store) => ({
  async getRange(
    key: zarr.AbsolutePath,
    range: zarr.RangeQuery,
    options?: Parameters<zarr.AsyncReadable["get"]>[1],
  ): Promise<Uint8Array | undefined> {
    const bytes = await store.get(key, options);
    if (bytes === undefined) {
      return undefined;
    }
    return "suffixLength" in range
      ? bytes.subarray(Math.max(0, bytes.length - range.suffixLength))
      : bytes.subarray(range.offset, range.offset + range.length);
  },
}));

/**
 * `store`, able to hold sharded arrays.
 *
 * zarrita opens a sharded array only on a store with `getRange`, which it
 * reads a shard's index and single chunks out of -- creating one included.
 * An in-memory `Map` has none; for it, and any other store without one, range
 * reads are added as slices of the whole object. A store with its own is
 * returned untouched.
 *
 * The result is still the store it wraps: a `Map` stays `instanceof Map`, and
 * what is written through it lands in the caller's `Map`.
 */
export function ensureRangeReads<S extends object>(store: S): S {
  if (typeof (store as { getRange?: unknown }).getRange === "function") {
    return store;
  }
  return withRangeReads(store as unknown as zarr.AsyncReadable) as unknown as S;
}
