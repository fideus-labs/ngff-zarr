// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Unified OMERO metadata computation using @fideus-labs/worker-pool.
 *
 * Works in Node.js, Deno, and browser environments. Each zarr chunk is
 * decoded and its per-channel statistics computed in parallel via a
 * WorkerPool. Decoded chunks are cached so that a subsequent zarrGet()
 * with the same cache avoids redundant decompression.
 */

import type { CodecChunkMeta } from "@fideus-labs/fizarrita";
import { createCacheKey, readArrayMetadata } from "@fideus-labs/fizarrita";
import type { WorkerPoolTask } from "@fideus-labs/worker-pool";
import { WorkerPool } from "@fideus-labs/worker-pool";
import type { Array as ZarrArray, DataType, Readable } from "zarrita";

import type { Multiscales } from "../types/multiscales.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import type { Omero } from "../types/zarr_metadata.ts";
import {
  buildOmeroFromAccumulators,
  type ChannelStatisticsAccumulator,
  type ComputeOmeroFromMultiscalesOptions,
  type ComputeOmeroOptions,
  createAccumulator,
  extractChannel,
  mergeAccumulators,
  updateAccumulator,
  validateColor,
  validateQuantiles,
} from "./compute_omero-shared.ts";
import { getMetaId, workerDecodeAndStats } from "./omero_worker_rpc.ts";

// Re-export shared utilities for backward compatibility
export {
  computeChannelStatistics,
  type ComputeOmeroFromMultiscalesOptions,
  type ComputeOmeroOptions,
  extractChannel,
  getDefaultColors,
  GLASBEY_COLORS,
  QUANTILE_SAMPLE_SIZE,
  validateColor,
  validateQuantiles,
} from "./compute_omero-shared.ts";

// fizarrita uses npm:zarrita while this project uses jsr:@zarrita/zarrita.
// The types are structurally identical but TypeScript sees them as
// incompatible due to differing #private class fields. We bridge with
// explicit `as any` casts when calling into fizarrita.
// deno-lint-ignore no-explicit-any
type AnyZarrArray = any;

// ---------------------------------------------------------------------------
// Worker pool for omero computation
// ---------------------------------------------------------------------------

/** Pool size for the omero worker pool. */
const POOL_SIZE = Math.min(
  typeof navigator !== "undefined" ? navigator?.hardwareConcurrency || 4 : 4,
  16,
);

/** Blob URL used by inlined browser builds — set by inline_worker.ts. */
let workerBlobUrl: string | null = null;

let _omeroPool: WorkerPool | null = null;

function getOmeroPool(): WorkerPool {
  if (!_omeroPool) {
    _omeroPool = new WorkerPool(POOL_SIZE);
  }
  return _omeroPool;
}

/** Create a new omero codec worker instance. */
function createOmeroWorker(): Worker {
  return new Worker(
    new URL("../workers/omero_codec_worker.ts", import.meta.url),
    { type: "module" },
  );
}

/**
 * Terminate the omero computation worker pool and release resources.
 * Call this when you're done with OMERO computations to free memory.
 */
export function terminateOmeroWorkerPool(): void {
  if (_omeroPool) {
    _omeroPool.terminateWorkers();
    _omeroPool = null;
  }
  if (workerBlobUrl) {
    URL.revokeObjectURL(workerBlobUrl);
    workerBlobUrl = null;
  }
}

// ---------------------------------------------------------------------------
// Chunk coordinate enumeration
// ---------------------------------------------------------------------------

/**
 * Enumerate all chunk coordinates for a given array shape and chunk shape.
 * Returns an array of coordinate tuples.
 */
function enumerateChunkCoords(
  shape: readonly number[],
  chunkShape: readonly number[],
): number[][] {
  const gridDims = shape.map((s, i) => Math.ceil(s / chunkShape[i]));
  const coords: number[][] = [];
  const ndim = gridDims.length;

  // Recursive enumeration of all grid coordinates
  function enumerate(dim: number, current: number[]): void {
    if (dim === ndim) {
      coords.push([...current]);
      return;
    }
    for (let i = 0; i < gridDims[dim]; i++) {
      current[dim] = i;
      enumerate(dim + 1, current);
    }
  }

  enumerate(0, new Array(ndim).fill(0));
  return coords;
}

// ---------------------------------------------------------------------------
// Shared TextDecoder for metadata reading
// ---------------------------------------------------------------------------

const _textDecoder = new TextDecoder();

// ---------------------------------------------------------------------------
// Codec detection — skip workers for unsupported codecs
// ---------------------------------------------------------------------------

const UNSUPPORTED_CODEC_NAMES = new Set(["sharding_indexed"]);

// deno-lint-ignore no-explicit-any
const _codecCheckCache = new WeakMap<any, Map<string, boolean>>();

async function hasUnsupportedCodecs<Store extends Readable>(
  arr: ZarrArray<DataType, Store>,
): Promise<boolean> {
  let pathMap = _codecCheckCache.get(arr.store);
  if (pathMap?.has(arr.path)) {
    return pathMap.get(arr.path)!;
  }

  const zarrJsonPath = (
    arr.path === "/" ? "/zarr.json" : `${arr.path}/zarr.json`
  ) as `/${string}`;
  const bytes = await arr.store.get(zarrJsonPath);

  let result = false;
  if (bytes) {
    try {
      const meta = JSON.parse(_textDecoder.decode(bytes));
      const codecs: Array<{ name: string }> = meta.codecs ?? [];
      result = codecs.some((c) => UNSUPPORTED_CODEC_NAMES.has(c.name));
    } catch {
      result = false;
    }
  }

  if (!pathMap) {
    pathMap = new Map();
    _codecCheckCache.set(arr.store, pathMap);
  }
  pathMap.set(arr.path, result);

  return result;
}

// ---------------------------------------------------------------------------
// Chunk key encoding (local fallback for when fizarrita metadata isn't usable)
// ---------------------------------------------------------------------------

function createChunkKeyEncoder(
  metadata: {
    chunk_key_encoding?: {
      name: string;
      configuration?: { separator?: string };
    };
    dimension_separator?: string;
  },
  isV2: boolean,
): (chunk_coords: number[]) => string {
  if (isV2) {
    const separator = metadata.dimension_separator ?? ".";
    return (chunk_coords) => chunk_coords.join(separator) || "0";
  }
  const encoding = metadata.chunk_key_encoding ?? { name: "default" };
  if (encoding.name === "default") {
    const separator = encoding.configuration?.separator ?? "/";
    return (chunk_coords) => ["c", ...chunk_coords].join(separator);
  }
  if (encoding.name === "v2") {
    const separator = encoding.configuration?.separator ?? ".";
    return (chunk_coords) => chunk_coords.join(separator) || "0";
  }
  throw new Error(`Unknown chunk key encoding: ${encoding.name}`);
}

// ---------------------------------------------------------------------------
// Read array metadata (local implementation to avoid fizarrita version issues)
// ---------------------------------------------------------------------------

interface LocalArrayMetadata {
  codecMeta: CodecChunkMeta;
  encodeChunkKey: (chunk_coords: number[]) => string;
}

async function readLocalArrayMetadata<Store extends Readable>(
  arr: ZarrArray<DataType, Store>,
): Promise<LocalArrayMetadata> {
  const store = arr.store;

  // Try v3 first: read zarr.json
  const v3Path = (
    arr.path === "/" ? "/zarr.json" : `${arr.path}/zarr.json`
  ) as `/${string}`;
  const v3Bytes = await store.get(v3Path);
  if (v3Bytes) {
    const metadata = JSON.parse(_textDecoder.decode(v3Bytes));
    return {
      codecMeta: {
        data_type: metadata.data_type,
        chunk_shape: metadata.chunk_grid.configuration.chunk_shape,
        codecs: metadata.codecs,
      },
      encodeChunkKey: createChunkKeyEncoder(metadata, false),
    };
  }

  // Try v2: read .zarray
  const v2Path = (
    arr.path === "/" ? "/.zarray" : `${arr.path}/.zarray`
  ) as `/${string}`;
  const v2Bytes = await store.get(v2Path);
  if (v2Bytes) {
    const metadata = JSON.parse(_textDecoder.decode(v2Bytes));
    const codecs: Array<{
      name: string;
      configuration: Record<string, unknown>;
    }> = [];
    if (metadata.order === "F") {
      codecs.push({ name: "transpose", configuration: { order: "F" } });
    }
    if (metadata.compressor) {
      const { id, ...configuration } = metadata.compressor;
      codecs.push({ name: id, configuration });
    }
    for (const { id, ...configuration } of metadata.filters ?? []) {
      codecs.push({ name: id, configuration });
    }
    return {
      codecMeta: {
        data_type: arr.dtype,
        chunk_shape: arr.chunks,
        codecs: codecs.length > 0
          ? codecs
          : [{ name: "bytes", configuration: { endian: "little" } }],
      },
      encodeChunkKey: createChunkKeyEncoder(metadata, true),
    };
  }

  // Fallback
  return {
    codecMeta: {
      data_type: arr.dtype,
      chunk_shape: arr.chunks,
      codecs: [{ name: "bytes", configuration: { endian: "little" } }],
    },
    encodeChunkKey: createChunkKeyEncoder(
      {
        chunk_key_encoding: { name: "default" },
      },
      false,
    ),
  };
}

// ---------------------------------------------------------------------------
// Main-thread fallback for stats computation from a cached/decoded chunk
// ---------------------------------------------------------------------------

function computeStatsFromDecodedChunk(
  chunkData: ArrayLike<number>,
  chunkShape: number[],
  nChannels: number,
  cIndex: number,
): ChannelStatisticsAccumulator[] {
  const accumulators: ChannelStatisticsAccumulator[] = [];
  if (cIndex >= 0) {
    for (let ch = 0; ch < nChannels; ch++) {
      const channelData = extractChannel(chunkData, chunkShape, ch, cIndex);
      const acc = createAccumulator();
      updateAccumulator(acc, channelData);
      accumulators.push(acc);
    }
  } else {
    const acc = createAccumulator();
    updateAccumulator(acc, chunkData);
    accumulators.push(acc);
  }
  return accumulators;
}

// ---------------------------------------------------------------------------
// Main-thread fallback for full computation (unsupported codecs)
// ---------------------------------------------------------------------------

async function computeOmeroMainThread(
  image: NgffImage,
  options: ComputeOmeroOptions = {},
): Promise<Omero> {
  // Lazy import to avoid circular dependency
  const { zarrGet } = await import("./worker_pool.ts");

  const quantiles = options.quantiles ?? [0.02, 0.98];
  validateQuantiles(quantiles as [number, number]);

  const dims = image.dims;
  const shape = image.data.shape;
  const cIndex = dims.indexOf("c");
  const hasChannelDim = cIndex !== -1;
  const nChannels = hasChannelDim ? shape[cIndex] : 1;

  const result = await zarrGet(image.data);
  const fullData = result.data as ArrayLike<number>;

  const accumulators = Array.from(
    { length: nChannels },
    () => createAccumulator(),
  );

  if (hasChannelDim) {
    for (let chIdx = 0; chIdx < nChannels; chIdx++) {
      const channelData = extractChannel(fullData, shape, chIdx, cIndex);
      updateAccumulator(accumulators[chIdx], channelData);
    }
  } else {
    updateAccumulator(accumulators[0], fullData);
  }

  return buildOmeroFromAccumulators(
    accumulators,
    quantiles as [number, number],
    options.colors,
    options.labels,
  );
}

// ---------------------------------------------------------------------------
// Public API: computeOmeroFromNgffImage
// ---------------------------------------------------------------------------

/**
 * Compute OMERO metadata from an NgffImage.
 *
 * Each zarr chunk is decoded and its per-channel statistics computed in
 * parallel via a WorkerPool. Decoded chunks are optionally cached so that
 * a subsequent zarrGet() with the same cache avoids redundant decompression.
 *
 * @param image - The NgffImage to compute metadata for
 * @param options - Optional configuration for quantiles, colors, labels, and cache
 * @returns Promise resolving to Omero metadata with computed window parameters
 */
export async function computeOmeroFromNgffImage(
  image: NgffImage,
  options: ComputeOmeroOptions = {},
): Promise<Omero> {
  const quantiles = options.quantiles ?? [0.02, 0.98];
  validateQuantiles(quantiles as [number, number]);

  const dims = image.dims;
  const shape = image.data.shape;
  const cIndex = dims.indexOf("c");
  const hasChannelDim = cIndex !== -1;
  const nChannels = hasChannelDim ? shape[cIndex] : 1;

  // Validate colors
  if (options.colors !== undefined) {
    if (options.colors.length < nChannels) {
      throw new Error(
        `Not enough colors provided. Got ${options.colors.length}, need ${nChannels}.`,
      );
    }
    for (const color of options.colors.slice(0, nChannels)) {
      validateColor(color);
    }
  }

  // Validate labels
  if (options.labels !== undefined) {
    if (options.labels.length < nChannels) {
      throw new Error(
        `Not enough labels provided. Got ${options.labels.length}, need ${nChannels}.`,
      );
    }
  }

  // Fall back to main-thread for unsupported codecs (e.g. sharding)
  if (await hasUnsupportedCodecs(image.data)) {
    return computeOmeroMainThread(image, options);
  }

  // Read array metadata for codec pipeline + chunk key encoding
  let codecMeta: CodecChunkMeta;
  let encodeChunkKey: (chunk_coords: number[]) => string;
  try {
    const metadata = await readArrayMetadata(image.data as AnyZarrArray);
    codecMeta = metadata.codecMeta;
    encodeChunkKey = metadata.encodeChunkKey;
  } catch {
    // Fall back to local metadata reading if fizarrita's version fails
    const localMeta = await readLocalArrayMetadata(image.data);
    codecMeta = localMeta.codecMeta;
    encodeChunkKey = localMeta.encodeChunkKey;
  }

  const chunkShape = codecMeta.chunk_shape;
  const metaId = getMetaId(codecMeta);

  // Optional cache for decoded chunks
  const cache = options.cache;

  // Enumerate all chunk coordinates
  const allChunkCoords = enumerateChunkCoords(shape, chunkShape);

  // Build worker pool tasks — one per chunk
  const pool = getOmeroPool();
  const tasks: WorkerPoolTask<ChannelStatisticsAccumulator[]>[] = [];

  for (const chunk_coords of allChunkCoords) {
    const chunkKey = encodeChunkKey(chunk_coords);
    const chunkPath = image.data.resolve(chunkKey).path as `/${string}`;

    // Compute edge chunk shape
    const edgeChunkShape = chunk_coords.map((coord, dim) =>
      Math.min(chunkShape[dim], shape[dim] - coord * chunkShape[dim])
    );

    // Check cache before building the task
    const cacheKey = cache
      ? createCacheKey(image.data as AnyZarrArray, encodeChunkKey, chunk_coords)
      : "";
    if (cache) {
      const cachedChunk = cache.get(cacheKey);
      if (cachedChunk) {
        // Cache hit — compute stats from cached decoded chunk on main thread
        const accs = computeStatsFromDecodedChunk(
          cachedChunk.data as unknown as ArrayLike<number>,
          cachedChunk.shape,
          nChannels,
          cIndex,
        );
        // No worker needed — return the slot directly for pool management.
        // For cached results, we use the provided workerSlot if available,
        // or create a worker to satisfy the type contract.
        tasks.push((workerSlot: Worker | null) => {
          const worker = workerSlot ?? createOmeroWorker();
          return Promise.resolve({
            worker,
            result: accs,
          });
        });
        continue;
      }
    }

    tasks.push(async (workerSlot: Worker | null) => {
      const worker = workerSlot ?? createOmeroWorker();

      // Fetch raw bytes from store on main thread
      const rawBytes = await image.data.store.get(chunkPath);

      if (!rawBytes) {
        // Missing chunk — fill with zeros, compute trivial stats
        const accs: ChannelStatisticsAccumulator[] = [];
        for (let ch = 0; ch < nChannels; ch++) {
          accs.push(createAccumulator());
        }
        return { worker, result: accs };
      }

      // Send to worker: decode + compute stats in one round-trip
      const { chunk, accumulators } = await workerDecodeAndStats(
        worker,
        rawBytes,
        metaId,
        codecMeta,
        nChannels,
        cIndex,
        edgeChunkShape,
      );

      // Cache decoded chunk for future zarrGet reuse.
      // Cast needed: jsr:zarrita vs npm:zarrita Chunk types are structurally
      // identical but TypeScript sees differing #private class fields.
      if (cache) {
        // deno-lint-ignore no-explicit-any
        cache.set(cacheKey, chunk as any);
      }

      return { worker, result: accumulators };
    });
  }

  // Execute all tasks with bounded concurrency
  let allAccumulators: ChannelStatisticsAccumulator[][];
  if (tasks.length > 0) {
    const { promise } = pool.runTasks(tasks, options.onProgress ?? null);
    allAccumulators = await promise;
  } else {
    allAccumulators = [];
  }

  // Merge partial accumulators across all chunks
  const mergedAccumulators: ChannelStatisticsAccumulator[] = Array.from(
    { length: nChannels },
    () => createAccumulator(),
  );

  for (const chunkAccs of allAccumulators) {
    for (let ch = 0; ch < nChannels; ch++) {
      if (chunkAccs[ch]) {
        mergedAccumulators[ch] = mergeAccumulators(
          mergedAccumulators[ch],
          chunkAccs[ch],
        );
      }
    }
  }

  return buildOmeroFromAccumulators(
    mergedAccumulators,
    quantiles as [number, number],
    options.colors,
    options.labels,
  );
}

// ---------------------------------------------------------------------------
// Public API: computeOmeroFromMultiscales
// ---------------------------------------------------------------------------

/**
 * Compute OMERO metadata from a Multiscales object.
 *
 * This is a convenience function that computes OMERO metadata from the
 * highest resolution image in a multiscales pyramid for accurate statistics.
 *
 * @param multiscales - The Multiscales object to compute metadata for
 * @param options - Optional configuration for quantiles, colors, and labels
 * @returns Promise resolving to Omero metadata with computed window parameters
 */
export async function computeOmeroFromMultiscales(
  multiscales: Multiscales,
  options: ComputeOmeroFromMultiscalesOptions = {},
): Promise<Omero> {
  if (!multiscales.images || multiscales.images.length === 0) {
    throw new Error("Multiscales has no images");
  }

  // Always use the highest resolution (first) image for accurate statistics
  const image = multiscales.images[0];

  const computeOptions: ComputeOmeroOptions = {};
  if (options.quantiles !== undefined) {
    computeOptions.quantiles = options.quantiles;
  }
  if (options.colors !== undefined) {
    computeOptions.colors = options.colors;
  }
  if (options.labels !== undefined) {
    computeOptions.labels = options.labels;
  }
  if (options.onProgress !== undefined) {
    computeOptions.onProgress = options.onProgress;
  }

  return await computeOmeroFromNgffImage(image, computeOptions);
}
