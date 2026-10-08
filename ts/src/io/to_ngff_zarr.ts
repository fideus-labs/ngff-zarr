// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import { NgffScene } from "../types/scene.ts";
import { localStore } from "./scene.ts";
import { sceneToOzx } from "./scene_common.ts";
import {
  isV06Transform,
  transformationToOzx,
} from "./transformation_common.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { arrayLayout, type ChunksPerShard } from "../utils/sharding.ts";
import { writableStoreBelow } from "../utils/store_below.ts";
import { createWriteQueue, zarrGet, zarrSet } from "../utils/worker_pool.ts";
import type { MemoryStore } from "./from_ngff_zarr.ts";
import { isOzxPath, memoryStoreToZip } from "./rfc9_zip.ts";
import { writeToStore } from "./to_ngff_zarr_common.ts";
import {
  type ArrayWriterFactory,
  DEFAULT_OZX_CHUNKS_PER_SHARD,
  gateOzxVersion,
  type OzxVersion,
  type ProgressCallback,
  storeToZipData,
  type StoreToZipOptions,
  writeNgffMultiscalesToMemoryStore,
} from "./to_ngff_zarr_ozx_common.ts";

export type { StoreToZipOptions } from "./to_ngff_zarr_ozx_common.ts";

export interface ToOmeZarrOptions {
  overwrite?: boolean;
  /**
   * OME-Zarr version to write. Defaults to "0.5", and to "0.6" for a scene
   * or a transformation, which that version introduced.
   * Version "0.6" writes RFC 5 coordinate systems and transformations.
   * For .ozx files (RFC-9), a version stored in Zarr v3 is required: 0.5,
   * 0.6 or 0.9.dev1. Version 0.4 (Zarr v2) cannot be zipped and throws.
   */
  version?: "0.4" | "0.5" | "0.6" | "0.9.dev1";
  /**
   * Store each block of chunks as one Zarr v3 shard: how many chunks a shard
   * spans along each axis -- one count for every axis, one per axis in order,
   * or per axis name (an axis left out spans 1). Requires version 0.5 or
   * later. Omitted, a directory or in-memory store is not sharded, while an
   * `.ozx` path is sharded 2 chunks a shard along every axis, as the Python
   * writer does.
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
   * or a directory before the scene or the transformation is written with
   * `{ overwrite: false }`; {@link storeToZip} then packs the store into an
   * `.ozx` archive. Not available for an `.ozx` path, which is written in
   * one piece.
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
}

/** @deprecated Use {@link ToOmeZarrOzxOptions} instead. */
export type ToNgffZarrOzxOptions = ToOmeZarrOzxOptions;

/**
 * Write multiscales data to an OME-Zarr store.
 *
 * This function automatically detects .ozx paths (RFC-9 zipped OME-Zarr format)
 * and handles them appropriately. For .ozx files:
 * - Version 0.5 is used when the version option is omitted (undefined)
 * - Any version stored in Zarr v3 can be requested: 0.5, 0.6 or 0.9.dev1
 * - Version 0.4 (Zarr v2) cannot be zipped; requesting it throws
 *
 * @param store - File path, MemoryStore, or FetchStore to write to
 * @param multiscales - NgffMultiscales data to write, or an {@link NgffScene}:
 *   its metadata lands in the root group's `ome.scene` and each of its images
 *   is written below its path, after the scene is checked against the spec,
 *   at version 0.6 unless given. Or a transformation (a {@link V06Transform}):
 *   it lands in the root group's `ome.coordinateTransformations` as the
 *   store's only transformation, at version 0.6 unless given. A node either
 *   references by `path`, such as a displacement field, is written below the
 *   store first (the `path` option), and the scene or transformation with
 *   `overwrite: false`; for an `.ozx` archive, which is written in one
 *   piece, the store is staged that way and packed with {@link storeToZip}.
 * @param options - Writing options
 *
 * @example
 * ```typescript
 * // Writing to .ozx file - version 0.5 is used when none is given
 * await toOmeZarr("output.ozx", multiscales);
 *
 * // Writing a v0.6 .ozx file
 * await toOmeZarr("output.ozx", multiscales, { version: "0.6" });
 *
 * // Writing to regular zarr with version 0.5 (default)
 * await toOmeZarr("output.zarr", multiscales);
 *
 * // Writing to regular zarr with version 0.4
 * await toOmeZarr("output.zarr", multiscales, { version: "0.4" });
 * ```
 */
export async function toOmeZarr(
  store: string | MemoryStore | zarr.FetchStore,
  multiscales: NgffMultiscales | NgffScene | V06Transform,
  options: ToOmeZarrOptions = {},
): Promise<void> {
  const { path, ...rest } = options;
  if (store instanceof zarr.FetchStore) {
    throw new Error(
      "FetchStore is read-only and cannot be used for writing. Use a local file path or MemoryStore instead.",
    );
  }
  if (
    typeof store === "string" &&
    (store.startsWith("http://") || store.startsWith("https://"))
  ) {
    throw new Error(
      "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use a local file path instead.",
    );
  }

  // Handle .ozx paths (RFC-9)
  if (typeof store === "string" && isOzxPath(store)) {
    if (path !== undefined) {
      throw new Error(
        "An .ozx archive is written in one piece, so nothing is written " +
          "below a path of it. Stage the store in a MemoryStore or a " +
          "directory, writing each part with toOmeZarr(store, part, " +
          "{ path }), then pack it with storeToZip().",
      );
    }
    // RFC-9 is defined on Zarr v3, so any version stored in Zarr v3 can be
    // zipped; 0.4 (Zarr v2) is refused. Omitted, the version defaults to 0.5,
    // or to 0.6 for a scene or a transformation.
    await toOmeZarrOzx(store, multiscales, {
      consolidateMetadata: options.consolidateMetadata,
      version: options.version === undefined
        ? undefined
        : gateOzxVersion(options.version),
      chunksPerShard: options.chunksPerShard,
      onProgress: options.onProgress,
    });
    return;
  }

  const resolved = store instanceof Map
    ? store as unknown as zarr.Mutable
    : await localStore(store);
  const target = path === undefined
    ? resolved
    : await writableStoreBelow(resolved, path);
  await writeToStore(target, multiscales, arrayWriter, rest);
}

/** @deprecated Use {@link toOmeZarr} instead. */
export const toNgffZarr = toOmeZarr;

/** Writes arrays the way this module does, with the given codecs and sharding. */
const arrayWriter: ArrayWriterFactory =
  ({ codecs, chunksPerShard }) => (group, image, path, onProgress) =>
    _writeImage(group, image, path, onProgress, codecs, chunksPerShard);

function _convertDtypeToZarrType(dtype: string): zarr.DataType {
  // Map common numpy/LazyArray dtypes to zarrita data types
  const dtypeMap: Record<string, zarr.DataType> = {
    int8: "int8",
    int16: "int16",
    int32: "int32",
    int64: "int64",
    uint8: "uint8",
    uint16: "uint16",
    uint32: "uint32",
    uint64: "uint64",
    float32: "float32",
    float64: "float64",
    bool: "bool",
    // Handle some alternative formats
    i1: "int8",
    i2: "int16",
    i4: "int32",
    i8: "int64",
    u1: "uint8",
    u2: "uint16",
    u4: "uint32",
    u8: "uint64",
    f4: "float32",
    f8: "float64",
  };

  if (dtype in dtypeMap) {
    return dtypeMap[dtype];
  } else {
    throw new Error(`Unsupported data type: ${dtype}`);
  }
}

type TypedArrayConstructor =
  | Uint8ArrayConstructor
  | Int8ArrayConstructor
  | Uint16ArrayConstructor
  | Int16ArrayConstructor
  | Uint32ArrayConstructor
  | Int32ArrayConstructor
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | BigInt64ArrayConstructor
  | BigUint64ArrayConstructor;

function getTypedArrayConstructor(dtype: zarr.DataType): TypedArrayConstructor {
  // Map zarrita data types to TypedArray constructors
  const constructorMap: Partial<Record<zarr.DataType, TypedArrayConstructor>> =
    {
      int8: Int8Array,
      int16: Int16Array,
      int32: Int32Array,
      int64: BigInt64Array,
      uint8: Uint8Array,
      uint16: Uint16Array,
      uint32: Uint32Array,
      uint64: BigUint64Array,
      float32: Float32Array,
      float64: Float64Array,
      bool: Uint8Array, // Use Uint8Array for boolean, where 0 represents false and 1 represents true
      // Note: float16 and "v2:object" not supported by standard TypedArrays
    };

  const constructor = constructorMap[dtype];
  if (constructor) {
    return constructor;
  } else {
    throw new Error(`Unsupported data type for typed array: ${dtype}`);
  }
}

async function _writeImage(
  group: zarr.Group<MemoryStore>,
  image: NgffImage,
  arrayPath: string,
  onProgress?: ((completedChunks: number, totalChunks: number) => void) | null,
  codecs?: ZarrCodec[],
  chunksPerShard?: ChunksPerShard,
): Promise<void> {
  try {
    const chunks = getChunksFromImage(image);

    // Convert LazyArray dtype to zarrita DataType
    const zarrDataType = _convertDtypeToZarrType(image.data.dtype);

    // Create array location
    const arrayLocation = group.resolve(arrayPath);

    // The chunk grid and codecs, wrapped in a shard when requested
    const layout = arrayLayout(
      image.data.shape,
      image.dims,
      chunks,
      codecs ?? defaultCodecs(zarrDataType),
      chunksPerShard,
    );

    // Create the zarr array with proper configuration
    const zarrArray = await zarr.create(arrayLocation, {
      shape: image.data.shape,
      dtype: zarrDataType,
      chunkShape: layout.chunkShape,
      fillValue: 0,
      codecs: layout.codecs,
    });

    await _writeArrayData(
      zarrArray as zarr.Array<zarr.DataType, MemoryStore>,
      image,
      onProgress ?? null,
      layout.chunkShape,
    );
  } catch (error) {
    throw new Error(
      `Failed to write image array: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

function getChunksFromImage(image: NgffImage): number[] {
  // zarr.Array.chunks is a number[] representing chunk shape
  if (image.data.chunks && image.data.chunks.length > 0) {
    return image.data.chunks;
  }

  return image.data.shape.map((dim: number) => Math.min(dim, 1024));
}

async function _writeArrayData(
  zarrArray: zarr.Array<zarr.DataType, MemoryStore>,
  image: NgffImage,
  onProgress: ((completedChunks: number, totalChunks: number) => void) | null,
  writeShape: number[],
): Promise<void> {
  try {
    // Get array shape for chunk calculation - we don't need the full data here
    const shape = image.data.shape;

    // Calculate chunk indices for parallel writing. The grid is the array's
    // outer one: for a sharded array each write is a whole shard, since two
    // writes into one shard at once would race.
    const chunkIndices = calculateChunkIndices(shape, writeShape);

    // Create a queue for parallel chunk writing
    const writeQueue = createWriteQueue();

    // Queue all chunks for writing
    for (const chunkIndex of chunkIndices) {
      writeQueue.add(async () => {
        await writeChunkWithGet(zarrArray, image, chunkIndex, writeShape);
      });
    }

    // Wait for all chunks to be written
    await writeQueue.onIdle(onProgress);
  } catch (error) {
    throw new Error(
      `Failed to write array data: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

async function writeChunkWithGet(
  zarrArray: zarr.Array<zarr.DataType, MemoryStore>,
  image: NgffImage,
  chunkIndex: number[],
  writeShape: number[],
): Promise<void> {
  // Calculate the chunk bounds
  const shape = image.data.shape;
  const chunkStart = chunkIndex.map((idx, dim) => idx * writeShape[dim]);
  const chunkEnd = chunkStart.map((start, dim) =>
    Math.min(start + writeShape[dim], shape[dim])
  );

  // Calculate chunk shape
  const chunkShape = chunkEnd.map((end, dim) => end - chunkStart[dim]);

  // Create selection for this chunk from the source data
  const sourceSelection = chunkStart.map((start, dim) =>
    zarr.slice(start, chunkEnd[dim])
  );

  // Get only the chunk data we need from the source
  const { data: chunkSourceData } = await zarrGet(image.data, sourceSelection);

  // Convert chunk data to target type
  const targetTypedArrayConstructor = getTypedArrayConstructor(zarrArray.dtype);
  const chunkTargetData = convertChunkToTargetType(
    chunkSourceData as ArrayBufferView,
    zarrArray.dtype,
    targetTypedArrayConstructor,
  );

  // Validate chunk data size
  const expectedSize = chunkShape.reduce((a, b) => a * b, 1);
  const actualSize = chunkTargetData.byteLength /
    ((chunkTargetData as Uint8Array | Uint16Array | Int16Array | Float32Array)
      .BYTES_PER_ELEMENT || 1);
  if (actualSize !== expectedSize) {
    console.error(`[writeChunkWithGet] Chunk data size mismatch!`);
    console.error(`  Image shape:`, shape);
    console.error(`  Chunk index:`, chunkIndex);
    console.error(`  Chunk start:`, chunkStart);
    console.error(`  Chunk end:`, chunkEnd);
    console.error(`  Chunk shape:`, chunkShape);
    console.error(`  Expected size:`, expectedSize);
    console.error(`  Actual size:`, actualSize);
    throw new Error(
      `Chunk data size mismatch: expected ${expectedSize} elements, got ${actualSize}`,
    );
  }

  // Create the selection for writing to the target zarr array
  const targetSelection = chunkStart.map((start, dim) =>
    zarr.slice(start, chunkEnd[dim])
  );

  // Write the chunk using zarrita's set function
  await zarrSet(zarrArray, targetSelection, {
    data: chunkTargetData,
    shape: chunkShape,
    stride: calculateChunkStride(chunkShape),
  });
}

function convertChunkToTargetType(
  chunkData: ArrayBufferView,
  targetDtype: zarr.DataType,
  targetTypedArrayConstructor: TypedArrayConstructor,
): ArrayBufferView {
  // Handle different source data types
  if (
    chunkData instanceof BigInt64Array ||
    chunkData instanceof BigUint64Array
  ) {
    // Handle BigInt arrays separately
    if (chunkData.constructor === targetTypedArrayConstructor) {
      return chunkData as ArrayBufferView;
    } else if (targetDtype === "int64" || targetDtype === "uint64") {
      // BigInt to BigInt conversion
      const bigIntArray = new targetTypedArrayConstructor(chunkData.length) as
        | BigInt64Array
        | BigUint64Array;
      for (let i = 0; i < chunkData.length; i++) {
        bigIntArray[i] = chunkData[i];
      }
      return bigIntArray;
    } else {
      // BigInt to regular number conversion
      const numberArray = new targetTypedArrayConstructor(chunkData.length) as
        | Uint8Array
        | Int8Array
        | Uint16Array
        | Int16Array
        | Uint32Array
        | Int32Array
        | Float32Array
        | Float64Array;
      for (let i = 0; i < chunkData.length; i++) {
        numberArray[i] = Number(chunkData[i]);
      }
      return numberArray;
    }
  } else if (
    chunkData instanceof Uint8Array ||
    chunkData instanceof Int8Array ||
    chunkData instanceof Uint16Array ||
    chunkData instanceof Int16Array ||
    chunkData instanceof Uint32Array ||
    chunkData instanceof Int32Array ||
    chunkData instanceof Float32Array ||
    chunkData instanceof Float64Array
  ) {
    // Handle regular typed arrays
    if (chunkData.constructor === targetTypedArrayConstructor) {
      return chunkData as ArrayBufferView;
    } else {
      // Convert between typed arrays
      if (targetDtype === "int64" || targetDtype === "uint64") {
        // Regular number to BigInt conversion
        const bigIntArray = new targetTypedArrayConstructor(chunkData.length) as
          | BigInt64Array
          | BigUint64Array;
        for (let i = 0; i < chunkData.length; i++) {
          bigIntArray[i] = BigInt(chunkData[i]);
        }
        return bigIntArray;
      } else {
        // Standard numeric conversion - use typed conversion
        const typedArrayMap = new Map<
          TypedArrayConstructor,
          (data: ArrayLike<number>) => ArrayBufferView
        >([
          [Uint8Array, (data) => new Uint8Array(Array.from(data))],
          [Int8Array, (data) => new Int8Array(Array.from(data))],
          [Uint16Array, (data) => new Uint16Array(Array.from(data))],
          [Int16Array, (data) => new Int16Array(Array.from(data))],
          [Uint32Array, (data) => new Uint32Array(Array.from(data))],
          [Int32Array, (data) => new Int32Array(Array.from(data))],
          [Float32Array, (data) => new Float32Array(Array.from(data))],
          [Float64Array, (data) => new Float64Array(Array.from(data))],
        ]);

        const createTypedArray = typedArrayMap.get(targetTypedArrayConstructor);
        if (createTypedArray) {
          return createTypedArray(chunkData);
        } else {
          throw new Error(
            `Unsupported target constructor: ${targetTypedArrayConstructor.name}`,
          );
        }
      }
    }
  } else {
    // Handle other types (fallback)
    throw new Error(
      `Unsupported source data type: ${chunkData.constructor.name}`,
    );
  }
}

function calculateChunkIndices(shape: number[], chunks: number[]): number[][] {
  const indices: number[][] = [];

  function generateIndices(dimIndex: number, currentIndex: number[]): void {
    if (dimIndex === shape.length) {
      indices.push([...currentIndex]);
      return;
    }

    const chunkSize = chunks[dimIndex];
    const dimSize = shape[dimIndex];

    for (let i = 0; i < Math.ceil(dimSize / chunkSize); i++) {
      currentIndex[dimIndex] = i;
      generateIndices(dimIndex + 1, currentIndex);
    }
  }

  generateIndices(0, new Array(shape.length));
  return indices;
}

function calculateChunkStride(chunkShape: number[]): number[] {
  const stride = new Array(chunkShape.length);
  stride[chunkShape.length - 1] = 1;

  for (let i = chunkShape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * chunkShape[i + 1];
  }

  return stride;
}

/**
 * Write OME-Zarr to .ozx ZIP file (RFC-9).
 *
 * This function creates an OME-Zarr hierarchy in memory and then writes it
 * to a ZIP file following RFC-9 specification:
 * - Root-level zarr.json is the first entry
 * - Other zarr.json files follow in breadth-first order
 * - ZIP-level compression is disabled (ZIP_STORED)
 * - A comment with OME-Zarr version is added
 *
 * @param path - Output .ozx file path
 * @param multiscales - NgffMultiscales, NgffScene, or V06Transform to write;
 *   see {@link toOmeZarrOzxData}
 * @param options - Options for writing
 * @throws Error if called in a browser environment (use toOmeZarrOzxData instead)
 *
 * @see https://ngff.openmicroscopy.org/rfc/9/index.html
 */
export async function toOmeZarrOzx(
  path: string,
  multiscales: NgffMultiscales | NgffScene | V06Transform,
  options: ToOmeZarrOzxOptions = {},
): Promise<void> {
  // Check for browser environment
  if (typeof window !== "undefined") {
    throw new Error(
      "toOmeZarrOzx cannot write files in browser environments. " +
        "Use toOmeZarrOzxData to get the ZIP data as Uint8Array.",
    );
  }

  // Get the ZIP data
  const zipData = await toOmeZarrOzxData(multiscales, options);

  // Write to file using fs module (works in both Node.js and Deno)
  try {
    // Use dynamic import for Node.js fs module
    // Deno also supports node:fs/promises via its Node compatibility layer
    const { writeFile } = await import("node:fs/promises");
    await writeFile(path, zipData);
  } catch (error) {
    throw new Error(
      `Failed to write .ozx file: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

/** @deprecated Use {@link toOmeZarrOzx} instead. */
export const toNgffZarrOzx = toOmeZarrOzx;

/**
 * Create OME-Zarr .ozx ZIP data (RFC-9).
 *
 * This function creates an OME-Zarr hierarchy in memory and returns
 * the ZIP data as a Uint8Array. Useful for browser environments or
 * when you need the raw ZIP data.
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
export async function toOmeZarrOzxData(
  multiscales: NgffMultiscales | NgffScene | V06Transform,
  options: ToOmeZarrOzxOptions = {},
): Promise<Uint8Array> {
  if (isV06Transform(multiscales)) {
    return await transformationToOzx(multiscales, { version: options.version });
  }
  if (multiscales instanceof NgffScene) {
    return await sceneToOzx(multiscales, arrayWriter, options);
  }
  const version = gateOzxVersion(options.version);

  // Create a memory store to hold the zarr data
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();

  // Write to the memory store using existing toOmeZarr logic
  // but we need to inline the core logic since toOmeZarr would
  // try to detect .ozx again
  await _writeToMemoryStore(
    memoryStore,
    multiscales,
    options.onProgress ?? null,
    options.consolidateMetadata ?? true,
    version,
    options.chunksPerShard ?? DEFAULT_OZX_CHUNKS_PER_SHARD,
  );

  // Convert the memory store to ZIP data; the ZIP comment records the
  // version the root metadata was written at.
  const zipData = memoryStoreToZip(memoryStore, { version });

  return zipData;
}

/** @deprecated Use {@link toOmeZarrOzxData} instead. */
export const toNgffZarrOzxData = toOmeZarrOzxData;

/**
 * The files below `directory` as a MemoryStore, keyed by their paths
 * relative to it, as zarrita addresses a store's keys.
 */
async function readDirectoryStore(directory: string): Promise<MemoryStore> {
  const { readdir, readFile } = await import("node:fs/promises");
  const { join } = await import("node:path");
  const store: MemoryStore = new Map();
  const walk = async (relative: string): Promise<void> => {
    const entries = await readdir(join(directory, relative), {
      withFileTypes: true,
    });
    for (const entry of entries) {
      const child = relative === "" ? entry.name : `${relative}/${entry.name}`;
      if (entry.isDirectory()) {
        await walk(child);
      } else if (entry.isFile()) {
        store.set(
          `/${child}`,
          new Uint8Array(await readFile(join(directory, child))),
        );
      }
    }
  };
  await walk("");
  return store;
}

/**
 * Pack a staged store into an RFC-9 `.ozx` archive: the in-memory form
 * returns the archive's bytes, and the Node.js form, given an archive path,
 * writes it there from a MemoryStore or a directory, as the Python
 * `write_store_to_zip` does.
 *
 * Staging is how a scene or a transformation that references a stored node,
 * such as a displacement field, reaches an archive: the node is written with
 * `toOmeZarr(store, node, { path })`, then the scene or the transformation
 * with `{ overwrite: false }`, and the store is packed whole. The archive
 * records the version the store's root document declares unless
 * `options.version` is given, and leads with its root `zarr.json`. A
 * directory is read into memory to be packed.
 *
 * @example
 * ```typescript
 * const store = new Map<string, Uint8Array>();
 * await toOmeZarr(store, field, { version: "0.6", path: warp.path });
 * await toOmeZarr(store, scene, { overwrite: false });
 * const archive = storeToZip(store);
 *
 * await storeToZip("scene.ome.zarr", "scene.ozx");
 * ```
 */
export function storeToZip(
  store: MemoryStore,
  options?: StoreToZipOptions,
): Uint8Array;
export function storeToZip(
  source: string | MemoryStore,
  zipPath: string,
  options?: StoreToZipOptions,
): Promise<void>;
export function storeToZip(
  source: string | MemoryStore,
  zipPathOrOptions?: string | StoreToZipOptions,
  options: StoreToZipOptions = {},
): Uint8Array | Promise<void> {
  if (typeof zipPathOrOptions !== "string") {
    if (!(source instanceof Map)) {
      throw new Error(
        "storeToZip(store) packs a MemoryStore into bytes; to pack a " +
          "directory, give the archive's path: storeToZip(directory, zipPath).",
      );
    }
    return storeToZipData(source, zipPathOrOptions);
  }
  const zipPath = zipPathOrOptions;
  return (async () => {
    if (typeof window !== "undefined") {
      throw new Error(
        "storeToZip cannot write files in browser environments; " +
          "storeToZip(store) returns the archive's bytes.",
      );
    }
    const store = typeof source === "string"
      ? await readDirectoryStore(source)
      : source;
    const zipData = storeToZipData(store, options);
    const { writeFile } = await import("node:fs/promises");
    await writeFile(zipPath, zipData);
  })();
}

/**
 * Internal function to write multiscales to a memory store.
 * Used by toOmeZarrOzxData to avoid recursion with toOmeZarr.
 */
async function _writeToMemoryStore(
  store: MemoryStore,
  multiscales: NgffMultiscales,
  onProgress?: ((completedChunks: number, totalChunks: number) => void) | null,
  consolidate: boolean = true,
  version: OzxVersion = "0.5",
  chunksPerShard?: ChunksPerShard,
): Promise<void> {
  await writeNgffMultiscalesToMemoryStore(
    store,
    multiscales,
    arrayWriter({ chunksPerShard }),
    onProgress,
    consolidate,
    version,
    chunksPerShard,
  );
}
