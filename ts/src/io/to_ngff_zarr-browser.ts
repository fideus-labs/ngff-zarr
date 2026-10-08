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
import type { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { arrayLayout, type ChunksPerShard } from "../utils/sharding.ts";
import { writableStoreBelow } from "../utils/store_below.ts";
import { createWriteQueue, zarrGet, zarrSet } from "../utils/worker_pool.ts";
import type { MemoryStore } from "./from_ngff_zarr-browser.ts";
import { memoryStoreToZip } from "./rfc9_zip.ts";
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

export { isOzxPath } from "./rfc9_zip.ts";

export interface ToOmeZarrOptions {
  overwrite?: boolean;
  /**
   * OME-Zarr version to write. Defaults to "0.5", as the Node.js/Deno
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
}

/** @deprecated Use {@link ToOmeZarrOzxOptions} instead. */
export type ToNgffZarrOzxOptions = ToOmeZarrOzxOptions;

/** Writes arrays the way this module does, with the given codecs and sharding. */
const arrayWriter: ArrayWriterFactory =
  ({ codecs, chunksPerShard }) => (group, image, path, onProgress) =>
    _writeImage(group, image, path, onProgress, codecs, chunksPerShard);

/**
 * Browser-compatible version of toOmeZarr.
 * Only supports MemoryStore (Map) for writing.
 * Does NOT support local file paths or HTTP URLs (use the full version in Node.js/Deno).
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
  const target = path === undefined
    ? store as unknown as zarr.Mutable
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
    return await transformationToOzx(multiscales, { version: options.version });
  }
  if (multiscales instanceof NgffScene) {
    return await sceneToOzx(multiscales, arrayWriter, options);
  }
  const version = gateOzxVersion(options.version);

  // Create a memory store to hold the zarr data
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();

  // Use the shared write function
  const chunksPerShard = options.chunksPerShard ??
    DEFAULT_OZX_CHUNKS_PER_SHARD;
  await writeNgffMultiscalesToMemoryStore(
    memoryStore,
    multiscales,
    arrayWriter({ chunksPerShard }),
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
