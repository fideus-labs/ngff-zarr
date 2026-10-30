// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * How the Node and browser `toOmeZarr` write an image's array data. No
 * Node.js dependency.
 */
import * as zarr from "zarrita";

import type { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { arrayLayout, type ChunksPerShard } from "../utils/sharding.ts";
import { createWriteQueue, zarrGet, zarrSet } from "../utils/worker_pool.ts";
import type { MemoryStore } from "./rfc9_zip.ts";
import type { ArrayWriterFactory } from "./to_ngff_zarr_ozx_common.ts";

/** Writes arrays the way this module does, with the given codecs and sharding. */
export const arrayWriter: ArrayWriterFactory =
  ({ codecs, chunksPerShard }) =>
  (group, image, path, onProgress) =>
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

export function getTypedArrayConstructor(
  dtype: zarr.DataType,
): TypedArrayConstructor {
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
    Math.min(start + writeShape[dim], shape[dim]),
  );

  // Calculate chunk shape
  const chunkShape = chunkEnd.map((end, dim) => end - chunkStart[dim]);

  // Create selection for this chunk from the source data
  const sourceSelection = chunkStart.map((start, dim) =>
    zarr.slice(start, chunkEnd[dim]),
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
  const actualSize = (chunkTargetData as unknown as ArrayLike<unknown>).length;
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
    zarr.slice(start, chunkEnd[dim]),
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
  // zarrita reads bool data as a BoolArray, which is not a TypedArray.
  if (chunkData instanceof zarr.BoolArray && targetDtype === "bool") {
    return chunkData as unknown as ArrayBufferView;
  }
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
        const bigIntArray = new targetTypedArrayConstructor(
          chunkData.length,
        ) as BigInt64Array | BigUint64Array;
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
