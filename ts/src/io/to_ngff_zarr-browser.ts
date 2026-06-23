// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible version of to_ngff_zarr that doesn't import @zarrita/storage
// (which contains Node.js-specific modules like node:fs, node:buffer, node:path)
import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { createWriteQueue, zarrGet, zarrSet } from "../utils/worker_pool.ts";
import type { MemoryStore } from "./from_ngff_zarr-browser.ts";
import { memoryStoreToZip } from "./rfc9_zip.ts";
import {
  processAxes,
  writeNgffMultiscalesToMemoryStore,
} from "./to_ngff_zarr_ozx_common.ts";
import {
  buildV06MultiscalesEntry,
  legacyTopLevelTransforms,
} from "../utils/v06_metadata.ts";

export { isOzxPath } from "./rfc9_zip.ts";

export interface ToOmeZarrOptions {
  overwrite?: boolean;
  version?: "0.4" | "0.5" | "0.6";
  chunksPerShard?: number | number[] | Record<string, number>;
  /**
   * Custom codec pipeline for array compression. When omitted the default
   * ``blosc(zstd)`` pipeline from {@link defaultCodecs} is used. Use
   * {@link codecFromName} or {@link bytesOnlyCodecs} to build common pipelines.
   */
  codecs?: ZarrCodec[];
}

/** @deprecated Use {@link ToOmeZarrOptions} instead. */
export type ToNgffZarrOptions = ToOmeZarrOptions;

/**
 * Options for writing to .ozx (RFC-9) format.
 * Note: chunksPerShard is NOT supported for .ozx files and will throw an error.
 */
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
}

/** @deprecated Use {@link ToOmeZarrOzxOptions} instead. */
export type ToNgffZarrOzxOptions = ToOmeZarrOzxOptions;

/**
 * Browser-compatible version of toOmeZarr.
 * Only supports MemoryStore (Map) for writing.
 * Does NOT support local file paths or HTTP URLs (use the full version in Node.js/Deno).
 */
export async function toOmeZarr(
  store: string | MemoryStore | zarr.FetchStore,
  multiscales: NgffMultiscales,
  options: ToOmeZarrOptions = {},
): Promise<void> {
  const _overwrite = options.overwrite ?? true;
  const _version = options.version ?? "0.4";

  try {
    // Determine the appropriate store type based on the path
    let _resolvedStore: MemoryStore;
    if (store instanceof Map) {
      _resolvedStore = store;
    } else if (store instanceof zarr.FetchStore) {
      throw new Error(
        "FetchStore is read-only and cannot be used for writing. Use MemoryStore instead.",
      );
    } else if (store.startsWith("http://") || store.startsWith("https://")) {
      throw new Error(
        "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use MemoryStore instead.",
      );
    } else {
      // Local file paths are not supported in browser environments
      throw new Error(
        "Local file paths are not supported in browser environments. Use MemoryStore instead.",
      );
    }

    // Create root location and group with zarrita v0.5.2 API
    const root = zarr.root(_resolvedStore);

    // Process axes (orientation included when present)
    const processedAxes = processAxes(multiscales.metadata.axes);

    // Create the root group with OME-Zarr metadata.
    // - v0.6 (RFC 5): coordinate systems + per-dataset sequence transforms,
    //   wrapped under the "ome" property.
    // - v0.5: axes carried directly, wrapped under "ome".
    // - v0.4: axes carried directly, placed at the root.
    let attributes: Record<string, unknown>;
    if (_version === "0.6") {
      const v06Entry = buildV06MultiscalesEntry(
        multiscales.metadata,
        processedAxes,
      );
      attributes = {
        ome: {
          version: _version,
          multiscales: [v06Entry],
          ...(multiscales.metadata.omero && {
            omero: multiscales.metadata.omero,
          }),
        },
      };
    } else {
      // v0.4/v0.5 only support the simple scale/translation subset of top-level
      // transformations; drop richer v0.6 transforms when downgrading.
      const legacyTransforms = legacyTopLevelTransforms(
        multiscales.metadata.coordinateTransformations,
      );
      const multiscalesMetadata = {
        version: _version,
        name: multiscales.metadata.name,
        axes: processedAxes,
        datasets: multiscales.metadata.datasets,
        ...(legacyTransforms && {
          coordinateTransformations: legacyTransforms,
        }),
        ...(multiscales.metadata.type && {
          type: multiscales.metadata.type,
        }),
        ...(multiscales.metadata.metadata && {
          metadata: multiscales.metadata.metadata,
        }),
      };

      attributes = _version === "0.5"
        ? {
          ome: {
            version: _version,
            multiscales: [multiscalesMetadata],
            ...(multiscales.metadata.omero && {
              omero: multiscales.metadata.omero,
            }),
          },
        }
        : {
          multiscales: [multiscalesMetadata],
          ...(multiscales.metadata.omero && {
            omero: multiscales.metadata.omero,
          }),
        };
    }

    const rootGroup = await zarr.create(root, { attributes });

    // Write each image in the multiscales
    for (let i = 0; i < multiscales.images.length; i++) {
      const image = multiscales.images[i];
      const dataset = multiscales.metadata.datasets[i];

      if (!dataset) {
        throw new Error(`No dataset configuration found for image ${i}`);
      }

      await _writeImage(
        rootGroup as zarr.Group<MemoryStore>,
        image,
        dataset.path,
        undefined, // onProgress
        options.codecs,
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
): Promise<void> {
  try {
    const chunks = getChunksFromImage(image);

    // Convert LazyArray dtype to zarrita DataType
    const zarrDataType = _convertDtypeToZarrType(image.data.dtype);

    // Create array location
    const arrayLocation = group.resolve(arrayPath);

    // Create the zarr array with proper configuration
    const zarrArray = await zarr.create(arrayLocation, {
      shape: image.data.shape,
      data_type: zarrDataType,
      chunk_shape: chunks,
      fill_value: 0,
      codecs: codecs ?? defaultCodecs(zarrDataType),
    });

    await _writeArrayData(
      zarrArray as zarr.Array<zarr.DataType, MemoryStore>,
      image,
      onProgress,
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
  onProgress?: ((completedChunks: number, totalChunks: number) => void) | null,
): Promise<void> {
  try {
    // Get array shape for chunk calculation - we don't need the full data here
    const shape = image.data.shape;

    // Calculate chunk indices for parallel writing
    const chunkIndices = calculateChunkIndices(shape, zarrArray.chunks);

    // Create a queue for parallel chunk writing
    const writeQueue = createWriteQueue();

    // Queue all chunks for writing
    for (const chunkIndex of chunkIndices) {
      writeQueue.add(async () => {
        await writeChunkWithGet(zarrArray, image, chunkIndex);
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
): Promise<void> {
  // Calculate the chunk bounds
  const shape = image.data.shape;
  const chunkStart = chunkIndex.map((idx, dim) => idx * zarrArray.chunks[dim]);
  const chunkEnd = chunkStart.map((start, dim) =>
    Math.min(start + zarrArray.chunks[dim], shape[dim])
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
 * Note: Sharding is NOT supported because zarrita does not currently
 * support writing shards. If you need sharding, use the Python implementation.
 *
 * @param multiscales - NgffMultiscales data to write
 * @param options - Options for writing
 * @returns ZIP file data as Uint8Array
 *
 * @see https://ngff.openmicroscopy.org/rfc/9/index.html
 */
export async function toOmeZarrOzx(
  multiscales: NgffMultiscales,
  _options: ToOmeZarrOzxOptions = {},
): Promise<Uint8Array> {
  // Create a memory store to hold the zarr data
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();

  // Use the shared write function
  await writeNgffMultiscalesToMemoryStore(
    memoryStore,
    multiscales,
    _writeImage,
    _options.onProgress ?? null,
  );

  // Convert the memory store to ZIP data
  const zipData = memoryStoreToZip(memoryStore, { version: "0.5" });

  return zipData;
}

/** @deprecated Use {@link toOmeZarrOzx} instead. */
export const toNgffZarrOzx = toOmeZarrOzx;
