// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Browser-specific OMERO metadata computation using WebWorker.
 * This module offloads CPU-intensive statistics computation to a WebWorker
 * to prevent blocking the main thread.
 *
 * Features:
 * - WebWorker-based computation via Comlink
 * - Chunked streaming for memory efficiency with large images
 * - Automatic fallback to main thread if WebWorker unavailable
 * - Inline worker code (bundled as Blob URL)
 */

import * as Comlink from "comlink";
import * as zarr from "zarrita";
import type { NgffImage } from "../types/ngff_image.ts";
import type { Multiscales } from "../types/multiscales.ts";
import type { Omero } from "../types/zarr_metadata.ts";
import type { ComputeOmeroWorkerApi } from "../workers/compute_omero_worker.ts";
import {
  buildOmeroFromAccumulators,
  type ComputeOmeroFromMultiscalesOptions,
  type ComputeOmeroOptions,
  createAccumulator,
  extractChannel,
  updateAccumulator,
  validateQuantiles,
} from "./compute_omero-shared.ts";

// Re-export shared utilities for convenience
export {
  type ComputeOmeroFromMultiscalesOptions,
  type ComputeOmeroOptions,
  getDefaultColors,
  GLASBEY_COLORS,
} from "./compute_omero-shared.ts";

// Worker state
let worker: Worker | null = null;
let workerProxy: Comlink.Remote<ComputeOmeroWorkerApi> | null = null;
let workerSupported: boolean | null = null;
let workerBlobUrl: string | null = null;

// Chunk size for streaming (256K elements = ~1MB for float32)
const CHUNK_SIZE = 256 * 1024;

// Type alias for all possible typed arrays returned by zarrita
type ZarrTypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

/**
 * Check if WebWorkers are supported in this environment.
 */
function isWorkerSupported(): boolean {
  if (workerSupported === null) {
    try {
      workerSupported = typeof Worker !== "undefined" &&
        typeof Blob !== "undefined" &&
        typeof URL !== "undefined" &&
        typeof URL.createObjectURL === "function";
    } catch {
      workerSupported = false;
    }
  }
  return workerSupported;
}

/**
 * Get or create the worker proxy.
 * Returns null if workers are not supported.
 */
function getWorkerProxy(): Comlink.Remote<ComputeOmeroWorkerApi> | null {
  if (!isWorkerSupported()) {
    return null;
  }

  if (!workerProxy) {
    try {
      // In bundled builds, the Worker constructor line below is replaced with
      // inline Blob URL code that assigns workerBlobUrl. In non-bundled mode,
      // we use a module URL and there's no Blob URL to clean up.
      workerBlobUrl = null; // Will be overwritten in bundled mode
      worker = new Worker(
        new URL("../workers/compute_omero_worker.ts", import.meta.url),
        { type: "module" },
      );
      workerProxy = Comlink.wrap<ComputeOmeroWorkerApi>(worker);
    } catch (e) {
      // Worker creation failed, fall back to main thread
      console.warn("Failed to create OMERO computation worker:", e);
      workerSupported = false;
      return null;
    }
  }

  return workerProxy;
}

/**
 * Compute OMERO metadata on the main thread (fallback implementation).
 * Used when WebWorker is not available.
 */
async function computeOmeroMainThread(
  image: NgffImage,
  options: ComputeOmeroOptions = {},
): Promise<Omero> {
  const quantiles = options.quantiles ?? [0.02, 0.98];
  validateQuantiles(quantiles as [number, number]);

  const dims = image.dims;
  const shape = image.data.shape;

  // Check if there's a channel dimension
  const cIndex = dims.indexOf("c");
  const hasChannelDim = cIndex !== -1;
  const nChannels = hasChannelDim ? shape[cIndex] : 1;

  // Read all data from the zarr array
  const result = await zarr.get(image.data);
  const fullData = result.data as ArrayLike<number>;

  // Create accumulators for each channel
  const accumulators = Array.from(
    { length: nChannels },
    () => createAccumulator(),
  );

  if (hasChannelDim) {
    // Multi-channel: extract each channel and compute statistics
    for (let chIdx = 0; chIdx < nChannels; chIdx++) {
      const channelData = extractChannel(fullData, shape, chIdx, cIndex);
      updateAccumulator(accumulators[chIdx], channelData);
    }
  } else {
    // Single channel: compute directly
    updateAccumulator(accumulators[0], fullData);
  }

  return buildOmeroFromAccumulators(
    accumulators,
    quantiles as [number, number],
    options.colors,
    options.labels,
  );
}

/**
 * Compute OMERO metadata from an NgffImage using WebWorker.
 *
 * This function computes visualization parameters (OMERO metadata) from image data:
 * - min/max: The actual data range
 * - start/end: Display window based on quantiles (default 2% and 98%)
 *
 * For multi-channel images (with 'c' dimension), statistics are computed
 * separately for each channel, resulting in per-channel OMERO windows.
 *
 * The computation is performed in a WebWorker to avoid blocking the main thread.
 * If WebWorkers are not available, falls back to main thread computation.
 *
 * @param image - The NgffImage to compute metadata for
 * @param options - Optional configuration for quantiles, colors, and labels
 * @returns Promise resolving to Omero metadata with computed window parameters
 */
export async function computeOmeroFromNgffImage(
  image: NgffImage,
  options: ComputeOmeroOptions = {},
): Promise<Omero> {
  const api = getWorkerProxy();

  // Fallback to main thread if worker unavailable
  if (!api) {
    return computeOmeroMainThread(image, options);
  }

  try {
    const quantiles = options.quantiles ?? [0.02, 0.98];
    const shape = image.data.shape;
    const dims = image.dims;

    // Read data from zarr array
    const result = await zarr.get(image.data);
    const fullData = result.data;
    const totalSize = fullData.length;

    // For small datasets, use single-call API
    if (totalSize <= CHUNK_SIZE * 2) {
      // Transfer the data buffer to the worker
      const typedArray = fullData as ZarrTypedArray;
      // Create a copy of the buffer that we can transfer
      const buffer = new ArrayBuffer(typedArray.byteLength);
      const view = new Uint8Array(buffer);
      view.set(
        new Uint8Array(
          typedArray.buffer,
          typedArray.byteOffset,
          typedArray.byteLength,
        ),
      );

      const TypedArrayConstructor = typedArray.constructor as new (
        b: ArrayBuffer,
      ) => typeof typedArray;
      return await api.computeOmero(
        Comlink.transfer(new TypedArrayConstructor(buffer), [buffer]),
        {
          shape,
          dims,
          quantiles: quantiles as [number, number],
          colors: options.colors,
          labels: options.labels,
        },
      );
    }

    // For large datasets, use streaming API
    await api.initializeComputation({
      shape,
      dims,
      quantiles: quantiles as [number, number],
      colors: options.colors,
      labels: options.labels,
    });

    // Stream data in chunks
    const typedArray = fullData as ZarrTypedArray;
    for (let offset = 0; offset < totalSize; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, totalSize);
      const chunk = typedArray.slice(offset, end);

      // Create a copy of the chunk buffer that we can transfer
      const chunkBuffer = new ArrayBuffer(chunk.byteLength);
      const chunkView = new Uint8Array(chunkBuffer);
      chunkView.set(
        new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength),
      );

      const ChunkConstructor = chunk.constructor as new (
        b: ArrayBuffer,
      ) => typeof chunk;
      await api.processChunk({
        chunkData: Comlink.transfer(new ChunkConstructor(chunkBuffer), [
          chunkBuffer,
        ]),
      });
    }

    // Finalize and return result
    return await api.finalizeComputation();
  } catch (error) {
    // Fallback to main thread on any error
    console.warn(
      "WebWorker OMERO computation failed, falling back to main thread:",
      error,
    );
    return computeOmeroMainThread(image, options);
  }
}

/**
 * Compute OMERO metadata from a Multiscales object using WebWorker.
 *
 * This is a convenience function that computes OMERO metadata from the
 * highest or lowest resolution image in a multiscales pyramid.
 *
 * @param multiscales - The Multiscales object to compute metadata for
 * @param options - Optional configuration for quantiles, colors, labels, and resolution
 * @returns Promise resolving to Omero metadata with computed window parameters
 */
export async function computeOmeroFromMultiscales(
  multiscales: Multiscales,
  options: ComputeOmeroFromMultiscalesOptions = {},
): Promise<Omero> {
  if (!multiscales.images || multiscales.images.length === 0) {
    throw new Error("Multiscales has no images");
  }

  const useLowestResolution = options.useLowestResolution ?? true;

  // Select which image to use based on resolution preference
  const image = useLowestResolution
    ? multiscales.images[multiscales.images.length - 1] // Last image (lowest resolution)
    : multiscales.images[0]; // First image (highest resolution)

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

  return await computeOmeroFromNgffImage(image, computeOptions);
}

/**
 * Terminate the OMERO computation worker and release resources.
 * Call this when you're done with OMERO computations to free memory.
 */
export function terminateOmeroWorker(): void {
  if (workerProxy) {
    workerProxy[Comlink.releaseProxy]();
    workerProxy = null;
  }
  if (worker) {
    worker.terminate();
    worker = null;
  }
  if (workerBlobUrl) {
    URL.revokeObjectURL(workerBlobUrl);
    workerBlobUrl = null;
  }
}

/**
 * Check if the OMERO computation will use a WebWorker.
 * Useful for testing or conditional logic.
 */
export function isUsingWorker(): boolean {
  return isWorkerSupported() && getWorkerProxy() !== null;
}
