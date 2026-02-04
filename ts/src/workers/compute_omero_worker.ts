// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * WebWorker for computing OMERO metadata.
 * This worker performs CPU-intensive statistics computation off the main thread.
 */

import * as Comlink from "comlink";
import {
  buildOmeroFromAccumulators,
  type ChannelStatisticsAccumulator,
  type ComputeOmeroChunkInput,
  type ComputeOmeroWorkerInput,
  createAccumulator,
  extractChannel,
  updateAccumulator,
  validateQuantiles,
} from "../utils/compute_omero-shared.ts";
import type { Omero } from "../types/zarr_metadata.ts";

/**
 * State maintained across streaming calls for a single computation.
 */
interface ComputationState {
  accumulators: ChannelStatisticsAccumulator[];
  input: ComputeOmeroWorkerInput;
}

let currentState: ComputationState | null = null;

/**
 * Worker API exposed via Comlink.
 */
const workerApi = {
  /**
   * Initialize a new OMERO computation.
   * Must be called before processChunk.
   */
  initializeComputation(input: ComputeOmeroWorkerInput): void {
    validateQuantiles(input.quantiles);

    const cIndex = input.dims.indexOf("c");
    const nChannels = cIndex !== -1 ? input.shape[cIndex] : 1;

    currentState = {
      accumulators: Array.from(
        { length: nChannels },
        () => createAccumulator(),
      ),
      input,
    };
  },

  /**
   * Process a chunk of data.
   * Call this multiple times for streaming large datasets.
   */
  processChunk(chunkInput: ComputeOmeroChunkInput): void {
    if (!currentState) {
      throw new Error(
        "Computation not initialized. Call initializeComputation first.",
      );
    }

    const { accumulators, input } = currentState;
    const { shape, dims } = input;
    const { chunkData } = chunkInput;

    const cIndex = dims.indexOf("c");
    const hasChannelDim = cIndex !== -1;

    if (hasChannelDim) {
      // Multi-channel: extract each channel and update its accumulator
      for (let chIdx = 0; chIdx < accumulators.length; chIdx++) {
        const channelData = extractChannel(chunkData, shape, chIdx, cIndex);
        updateAccumulator(accumulators[chIdx], channelData);
      }
    } else {
      // Single channel: update directly
      updateAccumulator(accumulators[0], chunkData);
    }
  },

  /**
   * Finalize computation and return OMERO metadata.
   * Resets state after returning.
   */
  finalizeComputation(): Omero {
    if (!currentState) {
      throw new Error(
        "Computation not initialized. Call initializeComputation first.",
      );
    }

    const { accumulators, input } = currentState;
    const result = buildOmeroFromAccumulators(
      accumulators,
      input.quantiles,
      input.colors,
      input.labels,
    );

    // Reset state
    currentState = null;

    return result;
  },

  /**
   * Compute OMERO metadata in a single call (non-streaming).
   * Useful for smaller datasets where streaming overhead isn't needed.
   */
  computeOmero(
    data: ArrayLike<number>,
    input: ComputeOmeroWorkerInput,
  ): Omero {
    validateQuantiles(input.quantiles);

    const { shape, dims, quantiles, colors, labels } = input;
    const cIndex = dims.indexOf("c");
    const hasChannelDim = cIndex !== -1;
    const nChannels = hasChannelDim ? shape[cIndex] : 1;

    // Create accumulators for each channel
    const accumulators: ChannelStatisticsAccumulator[] = Array.from(
      { length: nChannels },
      () => createAccumulator(),
    );

    if (hasChannelDim) {
      // Multi-channel: extract each channel and compute statistics
      for (let chIdx = 0; chIdx < nChannels; chIdx++) {
        const channelData = extractChannel(data, shape, chIdx, cIndex);
        updateAccumulator(accumulators[chIdx], channelData);
      }
    } else {
      // Single channel: compute directly
      updateAccumulator(accumulators[0], data);
    }

    return buildOmeroFromAccumulators(accumulators, quantiles, colors, labels);
  },

  /**
   * Check if worker is ready.
   */
  ping(): string {
    return "pong";
  },
};

export type ComputeOmeroWorkerApi = typeof workerApi;

// Expose the API via Comlink
Comlink.expose(workerApi);
