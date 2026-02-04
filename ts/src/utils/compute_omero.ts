// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Compute OMERO metadata from NgffImage data.
 *
 * This is the main-thread (Node.js/Deno) implementation.
 * For browser environments with WebWorker support, see compute_omero-browser.ts.
 */

import * as zarr from "zarrita";
import type { NgffImage } from "../types/ngff_image.ts";
import type { Multiscales } from "../types/multiscales.ts";
import type {
  Omero,
  OmeroChannel,
  OmeroWindow,
} from "../types/zarr_metadata.ts";

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

// Import for internal use
import {
  computeChannelStatistics,
  type ComputeOmeroFromMultiscalesOptions,
  type ComputeOmeroOptions,
  extractChannel,
  getDefaultColors,
  validateColor,
  validateQuantiles,
} from "./compute_omero-shared.ts";

/**
 * Compute OMERO metadata from an NgffImage.
 *
 * This function computes visualization parameters (OMERO metadata) from image data:
 * - min/max: The actual data range
 * - start/end: Display window based on quantiles (default 2% and 98%)
 *
 * For multi-channel images (with 'c' dimension), statistics are computed
 * separately for each channel, resulting in per-channel OMERO windows.
 *
 * Memory requirements: This function loads the entire image data into memory
 * for statistics computation. For large images, use the lowest resolution
 * image from a multiscales pyramid.
 *
 * Edge cases:
 * - If all values in a channel are NaN, the statistics will be NaN.
 * - If a channel has constant values, min/max/start/end will all be the same.
 *
 * @param image - The NgffImage to compute metadata for
 * @param options - Optional configuration for quantiles, colors, and labels
 *   - quantiles: Tuple of (low, high) quantile values. Must be between 0 and 1,
 *                with low < high. Default is [0.02, 0.98] for 2% and 98% quantiles.
 *   - colors: List of hex color strings (without #) for each channel.
 *             Must be 6-digit hexadecimal strings (e.g., "FF0000" for red).
 *             If not provided, uses white for single channel or Glasbey
 *             progression for multi-channel.
 *   - labels: List of label strings for each channel.
 *             If not provided, uses empty strings.
 * @returns Promise resolving to Omero metadata with computed window parameters
 * @throws {Error} If quantiles are invalid, colors are invalid format,
 *                 or not enough colors/labels provided.
 *
 * @example
 * ```ts
 * const omero = await computeOmeroFromNgffImage(image);
 * multiscales.metadata.omero = omero;
 * ```
 */
export async function computeOmeroFromNgffImage(
  image: NgffImage,
  options: ComputeOmeroOptions = {},
): Promise<Omero> {
  const quantiles = options.quantiles ?? [0.02, 0.98];

  // Validate quantiles
  validateQuantiles(quantiles as [number, number]);

  const dims = image.dims;
  const shape = image.data.shape;

  // Check if there's a channel dimension
  const hasChannelDim = dims.includes("c");
  const cIndex = hasChannelDim ? dims.indexOf("c") : -1;
  const nChannels = hasChannelDim ? shape[cIndex] : 1;

  // Get colors
  let channelColors: string[];
  if (options.colors !== undefined) {
    if (options.colors.length < nChannels) {
      throw new Error(
        `Not enough colors provided. Got ${options.colors.length}, need ${nChannels}.`,
      );
    }
    // Validate each color
    for (const color of options.colors.slice(0, nChannels)) {
      validateColor(color);
    }
    channelColors = options.colors.slice(0, nChannels);
  } else {
    channelColors = getDefaultColors(nChannels);
  }

  // Get labels
  let channelLabels: string[];
  if (options.labels !== undefined) {
    if (options.labels.length < nChannels) {
      throw new Error(
        `Not enough labels provided. Got ${options.labels.length}, need ${nChannels}.`,
      );
    }
    channelLabels = options.labels.slice(0, nChannels);
  } else {
    channelLabels = Array(nChannels).fill("");
  }

  // Read all data from the zarr array
  const result = await zarr.get(image.data);
  const fullData = result.data as ArrayLike<number>;

  // Compute statistics for each channel
  const channels: OmeroChannel[] = [];

  for (let chIdx = 0; chIdx < nChannels; chIdx++) {
    let channelData: ArrayLike<number>;

    if (hasChannelDim) {
      // Extract this channel's data
      channelData = extractChannel(fullData, shape, chIdx, cIndex);
    } else {
      channelData = fullData;
    }

    // Compute statistics
    const stats = computeChannelStatistics(
      channelData,
      quantiles as [number, number],
    );

    // Create OMERO window
    const window: OmeroWindow = {
      min: stats.min,
      max: stats.max,
      start: stats.qLow,
      end: stats.qHigh,
    };

    // Create channel metadata
    const channel: OmeroChannel = {
      color: channelColors[chIdx],
      window,
      label: channelLabels[chIdx],
    };
    channels.push(channel);
  }

  return { channels };
}

/**
 * Compute OMERO metadata from a Multiscales object.
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
