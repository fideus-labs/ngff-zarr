// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Compute OMERO metadata from NgffImage data.
 */

import * as zarr from "zarrita";
import type { NgffImage } from "../types/ngff_image.ts";
import type { Multiscales } from "../types/multiscales.ts";
import type {
  Omero,
  OmeroChannel,
  OmeroWindow,
} from "../types/zarr_metadata.ts";

/**
 * Glasbey color palette from colorcet's glasbey_hv.
 * Extended from HoloViews default colors using the Glasbey algorithm
 * for maximum distinguishability across 256 categorical colors.
 * See: https://colorcet.holoviz.org/user_guide/Categorical.html
 */
export const GLASBEY_COLORS: readonly string[] = [
  "30A2DA",
  "FC4F30",
  "E5AE38",
  "6D904F",
  "8B8B8B",
  "17BECF",
  "9467BD",
  "D62728",
  "1F77B4",
  "E377C2",
  "8C564B",
  "BCBD22",
  "3A0183",
  "004300",
  "0FFFA9",
  "5E0040",
  "C6BDFF",
  "425052",
  "B80080",
  "FFB7B3",
  "7D0200",
  "6126FF",
  "FFFF9A",
  "AEC9AB",
  "00867C",
  "553A00",
  "94FCFF",
  "00BF00",
  "7D00A0",
  "AB7200",
  "91FF00",
  "01BE8A",
  "00457B",
  "C8826F",
  "FF1F83",
  "DD00FF",
  "057400",
  "644461",
  "888FFF",
  "FFB6F4",
  "536237",
  "CE85FF",
  "686A84",
  "BEB4BE",
  "A56089",
  "95D3FF",
  "0100F8",
  "FF8002",
  "8B2945",
  "ADA06D",
  "53458B",
  "C8FFD9",
  "AA4600",
  "FF798F",
  "83D371",
  "909EBF",
  "9400F5",
  "EBD09B",
  "AD8BB1",
  "00634A",
  "FFDC00",
  "887751",
  "7EABA3",
  "000097",
  "F500C6",
  "653329",
  "006678",
  "04E3C8",
  "A737AE",
  "C5DBE1",
  "4D6EFF",
  "9B9301",
  "CD586B",
  "EFDEFE",
  "795A00",
  "5F889A",
  "B4FF92",
  "5E726B",
  "520066",
  "058751",
  "84206F",
  "3C9605",
  "657300",
  "F1A06C",
  "5F5045",
  "BD004A",
  "D06827",
  "D796AB",
  "895DFF",
  "826C76",
  "2B55B9",
  "6E7CBB",
  "E7D5D3",
  "5D0018",
  "7C3B01",
  "80B17D",
  "C8D97D",
  "00E83B",
  "7CB2FF",
  "FF55FF",
  "A42721",
  "1DE4FF",
  "7DAF3B",
  "7B4B91",
  "E0FF48",
  "6B00C4",
  "CDA897",
  "BE63C4",
  "89CDCE",
  "4603C8",
  "5E9279",
  "414A01",
  "05A79D",
  "CF8C37",
  "FFF8D0",
  "435471",
  "B544FF",
  "CF4993",
  "CFA4DF",
  "94D400",
  "A794DA",
  "2DA558",
  "8DE3B6",
  "A4A99D",
  "6C5CB7",
  "FF7E5E",
  "A7838A",
  "AFBED8",
  "2AC4FF",
  "A6683D",
  "F691FE",
  "874B64",
  "FF0C4B",
  "215E23",
  "4292FF",
  "87839D",
  "672D45",
  "B14F41",
  "004E53",
  "5F1B00",
  "AD4167",
  "503267",
  "D6FFFD",
  "7FB5D1",
  "A9B969",
  "FF96CB",
  "C87495",
  "365039",
  "FFD063",
  "5E5862",
  "879476",
  "A978FF",
  "03C863",
  "E7BED4",
  "D4E3D0",
  "876790",
  "897C27",
  "CDDCFF",
  "AA676B",
  "323474",
  "FF5EA9",
  "009BB0",
  "71FFDD",
  "785C38",
  "50659B",
  "CC00B3",
  "577B55",
  "516E7B",
  "015F92",
  "AABDBE",
  "017F99",
  "04DD97",
  "873A2C",
  "F0968E",
  "75C6AA",
  "70695D",
  "CCDC09",
  "AF8557",
  "D80075",
  "9D3F81",
  "D94500",
  "DD6754",
  "5FFF79",
  "D5B173",
  "62265E",
  "BAA23D",
  "D9F2B3",
  "57028F",
  "A19BAA",
  "4D4A27",
  "A4A9FF",
  "ACE8DB",
  "995901",
  "AC00E2",
  "47822F",
  "CBC3AD",
  "00C5B6",
  "615378",
  "336D68",
  "A59280",
  "8499A2",
  "FD5764",
  "7096D2",
  "728D07",
  "7F004C",
  "1530A0",
  "D1C1E2",
  "C985D0",
  "6C454B",
  "7F0024",
  "00A279",
  "B2A9CF",
  "F90000",
  "B0E9FF",
  "939E50",
  "727A82",
  "D92E55",
  "476101",
  "0059FF",
  "7740B5",
  "ACE460",
  "674525",
  "525D51",
  "957368",
  "A9E49A",
  "A30058",
  "D962F6",
  "8E7DCF",
  "FFBD93",
  "A30092",
  "9AFFB9",
  "A7C2FF",
  "F46200",
  "E5F0FF",
  "B89CA4",
  "609694",
  "FF9F35",
  "8C2900",
  "726B32",
  "DF824E",
  "AF7BD5",
  "BC2D00",
  "7B6FA3",
  "484362",
  "C7A3FF",
  "004D28",
  "C4C68E",
  "E048D7",
  "E7E965",
  "E5C10B",
  "00F4F1",
  "9F5BA2",
  "4C41B7",
  "65338E",
  "767E6C",
  "A98A36",
] as const;

/**
 * Get default colors for channels.
 *
 * For a single channel, returns white (FFFFFF).
 * For multiple channels, uses the Glasbey color progression.
 */
export function getDefaultColors(nChannels: number): string[] {
  if (nChannels === 1) {
    return ["FFFFFF"];
  }
  // Cycle through Glasbey colors if we have more channels than colors
  return Array.from(
    { length: nChannels },
    (_, i) => GLASBEY_COLORS[i % GLASBEY_COLORS.length],
  );
}

/**
 * Options for computing OMERO metadata.
 */
export interface ComputeOmeroOptions {
  /** Tuple of (low, high) quantile values for the display window. Default is [0.02, 0.98]. */
  quantiles?: [number, number];
  /** Optional list of hex color strings (without #) for each channel. */
  colors?: string[];
  /** Optional list of label strings for each channel. */
  labels?: string[];
}

/**
 * Validate that quantiles are in valid range and properly ordered.
 */
function validateQuantiles(quantiles: [number, number]): void {
  const [low, high] = quantiles;
  if (!(low >= 0.0 && low <= 1.0)) {
    throw new Error(`Low quantile must be between 0 and 1, got ${low}`);
  }
  if (!(high >= 0.0 && high <= 1.0)) {
    throw new Error(`High quantile must be between 0 and 1, got ${high}`);
  }
  if (low >= high) {
    throw new Error(
      `Low quantile must be less than high quantile, got (${low}, ${high})`,
    );
  }
}

/**
 * Validate that a color is a valid 6-digit hexadecimal string.
 */
function validateColor(color: string): void {
  const hexPattern = /^[0-9A-Fa-f]{6}$/;
  if (!hexPattern.test(color)) {
    throw new Error(
      `Color must be a 6-digit hexadecimal string without # prefix, got '${color}'`,
    );
  }
}

/**
 * Maximum sample size for approximate quantile computation.
 * For datasets larger than this, we use reservoir sampling for memory efficiency.
 */
const QUANTILE_SAMPLE_SIZE = 10_000;

/**
 * Helper function to compute quantile from a sorted array.
 */
function computeQuantile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) {
    return NaN;
  }
  if (sortedValues.length === 1) {
    return sortedValues[0];
  }

  const index = q * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;

  if (lower === upper) {
    return sortedValues[lower];
  }

  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

/**
 * Compute channel statistics (min, max, and quantiles) from array data.
 *
 * For large datasets (>10,000 non-NaN values), uses reservoir sampling
 * to compute approximate quantiles with reduced memory footprint.
 * Min and max are always exact.
 */
function computeChannelStatistics(
  data: ArrayLike<number>,
  quantiles: [number, number],
): { min: number; max: number; qLow: number; qHigh: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  // Reservoir sample for approximate quantile computation
  const sample: number[] = [];
  let count = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v)) {
      continue;
    }

    // Update min and max (always exact)
    if (v < min) min = v;
    if (v > max) max = v;

    // Update reservoir sample for quantiles
    if (sample.length < QUANTILE_SAMPLE_SIZE) {
      sample.push(v);
    } else {
      // Reservoir sampling: replace random element with probability QUANTILE_SAMPLE_SIZE/(count+1)
      const j = Math.floor(Math.random() * (count + 1));
      if (j < QUANTILE_SAMPLE_SIZE) {
        sample[j] = v;
      }
    }

    count++;
  }

  if (count === 0) {
    return { min: NaN, max: NaN, qLow: NaN, qHigh: NaN };
  }

  // Sort sample for quantile computation
  sample.sort((a, b) => a - b);

  const qLow = computeQuantile(sample, quantiles[0]);
  const qHigh = computeQuantile(sample, quantiles[1]);

  return { min, max, qLow, qHigh };
}

/**
 * Extract a single channel from multi-dimensional typed array data.
 */
function extractChannel(
  data: ArrayLike<number>,
  shape: readonly number[],
  channelIndex: number,
  channelDimIndex: number,
): number[] {
  const nChannels = shape[channelDimIndex];
  const result: number[] = [];

  // Calculate the stride for the channel dimension
  let channelStride = 1;
  for (let i = channelDimIndex + 1; i < shape.length; i++) {
    channelStride *= shape[i];
  }

  // Calculate the total size of dimensions before the channel dimension
  let outerSize = 1;
  for (let i = 0; i < channelDimIndex; i++) {
    outerSize *= shape[i];
  }

  // Extract all values for this channel
  for (let outer = 0; outer < outerSize; outer++) {
    const outerOffset = outer * nChannels * channelStride;
    const channelOffset = outerOffset + channelIndex * channelStride;
    for (let inner = 0; inner < channelStride; inner++) {
      result.push(data[channelOffset + inner]);
    }
  }

  return result;
}

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
 * Options for computing OMERO metadata from multiscales.
 */
export interface ComputeOmeroFromMultiscalesOptions
  extends ComputeOmeroOptions {
  /**
   * If true (default), use the lowest resolution (smallest) image for faster computation.
   * If false, use the highest resolution (largest) image for more accurate statistics.
   */
  useLowestResolution?: boolean;
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
