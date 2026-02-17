// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Shared utilities for to_multiscales implementations
 * This module contains types and helper functions used by both
 * browser and Node versions of toMultiscales.
 */

import { Methods } from "../types/methods.ts";
import type { Multiscales } from "../types/multiscales.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import {
  bytesOnlyCodecs,
  defaultCodecs,
} from "../utils/codecs.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../utils/factory.ts";
import { getMethodMetadata } from "../utils/method_metadata.ts";

export type { ZarrCodec };

export interface ToMultiscalesOptions {
  scaleFactors?: (Record<string, number> | number)[];
  method?: Methods;
  chunks?: number | number[] | Record<string, number>;

  /**
   * Codec pipeline for intermediate zarr arrays created during
   * downsampling.  When omitted the default blosc+zstd pipeline is
   * used ({@link defaultCodecs}).
   *
   * Pass {@link bytesOnlyCodecs | `bytesOnlyCodecs()`} to skip
   * compression entirely — useful when the multiscale result will be
   * immediately re-encoded into another format (e.g. OME-TIFF) and
   * the compress/decompress round-trip would be wasted work.
   */
  codecs?: ZarrCodec[];
}

/**
 * Downsampling function type for ITK-Wasm implementations
 */
export type DownsampleFunction = (
  image: NgffImage,
  scaleFactors: (Record<string, number> | number)[],
  smoothing: "gaussian" | "bin_shrink" | "label_image",
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
) => Promise<NgffImage[]>;

/**
 * Generate multiple resolution scales for an NgffImage
 * This is the core implementation used by both browser and Node versions.
 *
 * @param image - Input NgffImage
 * @param options - Configuration options
 * @param downsampleItkWasm - The platform-specific downsampling function
 * @returns Multiscales object
 */
export async function toMultiscalesCore(
  image: NgffImage,
  options: ToMultiscalesOptions,
  downsampleItkWasm: DownsampleFunction,
): Promise<Multiscales> {
  const {
    scaleFactors = [2, 4],
    method = Methods.ITKWASM_GAUSSIAN,
    chunks: _chunks,
    codecs,
  } = options;

  let images: NgffImage[];

  // Check if we should perform actual downsampling
  if (
    method === Methods.ITKWASM_GAUSSIAN ||
    method === Methods.ITKWASM_BIN_SHRINK ||
    method === Methods.ITKWASM_LABEL_IMAGE
  ) {
    // Perform actual downsampling using ITK-Wasm
    const smoothing = method === Methods.ITKWASM_GAUSSIAN
      ? "gaussian"
      : method === Methods.ITKWASM_BIN_SHRINK
      ? "bin_shrink"
      : "label_image";

    images = await downsampleItkWasm(
      image,
      scaleFactors as (Record<string, number> | number)[],
      smoothing,
      _chunks,
      codecs,
    );
  } else {
    // Fallback: create only the base image (no actual downsampling)
    images = [image];
  }

  // Create axes from image dimensions, including orientation if present
  const axes = image.dims.map((dim) => {
    if (dim === "x" || dim === "y" || dim === "z") {
      const orientation = image.axesOrientations?.[dim];
      return createAxis(
        dim as "x" | "y" | "z",
        "space",
        image.axesUnits?.[dim],
        orientation,
      );
    } else if (dim === "c") {
      return createAxis(dim as "c", "channel");
    } else if (dim === "t") {
      return createAxis(dim as "t", "time");
    } else {
      throw new Error(`Unsupported dimension: ${dim}`);
    }
  });

  // Create datasets for all images
  const datasets = images.map((img, index) => {
    return createDataset(
      `scale${index}`,
      img.dims.map((dim) => img.scale[dim]),
      img.dims.map((dim) => img.translation[dim]),
    );
  });

  // Create metadata with method information
  const methodMetadata = getMethodMetadata(method);
  const metadata = createMetadata(axes, datasets, image.name);
  // The 'type' field, part of the OME-Zarr specification, in metadata is used here to
  // record the downsampling method applied to generate multiscale images for provenance.
  metadata.type = method;
  if (methodMetadata) {
    metadata.metadata = methodMetadata;
  }

  return createMultiscales(images, metadata, scaleFactors, method);
}
