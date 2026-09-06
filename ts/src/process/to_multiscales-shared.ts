// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Shared utilities for to_multiscales implementations
 * This module contains types and helper functions used by both
 * browser and Node versions of toMultiscales.
 */

import { Methods } from "../types/methods.ts";
import type { NgffMultiscales } from "../types/multiscales.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import { canonicalAxisOrder } from "../utils/axis_order.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
// deno-lint-ignore no-unused-vars
import { bytesOnlyCodecs, defaultCodecs } from "../utils/codecs.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createNgffMultiscales,
} from "../utils/factory.ts";
import { getMethodMetadata } from "../utils/method_metadata.ts";
import type { AnatomicalOrientation } from "../types/rfc4.ts";
import { orientationFromName } from "../types/rfc4.ts";
import type { AxesType, SupportedDims } from "../types/units.ts";
import {
  createCoordinateSystem,
  INTRINSIC_COORDINATE_SYSTEM_NAME,
} from "../types/zarr_metadata.ts";

export type { ZarrCodec };

export interface ToMultiscalesOptions {
  scaleFactors?: (Record<string, number> | number)[];
  method?: Methods;
  chunks?: number | number[] | Record<string, number>;

  /**
   * Anatomical orientation (RFC 4) for the spatial axes. Either a preset
   * coordinate-system name ("LPS" or "RAS", case-insensitive) or a mapping of
   * axis name to {@link AnatomicalOrientation}. When provided, it takes
   * precedence over any `axesOrientations` on the input image. Orientation is
   * written to the output automatically whenever it is present.
   */
  orientation?: string | Record<string, AnatomicalOrientation>;

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

  /**
   * What to do with an input whose axes are not in the OME-Zarr specification
   * order. `"canonical"`, the default, reorders them to time, then channel,
   * then space (t, c, z, y, x), which is what conversion sources need: ITK
   * component images and the 4-D/5-D default dims inference both yield
   * channel-last. `"preserve"` writes the axes in the order given, which RFC-3
   * permits from OME-Zarr 0.9.dev1 on and which no earlier version accepts.
   */
  axisOrder?: "canonical" | "preserve";
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
 * @returns NgffMultiscales object
 */
export async function toMultiscalesCore(
  inputImage: NgffImage,
  options: ToMultiscalesOptions,
  downsampleItkWasm: DownsampleFunction,
): Promise<NgffMultiscales> {
  const {
    scaleFactors = [2, 4],
    method = Methods.ITKWASM_GAUSSIAN,
    chunks: requestedChunks,
    codecs,
    orientation,
    axisOrder = "canonical",
  } = options;

  if (axisOrder !== "canonical" && axisOrder !== "preserve") {
    throw new Error(
      `axisOrder must be "canonical" or "preserve"; got ${
        JSON.stringify(axisOrder)
      }`,
    );
  }
  // OME-Zarr orders axes time, then channel, then space. Channel-last input
  // (ITK component images, the 4-D/5-D default dims) is normalized here so the
  // generated metadata and every scale are spec-ordered, and so a model the
  // writer would refuse below 0.9.dev1 never reaches it. An order the caller
  // chose is kept on request: RFC-3 lifts the ordering rule, and the reorder is
  // silent otherwise.
  const image = axisOrder === "canonical"
    ? await canonicalAxisOrder(inputImage, codecs)
    : inputImage;
  // A positional `chunks` array indexes the caller's dims, so it follows them
  // through the reordering. The dim-keyed and scalar forms need no change.
  const _chunks = Array.isArray(requestedChunks) && image !== inputImage
    ? image.dims.map((dim) => requestedChunks[inputImage.dims.indexOf(dim)])
    : requestedChunks;

  // The vector-component axis type (RFC-5 displacement/coordinate fields) is
  // carried on the input image, mirroring axesUnits / axesOrientations.
  const axesTypes = image.axesTypes;
  if (axesTypes) {
    const unknown = Object.keys(axesTypes).filter(
      (dim) => !image.dims.includes(dim),
    );
    if (unknown.length > 0) {
      throw new Error(
        `axesTypes refers to unknown dimensions: ${unknown.join(", ")}`,
      );
    }
  }

  // Resolve anatomical orientation (RFC 4). An explicit `orientation` option
  // (preset name or mapping) takes precedence over the input image's
  // axesOrientations.
  const axesOrientations: Record<string, AnatomicalOrientation> | undefined =
    orientation === undefined
      ? image.axesOrientations
      : typeof orientation === "string"
      ? orientationFromName(orientation)
      : orientation;

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
    const overrideType = axesTypes?.[dim];
    if (overrideType !== undefined) {
      const axis = createAxis(
        dim as SupportedDims,
        overrideType as AxesType,
        image.axesUnits?.[dim],
        axesOrientations?.[dim],
      );
      if (overrideType === "displacement" || overrideType === "coordinate") {
        axis.discrete = true;
      }
      return axis;
    }
    if (dim === "x" || dim === "y" || dim === "z") {
      const axisOrientation = axesOrientations?.[dim];
      return createAxis(
        dim as "x" | "y" | "z",
        "space",
        image.axesUnits?.[dim],
        axisOrientation,
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
  // RFC 5 / v0.6 exposes the implicit "intrinsic" coordinate system (the axes
  // of the full-resolution image); each dataset maps into it. Older versions
  // ignore this field on write. Mirrors the Python v0.6 `to_multiscales`.
  metadata.coordinateSystems = [
    createCoordinateSystem(INTRINSIC_COORDINATE_SYSTEM_NAME, axes),
  ];
  // The 'type' field, part of the OME-Zarr specification, in metadata is used here to
  // record the downsampling method applied to generate multiscale images for provenance.
  metadata.type = method;
  if (methodMetadata) {
    metadata.metadata = methodMetadata;
  }

  return createNgffMultiscales(images, metadata, scaleFactors, method);
}
