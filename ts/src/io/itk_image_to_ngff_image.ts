// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Convert ITK-Wasm Image to NgffImage
 */

import type { Image } from "itk-wasm";
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";
import { itkLpsToAnatomicalOrientation } from "../types/rfc4.ts";
import type { AnatomicalOrientation } from "../types/rfc4.ts";

// Import the get_strides function from zarrita utilities
import { _zarrita_internal_get_strides as getStrides } from "zarrita";

export interface ItkImageToNgffImageOptions {
  /**
   * Whether to add anatomical orientation metadata based on ITK LPS coordinate system
   * @default true
   */
  addAnatomicalOrientation?: boolean;

  /**
   * Path prefix for the zarr array (e.g., "scale0/", "scale1/")
   * @default "image"
   */
  path?: string;

  /**
   * Chunk size for the zarr array. Can be:
   * - A single number (applied to all dimensions, capped at dimension size)
   * - An array of numbers (one per dimension)
   * @default 256
   */
  chunks?: number | number[];
}

/**
 * Convert an ITK-Wasm Image to an NgffImage, preserving spatial metadata.
 *
 * This function converts ITK-Wasm Image objects to NgffImage format while
 * preserving spatial information like spacing, origin, and optionally
 * anatomical orientation based on the ITK LPS coordinate system.
 *
 * @param itkImage - The ITK-Wasm Image to convert
 * @param options - Conversion options
 * @returns Promise resolving to NgffImage
 */
export async function itkImageToNgffImage(
  itkImage: Image,
  options: ItkImageToNgffImageOptions = {},
): Promise<NgffImage> {
  const { addAnatomicalOrientation = true, path = "image", chunks = 256 } =
    options;

  // Extract image properties from ITK-Wasm Image
  const _data = itkImage.data;
  // ITK stores size in physical space order [x, y, z], but the data buffer is in
  // C-order (row-major) which means [z, y, x] indexing. Reverse to match data layout.
  const shape = [...itkImage.size].reverse();
  const spacing = itkImage.spacing;
  const origin = itkImage.origin;
  const ndim = shape.length;

  // Determine dimension names based on shape and image type
  // This logic matches the Python implementation
  let dims: string[];

  // Check if this is a vector image (multi-component)
  const imageType = itkImage.imageType;
  const isVector = imageType.components > 1;

  if (ndim === 3 && isVector) {
    // 2D RGB/vector image: 2D spatial + components
    dims = ["y", "x", "c"];
  } else if (ndim < 4) {
    // Scalar images up to 3D: take the last ndim spatial dimensions
    dims = ["z", "y", "x"].slice(-ndim);
  } else if (ndim < 5) {
    // 3D RGB/vector or 4D scalar
    if (isVector) {
      dims = ["z", "y", "x", "c"];
    } else {
      dims = ["t", "z", "y", "x"];
    }
  } else if (ndim < 6) {
    // 4D RGB/vector
    dims = ["t", "z", "y", "x", "c"];
  } else {
    throw new Error(`Unsupported number of dimensions: ${ndim}`);
  }

  // Identify spatial dimensions
  const allSpatialDims = new Set(["x", "y", "z"]);
  const spatialDims = dims.filter((dim) => allSpatialDims.has(dim));

  // Create scale mapping from spacing
  // ITK stores spacing/origin in physical space order (x, y, z),
  // but we need to map them to array order (z, y, x).
  // Reverse the arrays to convert from physical to array order, matching Python implementation.
  const scale: Record<string, number> = {};
  const reversedSpacing = spacing.slice().reverse();
  spatialDims.forEach((dim, idx) => {
    scale[dim] = reversedSpacing[idx];
  });

  // Create translation mapping from origin
  const translation: Record<string, number> = {};
  const reversedOrigin = origin.slice().reverse();
  spatialDims.forEach((dim, idx) => {
    translation[dim] = reversedOrigin[idx];
  });

  // Create Zarr array from ITK-Wasm data
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  // Determine appropriate chunk size based on options
  const chunkShape = typeof chunks === "number"
    ? shape.map((s) => Math.min(s, chunks))
    : shape.map((s, i) => Math.min(s, chunks[i] ?? 256));

  const zarrArray = await zarr.create(root.resolve(path), {
    shape: shape,
    chunk_shape: chunkShape,
    data_type: imageType.componentType as zarr.DataType,
    fill_value: 0,
  });

  // Write the ITK-Wasm data to the zarr array
  // We use zarrita's set function to write the entire data efficiently

  // Create a selection that covers the entire array (null means "all" for each dimension)
  const selection = new Array(ndim).fill(null);

  // Create a chunk object with the ITK-Wasm data in zarrita format
  // ITK-Wasm stores data in column-major order with size [x, y, z],
  // which has the same memory layout as C-order (row-major) with shape [z, y, x].
  // We reversed the shape above, and now use C-order strides for that reversed shape.
  const dataChunk = {
    data: itkImage.data as zarr.TypedArray<typeof imageType.componentType>,
    shape: shape,
    stride: getStrides(shape, "C"), // C-order strides for the reversed shape
  };

  // Write all data to the zarr array using zarrita's set function
  // This handles chunking and encoding automatically
  await zarr.set(zarrArray, selection, dataChunk); // Add anatomical orientation if requested
  let axesOrientations: Record<string, AnatomicalOrientation> | undefined;
  if (addAnatomicalOrientation) {
    axesOrientations = {};
    for (const dim of spatialDims) {
      const orientation = itkLpsToAnatomicalOrientation(dim);
      if (orientation !== undefined) {
        axesOrientations[dim] = orientation;
      }
    }
  }

  return new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "image",
    axesUnits: undefined,
    axesOrientations,
    computedCallbacks: undefined,
  });
}
