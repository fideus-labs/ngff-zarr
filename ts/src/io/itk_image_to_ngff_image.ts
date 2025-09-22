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
  const { addAnatomicalOrientation = true } = options;

  // Extract image properties from ITK-Wasm Image
  const _data = itkImage.data;
  const shape = itkImage.size;
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

  // Create scale mapping from spacing (ITK order: [z, y, x] for 3D, [y, x] for 2D)
  const scale: Record<string, number> = {};
  // Define canonical ITK spatial order for up to 3D
  const itkSpatialOrder = ["z", "y", "x"].slice(-spatialDims.length);
  spatialDims.forEach((dim) => {
    const spatialIndex = itkSpatialOrder.indexOf(dim);
    if (spatialIndex === -1) {
      throw new Error(`Unknown spatial dimension: ${dim}`);
    }
    scale[dim] = spacing[spatialIndex];
  });

  // Create translation mapping from origin (ITK order: [z, y, x] for 3D, [y, x] for 2D)
  const translation: Record<string, number> = {};
  spatialDims.forEach((dim) => {
    const spatialIndex = itkSpatialOrder.indexOf(dim);
    if (spatialIndex === -1) {
      throw new Error(`Unknown spatial dimension: ${dim}`);
    }
    translation[dim] = origin[spatialIndex];
  });

  // Create Zarr array from ITK-Wasm data
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  // Determine appropriate chunk size
  const chunkShape = shape.map((s) => Math.min(s, 256));

  const zarrArray = await zarr.create(root.resolve("image"), {
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
  const dataChunk = {
    data: itkImage.data as zarr.TypedArray<typeof imageType.componentType>,
    shape: shape,
    stride: getStrides(shape, "C"), // C-order (row-major) strides for compatibility
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
