// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Convert ITK-Wasm Image to NgffImage
 */

import type { Image } from "itk-wasm";
import * as zarr from "zarrita";
// Import the get_strides function from zarrita utilities
import { _zarrita_internal_getStrides as getStrides } from "zarrita";

import { NgffImage } from "../types/ngff_image.ts";
import type { AnatomicalOrientation } from "../types/rfc4.ts";
import {
  itkDirectionToAnatomicalOrientation,
  itkLpsToAnatomicalOrientation,
} from "../types/rfc4.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { zarrSet } from "../utils/worker_pool.ts";

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
 * Check whether an ITK direction matrix is the identity matrix.
 *
 * When the direction is identity, the image axes align directly with
 * the physical LPS axes and the simple LPS lookup is sufficient.
 *
 * @param direction - Flat (row-major) direction matrix from ITK-Wasm
 * @param n - Number of spatial dimensions (2 or 3)
 * @returns `true` if every element matches the identity matrix
 */
function isIdentityDirection(
  direction: ArrayLike<number | bigint>,
  n: number,
): boolean {
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const expected = row === col ? 1 : 0;
      if (Number(direction[row * n + col]) !== expected) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Warn when the NgffImage cannot carry the ITK direction.
 *
 * An NgffImage places its grid by scale and translation along its own axes,
 * and RFC-4 orientation adds per-axis flips and swaps. An oblique direction is
 * neither, and without orientation any non-identity one is dropped: the image
 * is then placed axis-aligned, and an ITK transform converted against it no
 * longer maps the voxels it maps in ITK. A rigid result of an oblique pair
 * landed 19 voxels off.
 *
 * @param direction - Spatial block of the direction matrix, row-major
 * @param n - Number of spatial dimensions
 * @param oriented - Whether the image is given an anatomical orientation
 */
function warnLostDirection(
  full: number[],
  dimension: number,
  n: number,
  oriented: boolean,
): void {
  const near = (a: number, b: number) => Math.abs(a - b) <= 1e-6;
  const at = (row: number, col: number) => full[row * dimension + col];
  const range = (length: number) => Array.from({ length }, (_, i) => i);
  // Outside the spatial block the matrix has to be the identity.
  const timeLost = range(dimension).some((row) =>
    range(dimension).some(
      (col) =>
        (row >= n || col >= n) && !near(at(row, col), row === col ? 1 : 0),
    ),
  );
  // RFC-4 holds a signed permutation: one unit entry per row and column.
  const magnitude = (row: number, col: number) => Math.abs(at(row, col));
  const signedPermutation = range(n).every(
    (row) =>
      range(n).every(
        (col) => near(magnitude(row, col), 0) || near(magnitude(row, col), 1),
      ) &&
      near(
        range(n).reduce((sum, col) => sum + magnitude(row, col), 0),
        1,
      ) &&
      near(
        range(n).reduce((sum, col) => sum + magnitude(col, row), 0),
        1,
      ),
  );
  const identity = range(n).every((row) =>
    range(n).every((col) => near(at(row, col), row === col ? 1 : 0)),
  );
  if (!timeLost && (oriented ? signedPermutation : identity)) {
    return;
  }
  const rows = range(dimension).map((row) =>
    full.slice(row * dimension, row * dimension + dimension),
  );
  const why = timeLost
    ? "mixes time with space or reverses it, which an NgffImage cannot carry"
    : oriented
      ? "is oblique, and RFC-4 orientation only flips and swaps axes: the " +
        "image is placed on its nearest axis-aligned grid"
      : "is dropped, as addAnatomicalOrientation is false: the image is " +
        "placed axis-aligned";
  console.warn(
    `The ITK direction ${JSON.stringify(rows)} ${why}. An ITK transform ` +
      "converted against this image no longer maps the voxels it maps in ITK.",
  );
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
  const {
    addAnatomicalOrientation = true,
    path = "image",
    chunks = 256,
  } = options;

  // Extract image properties from ITK-Wasm Image
  const _data = itkImage.data;
  // ITK stores size in physical space order [x, y, z], but the data buffer is in
  // C-order (row-major) which means [z, y, x] indexing. Reverse to match data layout.
  const shape = [...itkImage.size].reverse();
  const spacing = itkImage.spacing;
  const origin = itkImage.origin;

  // Check if this is a vector image (multi-component)
  const imageType = itkImage.imageType;
  const isVector = imageType.components > 1;

  // ITK-Wasm stores components separately from size (size is spatial-only).
  // For vector images, append the component count to shape so that ndim and
  // dims correctly include the "c" dimension — matching the Python
  // implementation where image_dict["data"] already includes the component
  // axis in its shape.
  if (isVector) {
    shape.push(imageType.components);
  }

  const ndim = shape.length;

  // Determine dimension names based on shape and image type
  // This logic matches the Python implementation
  let dims: string[];

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

  // ITK stores spacing and origin per ITK axis (x, y, z, then t); pick each
  // axis's entry by name, so a time axis's spacing is not shifted into z.
  const itkAxes = ["x", "y", "z", "t"].slice(0, imageType.dimension);
  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  for (const dim of dims.filter((d) => itkAxes.includes(d))) {
    scale[dim] = spacing[itkAxes.indexOf(dim)];
    translation[dim] = origin[itkAxes.indexOf(dim)];
  }

  // Create Zarr array from ITK-Wasm data
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  // Determine appropriate chunk size based on options
  const chunkShape =
    typeof chunks === "number"
      ? shape.map((s) => Math.min(s, chunks))
      : shape.map((s, i) => Math.min(s, chunks[i] ?? 256));

  const zarrArray = await zarr.create(root.resolve(path), {
    shape: shape,
    chunkShape,
    dtype: imageType.componentType as zarr.DataType,
    fillValue: 0,
    codecs: defaultCodecs(imageType.componentType),
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

  // Write all data to the zarr array using worker-accelerated set function
  // This handles chunking and encoding automatically
  await zarrSet(zarrArray, selection, dataChunk);

  // The direction matrix is dimension x dimension, row-major; orientation
  // reads its spatial block.
  const nSpatial = spatialDims.length;
  const dimension = imageType.dimension;
  const fullDirection = itkImage.direction;
  const direction =
    fullDirection && fullDirection.length > 0
      ? Array.from({ length: nSpatial * nSpatial }, (_, k) =>
          Number(
            fullDirection[
              Math.floor(k / nSpatial) * dimension + (k % nSpatial)
            ],
          ),
        )
      : fullDirection;
  if (fullDirection && fullDirection.length > 0) {
    warnLostDirection(
      Array.from(fullDirection, Number),
      dimension,
      nSpatial,
      addAnatomicalOrientation,
    );
  }

  // Add anatomical orientation if requested
  let axesOrientations: Record<string, AnatomicalOrientation> | undefined;
  if (addAnatomicalOrientation) {
    axesOrientations = {};

    // Check whether the direction matrix is available and has a
    // non-identity rotation (permutation or oblique).
    const hasDirection = direction !== undefined && direction.length > 0;
    const hasNonIdentityDirection =
      hasDirection && !isIdentityDirection(direction, nSpatial);

    if (hasNonIdentityDirection) {
      // Use the direction cosine matrix to determine each axis'
      // anatomical orientation. The direction matrix is stored
      // row-major as a flat array: direction[row * nSpatial + col].
      //
      // spatialDims is in array order (e.g. ["z", "y", "x"]) and
      // maps to reversed ITK axis indices:
      //   spatialDims[0] = "z" → ITK axis 2 → direction column 2
      //   spatialDims[1] = "y" → ITK axis 1 → direction column 1
      //   spatialDims[2] = "x" → ITK axis 0 → direction column 0
      //
      // Column j of the direction matrix contains the LPS-space
      // direction cosine for ITK axis j.
      for (let i = 0; i < nSpatial; i++) {
        const dim = spatialDims[i];
        // Reverse index: spatialDims[0] → last ITK axis, etc.
        const itkAxisIndex = nSpatial - 1 - i;
        // Extract direction column, handling both 2D and 3D cases
        const col: number[] = [];
        for (let row = 0; row < nSpatial; row++) {
          col.push(Number(direction[row * nSpatial + itkAxisIndex]));
        }
        axesOrientations[dim] = itkDirectionToAnatomicalOrientation(col);
      }
    } else {
      // Identity direction or no direction: fall back to LPS labels
      for (const dim of spatialDims) {
        const orientation = itkLpsToAnatomicalOrientation(dim);
        if (orientation !== undefined) {
          axesOrientations[dim] = orientation;
        }
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
