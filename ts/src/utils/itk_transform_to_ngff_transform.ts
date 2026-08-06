// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Convert ITK transforms to RFC-5 coordinate transformations.
 *
 * The inverse of `ngff_transform_to_itk_transform.ts`. Its purpose is
 * persistence: a registration produces an ITK transform, and RFC-5 is where
 * that result belongs once it is written next to the image.
 *
 * The same two conventions apply in reverse -- the spatial block and the
 * offset are reversed from ITK's fastest-axis-first order back to Zarr order,
 * and ITK's center of rotation is folded into the offset, since an RFC-5
 * affine has no center:
 *
 * ```
 * ITK:   y = A (x - c) + t + c
 * RFC-5: y = A x + b          with  b = t + c - A c
 * ```
 *
 * Unlike the Python port there is no `itk` package to fall back on here, so
 * only the parameterizations that carry a matrix are supported. Angle- and
 * quaternion-based ones (Euler, Versor, Similarity, ...) must be converted to
 * an affine first.
 */

import type { Transform, TransformList } from "itk-wasm";
import {
  type Affine,
  createAffine,
  createIdentity,
  createScale,
  createTransformSequence,
  createTranslation,
  type V06Transform,
} from "../types/zarr_metadata.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

/** An RFC-5 matrix and offset for the spatial axes, in Zarr order. */
export interface NgffMatrixAndOffset {
  /** Row-major square matrix, in RFC-5 (Zarr) axis order. */
  matrix: number[][];
  /** Translation, in RFC-5 (Zarr) axis order. */
  offset: number[];
}

function identityMatrix(size: number): number[][] {
  return Array.from(
    { length: size },
    (_, row) => Array.from({ length: size }, (_, col) => (row === col ? 1 : 0)),
  );
}

function multiply(left: number[][], right: number[][]): number[][] {
  const size = left.length;
  return Array.from(
    { length: size },
    (_, row) =>
      Array.from({ length: size }, (_, col) => {
        let total = 0;
        for (let k = 0; k < size; k++) total += left[row][k] * right[k][col];
        return total;
      }),
  );
}

/**
 * Read a parameter array as plain numbers.
 *
 * ITK-Wasm types these as `TypedArray`, which admits the BigInt variants even
 * though transform parameters are always floating point.
 */
function asNumbers(values: unknown): number[] {
  if (values === undefined || values === null) return [];
  return Array.from(values as ArrayLike<number | bigint>, Number);
}

function decode(
  entry: Transform,
  dimension: number,
): { matrix: number[][]; offset: number[] } {
  const parameterization = String(
    entry.transformType.transformParameterization,
  );
  const parameters = asNumbers(entry.parameters);
  const fixed = asNumbers(entry.fixedParameters);

  if (parameterization === "Identity") {
    return {
      matrix: identityMatrix(dimension),
      offset: new Array(dimension).fill(0),
    };
  }
  if (parameterization === "Translation") {
    return {
      matrix: identityMatrix(dimension),
      offset: parameters.slice(0, dimension),
    };
  }
  if (parameterization === "Scale") {
    const matrix = identityMatrix(dimension);
    for (let i = 0; i < dimension; i++) matrix[i][i] = parameters[i];
    return { matrix, offset: new Array(dimension).fill(0) };
  }
  if (parameterization === "Affine") {
    const matrix = Array.from(
      { length: dimension },
      (_, row) => parameters.slice(row * dimension, (row + 1) * dimension),
    );
    const translation = parameters.slice(
      dimension * dimension,
      dimension * dimension + dimension,
    );
    const center = fixed.length >= dimension
      ? fixed.slice(0, dimension)
      : new Array(dimension).fill(0);
    // ITK applies the matrix about the center, so fold it into the offset.
    const offset = translation.map((value, row) => {
      let rotated = 0;
      for (let col = 0; col < dimension; col++) {
        rotated += matrix[row][col] * center[col];
      }
      return value + center[row] - rotated;
    });
    return { matrix, offset };
  }

  throw new Error(
    `cannot decode an ITK-Wasm '${parameterization}' transform, because that ` +
      `parameterization stores angles or a quaternion rather than a matrix. ` +
      `Convert it to an Affine transform first.`,
  );
}

/**
 * Convert an ITK transform to a matrix and offset in RFC-5 axis order.
 *
 * @param transform An ITK-Wasm `Transform` or `TransformList`.
 * @param dims The axis names of the coordinate system the result should be
 *   expressed on, in RFC-5 (Zarr) order. Only the spatial axes take part.
 * @returns The matrix and offset over the spatial axes, in Zarr order.
 */
export function itkTransformToNgffMatrix(
  transform: Transform | TransformList,
  dims: string[],
): NgffMatrixAndOffset {
  const spatial = dims.filter((dim) => SPATIAL_DIMS.includes(dim));
  if (spatial.length === 0) {
    throw new Error(`no spatial axes among dims [${dims.join(", ")}]`);
  }
  const dimension = spatial.length;

  const entries = Array.isArray(transform) ? transform : [transform];
  if (entries.length === 0) throw new Error("transform list is empty");

  // An ITK transform list applies its last entry first, so the homogeneous
  // matrices multiply left to right in list order.
  let total = identityMatrix(dimension + 1);
  for (const entry of entries) {
    const { matrix, offset } = decode(entry, dimension);
    const homogeneous = identityMatrix(dimension + 1);
    for (let row = 0; row < dimension; row++) {
      for (let col = 0; col < dimension; col++) {
        homogeneous[row][col] = matrix[row][col];
      }
      homogeneous[row][dimension] = offset[row];
    }
    total = multiply(total, homogeneous);
  }

  // ITK (fastest-axis-first) order -> RFC-5 (Zarr) order.
  const order = Array.from({ length: dimension }, (_, i) => dimension - 1 - i);
  return {
    matrix: order.map((row) => order.map((col) => total[row][col])),
    offset: order.map((row) => total[row][dimension]),
  };
}

/**
 * Convert an ITK transform to an RFC-5 coordinate transformation.
 *
 * This is what lets a registration result be written into an OME-Zarr store:
 * run the registration, convert the transform, and attach it to the
 * multiscales metadata.
 *
 * @param transform An ITK-Wasm `Transform` or `TransformList`.
 * @param dims The coordinate system's axis names, in RFC-5 (Zarr) order.
 *   Non-spatial axes (`t`, `c`) are left untransformed.
 * @param simplify Return the least expressive transformation that represents
 *   the mapping exactly -- `identity`, `translation`, `scale`, or a `sequence`
 *   of scale and translation -- falling back to `affine`. RFC-5 recommends
 *   this, and only these simpler forms are legal inside
 *   `multiscales > datasets`. Pass `false` to always get an `affine`.
 * @returns An RFC-5 coordinate transformation over `dims`.
 */
export function itkTransformToNgffTransform(
  transform: Transform | TransformList,
  dims: string[],
  simplify = true,
): V06Transform {
  const { matrix, offset } = itkTransformToNgffMatrix(transform, dims);

  const spatialIndices: number[] = [];
  dims.forEach((dim, index) => {
    if (SPATIAL_DIMS.includes(dim)) spatialIndices.push(index);
  });
  const ndim = dims.length;

  // Embed the spatial block into the full coordinate system, leaving any
  // non-spatial axis untouched.
  const full = identityMatrix(ndim);
  const fullOffset = new Array(ndim).fill(0);
  spatialIndices.forEach((rowIndex, row) => {
    spatialIndices.forEach((colIndex, col) => {
      full[rowIndex][colIndex] = matrix[row][col];
    });
    fullOffset[rowIndex] = offset[row];
  });

  if (simplify) {
    const simplified = simplifyTransform(full, fullOffset, ndim);
    if (simplified !== undefined) return simplified;
  }

  // RFC-5 stores the upper M x (N+1) block: the matrix followed by the
  // translation as the last column.
  return createAffine(full.map((row, i) => [...row, fullOffset[i]])) as Affine;
}

function simplifyTransform(
  matrix: number[][],
  offset: number[],
  ndim: number,
): V06Transform | undefined {
  let isIdentity = true;
  let isDiagonal = true;
  for (let row = 0; row < ndim; row++) {
    for (let col = 0; col < ndim; col++) {
      const expected = row === col ? 1 : 0;
      if (matrix[row][col] !== expected) isIdentity = false;
      if (row !== col && matrix[row][col] !== 0) isDiagonal = false;
    }
  }
  const noOffset = offset.every((value) => value === 0);
  const diagonal = matrix.map((row, i) => row[i]);

  if (isIdentity && noOffset) return createIdentity();
  if (isIdentity) return createTranslation(offset);
  if (isDiagonal && noOffset) return createScale(diagonal);
  if (isDiagonal) {
    // Scale first, then translate: y = scale * x + translation, which is the
    // form `multiscales > datasets` accepts.
    return createTransformSequence([
      createScale(diagonal),
      createTranslation(offset),
    ]);
  }
  return undefined;
}
