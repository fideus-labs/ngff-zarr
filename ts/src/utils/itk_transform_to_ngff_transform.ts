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
 * offset are permuted from ITK's fastest-axis-first order back to `dims` order,
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
import type { NgffImage } from "../types/ngff_image.ts";
import { changeOfFrame, frameGeometry } from "./itk_direction.ts";
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

/**
 * ITK-Wasm parameterizations that describe a deformation rather than an affine
 * mapping. They are refused with their own message, because "convert it to an
 * affine first" is not advice that applies: a deformation has no affine
 * equivalent. RFC-5 represents those with `displacements` or `coordinates`
 * field arrays instead.
 */
const NON_LINEAR_PARAMETERIZATIONS = new Set([
  "AzimuthElevationToCartesian",
  "BSpline",
  "BSplineSmoothingOnUpdateDisplacementField",
  "ConstantVelocityField",
  "DisplacementField",
  "GaussianExponentialDiffeomorphic",
  "GaussianSmoothingOnUpdateDisplacementField",
  "GaussianSmoothingOnUpdateTimeVaryingVelocityField",
  "Rigid3DPerspective",
  "TimeVaryingVelocityField",
  "VelocityField",
]);

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

/**
 * Exact parameter count each directly decoded parameterization must carry for
 * a given spatial dimension. What ITK's MatrixOffsetTransformBase and friends
 * pack, and therefore the ground truth for the transform's dimensionality.
 */
const DIRECT_PARAMETER_COUNTS: Record<string, (dimension: number) => number> = {
  Identity: () => 0,
  Translation: (dimension) => dimension,
  Scale: (dimension) => dimension,
  Affine: (dimension) => dimension * dimension + dimension,
};

function decode(
  entry: Transform,
  dimension: number,
): { matrix: number[][]; offset: number[] } {
  const parameterization = String(
    entry.transformType.transformParameterization,
  );
  const parameters = asNumbers(entry.parameters);
  const fixed = asNumbers(entry.fixedParameters);

  // A transform of the wrong dimensionality must not be decoded. Slicing a 3D
  // parameter vector down to 2D silently projects the transform (or, for an
  // affine, scrambles rows into a singular matrix); reading a 2D one as 3D
  // runs the slices off the end and fills the result with NaN, which
  // JSON.stringify then writes into the store as null. The parameter count is
  // the ground truth here, not `transformType.inputDimension`, which a
  // hand-built entry may leave at a default.
  const expected = Object.hasOwn(DIRECT_PARAMETER_COUNTS, parameterization)
    ? DIRECT_PARAMETER_COUNTS[parameterization]
    : undefined;
  if (expected !== undefined) {
    if (parameters.length !== expected(dimension)) {
      throw new Error(
        `ITK-Wasm '${parameterization}' transform carries ` +
          `${parameters.length} parameters, but a coordinate system with ` +
          `${dimension} spatial axes requires exactly ${
            expected(dimension)
          }. ` +
          `The transform's dimensionality does not match dims.`,
      );
    }
    // Only the directly decoded parameterizations read fixedParameters as a
    // center; elsewhere (a displacement field's grid, say) it holds other
    // metadata and is none of this branch's business.
    if (fixed.length !== 0 && fixed.length !== dimension) {
      throw new Error(
        `ITK-Wasm '${parameterization}' transform carries ${fixed.length} ` +
          `fixed parameters (the center), but a coordinate system with ` +
          `${dimension} spatial axes requires 0 or ${dimension}`,
      );
    }
  }
  const center = fixed.length === dimension
    ? fixed
    : new Array(dimension).fill(0);

  /** Fold ITK's center of rotation into the offset: `b = t + c - A c`. */
  const foldCenter = (matrix: number[][], translation: number[]): number[] =>
    translation.map((value, row) => {
      let rotated = 0;
      for (let col = 0; col < dimension; col++) {
        rotated += matrix[row][col] * center[col];
      }
      return value + center[row] - rotated;
    });

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
    // itk.ScaleTransform scales about its center, like the affine below.
    const matrix = identityMatrix(dimension);
    for (let i = 0; i < dimension; i++) matrix[i][i] = parameters[i];
    return { matrix, offset: foldCenter(matrix, new Array(dimension).fill(0)) };
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
    return { matrix, offset: foldCenter(matrix, translation) };
  }

  if (NON_LINEAR_PARAMETERIZATIONS.has(parameterization)) {
    throw new Error(
      `only linear ITK transforms can be expressed as an RFC-5 affine; ` +
        `'${parameterization}' describes a deformation. RFC-5 represents ` +
        `those with a 'displacements' or 'coordinates' field instead.`,
    );
  }

  const declared = Number(entry.transformType.inputDimension);
  if (Number.isFinite(declared) && declared !== dimension) {
    throw new Error(
      `ITK-Wasm '${parameterization}' transform is ${declared}D, but the ` +
        `coordinate system has ${dimension} spatial axes`,
    );
  }

  throw new Error(
    `cannot decode an ITK-Wasm '${parameterization}' transform, because that ` +
      `parameterization stores angles or a quaternion rather than a matrix. ` +
      `Convert it to an Affine transform first.`,
  );
}

/** The fixed and moving images a converted transform relates. */
export interface FrameImages {
  /** The image whose grid the transform maps from. */
  fixed?: NgffImage;
  /** The image the transform maps into. */
  moving?: NgffImage;
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
  frames: FrameImages = {},
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
    const parameterization = String(
      entry.transformType.transformParameterization,
    );
    if (parameterization === "Composite") {
      // A parameterless 'Composite' entry is ambiguous. The ITK-Wasm
      // pipeline writes one as a grouping header before the children, but
      // itk.dict_from_transform never writes a header at all: there it is a
      // *nested* composite whose children the serialization dropped, at any
      // position including the first. Decoding past one would silently
      // compose the wrong mapping, so refuse it wherever it appears.
      throw new Error(
        `a 'Composite' entry in an ITK-Wasm transform list cannot be ` +
          `decoded: the serialization drops a nested composite's children, ` +
          `leaving this entry indistinguishable from a pipeline grouping ` +
          `header. For a pipeline-serialized list, drop the leading header ` +
          `entry and pass the children.`,
      );
    }
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

  let matrix = Array.from(
    { length: dimension },
    (_, row) => total[row].slice(0, dimension),
  );
  let offset = Array.from(
    { length: dimension },
    (_, row) => total[row][dimension],
  );

  if ((frames.fixed === undefined) !== (frames.moving === undefined)) {
    throw new Error("pass both fixed and moving, or neither");
  }
  if (frames.fixed !== undefined && frames.moving !== undefined) {
    // ITK physical space -> the intrinsic systems: phi_m^-1 . T . phi_f.
    const itkDims = SPATIAL_DIMS.filter((dim) => spatial.includes(dim));
    const geometry = frameGeometry(frames.fixed, frames.moving, itkDims);
    ({ matrix, offset } = changeOfFrame(
      matrix,
      offset,
      geometry.directionFixed,
      geometry.directionMoving,
      geometry.originFixed,
      geometry.originMoving,
    ));
  }

  // ITK (fastest-axis-first) order -> the order the axes appear in `dims`.
  // ITK orders components by name (x, then y, then z), so the mapping is
  // built by name rather than by reversing, which is only equivalent for
  // the canonical (z, y, x).
  const itkOrder = SPATIAL_DIMS.filter((dim) => spatial.includes(dim));
  const order = spatial.map((dim) => itkOrder.indexOf(dim));
  return {
    matrix: order.map((row) => order.map((col) => matrix[row][col])),
    offset: order.map((row) => offset[row]),
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
 *   this. A mirror never simplifies to a `scale`, since RFC-5 requires strictly
 *   positive scale factors. Note that `multiscales > datasets` accepts only a
 *   single `scale`, a single `identity`, or a two-element `sequence` of scale
 *   and translation, so a bare `translation` or an `affine` belongs in the
 *   multiscales-level `coordinateTransformations` instead. Pass `false` to
 *   always get an `affine`.
 * @returns An RFC-5 coordinate transformation over `dims`.
 */
export function itkTransformToNgffTransform(
  transform: Transform | TransformList,
  dims: string[],
  simplify = true,
  frames: FrameImages = {},
): V06Transform {
  const { matrix, offset } = itkTransformToNgffMatrix(transform, dims, frames);

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
  // RFC-5 requires every scale factor to be strictly positive, so a mirror or
  // a flip is not a `scale` however diagonal its matrix looks. Emitting one
  // anyway writes metadata the schema rejects, and an LPS to RAS flip
  // (diag(-1, -1, 1)) is an ordinary registration result. Fall through to
  // `affine`, which carries the sign.
  const isScale = isDiagonal && diagonal.every((value) => value > 0);

  if (isIdentity && noOffset) return createIdentity();
  if (isIdentity) return createTranslation(offset);
  if (isScale && noOffset) return createScale(diagonal);
  if (isScale) {
    // Scale first, then translate: y = scale * x + translation, which is the
    // form `multiscales > datasets` accepts.
    return createTransformSequence([
      createScale(diagonal),
      createTranslation(offset),
    ]);
  }
  return undefined;
}
