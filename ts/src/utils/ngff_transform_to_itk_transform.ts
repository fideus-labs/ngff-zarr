// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Bridge RFC-5 coordinate transformations to ITK transforms.
 *
 * RFC-5 and ITK describe the same affine geometry with two different
 * conventions, and the differences are silent rather than loud: getting one
 * wrong yields a plausible transform that is simply in the wrong place.
 *
 * Axis order
 *   RFC-5 orders transformation parameters the same way the Zarr array is
 *   ordered, so a `zyx` image has `z` first. ITK orders points
 *   fastest-axis-first, so the same point is `xyz`. Writing `R` for the
 *   axis-reversal permutation, an RFC-5 affine `q = M p + b` becomes
 *   `A = R M R` and `t = R b` in ITK.
 *
 * Composition order
 *   An RFC-5 `sequence` applies its first entry first. An ITK transform list
 *   applies its *last* entry first. Rather than emit a list and rely on that
 *   inversion, this module composes the chain into a single matrix here,
 *   where the order is explicit and testable.
 *
 * Both conventions place the pixel center at the integer index, so no
 * half-pixel correction is involved.
 */

import type { Transform, TransformList } from "itk-wasm";
import type { V06Transform } from "../types/zarr_metadata.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

/** A square matrix stored as an array of rows. */
type Matrix = number[][];

function identityMatrix(size: number): Matrix {
  return Array.from(
    { length: size },
    (_, row) => Array.from({ length: size }, (_, col) => (row === col ? 1 : 0)),
  );
}

function multiply(left: Matrix, right: Matrix): Matrix {
  const size = left.length;
  return Array.from(
    { length: size },
    (_, row) =>
      Array.from({ length: size }, (_, col) => {
        let total = 0;
        for (let k = 0; k < size; k++) {
          total += left[row][k] * right[k][col];
        }
        return total;
      }),
  );
}

function assertMatrix(
  values: number[][] | undefined,
  path: string | undefined,
  field: string,
): Matrix {
  if (values === undefined || values.length === 0) {
    if (path !== undefined) {
      throw new Error(
        `${field} transformation stores its parameters at '${path}'. Reading ` +
          `matrix parameters from a Zarr array is not supported; supply the ` +
          `'${field}' values inline.`,
      );
    }
    throw new Error(`${field} transformation has no parameters`);
  }
  return values;
}

/**
 * Collapse one RFC-5 transformation into an `(ndim+1) x (ndim+1)` homogeneous
 * matrix, in RFC-5 (Zarr) axis order.
 */
function homogeneousFromTransform(
  transform: V06Transform,
  ndim: number,
): Matrix {
  switch (transform.type) {
    case "identity":
      return identityMatrix(ndim + 1);

    case "scale": {
      if (transform.scale.length !== ndim) {
        throw new Error(
          `scale transformation has ${transform.scale.length} parameters but ` +
            `the coordinate system has ${ndim} axes`,
        );
      }
      const matrix = identityMatrix(ndim + 1);
      for (let i = 0; i < ndim; i++) matrix[i][i] = transform.scale[i];
      return matrix;
    }

    case "translation": {
      if (transform.translation.length !== ndim) {
        throw new Error(
          `translation transformation has ${transform.translation.length} ` +
            `parameters but the coordinate system has ${ndim} axes`,
        );
      }
      const matrix = identityMatrix(ndim + 1);
      for (let i = 0; i < ndim; i++) matrix[i][ndim] = transform.translation[i];
      return matrix;
    }

    case "rotation": {
      const rotation = assertMatrix(
        transform.rotation,
        transform.path,
        "rotation",
      );
      if (rotation.length !== ndim || rotation.some((r) => r.length !== ndim)) {
        throw new Error(
          `rotation transformation is ${rotation.length}x` +
            `${rotation[0]?.length} but the coordinate system has ${ndim} axes`,
        );
      }
      const matrix = identityMatrix(ndim + 1);
      for (let row = 0; row < ndim; row++) {
        for (let col = 0; col < ndim; col++) {
          matrix[row][col] = rotation[row][col];
        }
      }
      return matrix;
    }

    case "affine": {
      const affine = assertMatrix(transform.affine, transform.path, "affine");
      // RFC-5 stores the upper M x (N+1) block of a homogeneous matrix: the
      // rotation/scale/shear part followed by the translation as the last
      // column, with the trailing [0 ... 0 1] row omitted.
      if (
        affine.length !== ndim || affine.some((r) => r.length !== ndim + 1)
      ) {
        throw new Error(
          `affine transformation is ${affine.length}x${
            affine[0]?.length
          } but ` +
            `a coordinate system with ${ndim} axes requires ` +
            `${ndim}x${ndim + 1} (the translation is the last column)`,
        );
      }
      const matrix = identityMatrix(ndim + 1);
      for (let row = 0; row < ndim; row++) {
        for (let col = 0; col <= ndim; col++) {
          matrix[row][col] = affine[row][col];
        }
      }
      return matrix;
    }

    case "sequence": {
      if (transform.transformations.length === 0) {
        throw new Error("sequence transformation has no transformations");
      }
      // RFC-5 applies the first entry first, so the matrix product runs
      // right-to-left over the list.
      let matrix = identityMatrix(ndim + 1);
      for (const sub of transform.transformations) {
        matrix = multiply(homogeneousFromTransform(sub, ndim), matrix);
      }
      return matrix;
    }

    default:
      throw new Error(
        `transformation type '${
          (transform as { type: string }).type
        }' cannot be converted to an ITK transform. Only identity, scale, ` +
          `translation, rotation, affine and sequences of them describe a ` +
          `linear mapping that ITK can represent as a single affine transform.`,
      );
  }
}

/** An ITK matrix and offset for the spatial axes, fastest-axis-first. */
export interface ItkMatrixAndOffset {
  /** Row-major square matrix, in ITK axis order. */
  matrix: number[][];
  /** Translation, in ITK axis order. */
  offset: number[];
}

/**
 * Convert an RFC-5 transformation to an ITK matrix and offset.
 *
 * @param transform An RFC-5 (OME-Zarr v0.6) coordinate transformation. It must
 *   describe a linear mapping: `identity`, `scale`, `translation`, `rotation`,
 *   `affine`, or a `sequence` of those.
 * @param dims The axis names of the coordinate system the transformation is
 *   defined on, in RFC-5 (Zarr) order, e.g. `["z", "y", "x"]`.
 * @returns The matrix and offset for the spatial axes, in ITK order.
 */
export function ngffTransformToItkMatrix(
  transform: V06Transform,
  dims: string[],
): ItkMatrixAndOffset {
  const ndim = dims.length;
  const homogeneous = homogeneousFromTransform(transform, ndim);

  const spatialIndices: number[] = [];
  const otherIndices: number[] = [];
  dims.forEach((dim, index) => {
    if (SPATIAL_DIMS.includes(dim)) spatialIndices.push(index);
    else otherIndices.push(index);
  });

  if (spatialIndices.length === 0) {
    throw new Error(`no spatial axes among dims [${dims.join(", ")}]`);
  }

  // A transform that mixes spatial and non-spatial axes has no ITK
  // equivalent, and silently dropping the coupling would move the image.
  for (const row of spatialIndices) {
    for (const col of otherIndices) {
      if (homogeneous[row][col] !== 0) {
        const names = otherIndices.map((i) => dims[i]).join(", ");
        throw new Error(
          `transformation couples spatial and non-spatial axes [${names}]; ` +
            `it cannot be expressed as an ITK spatial transform`,
        );
      }
    }
  }

  // RFC-5 (Zarr) order -> ITK (fastest-axis-first) order: reverse both the
  // row and the column ordering of the spatial block.
  const reversed = [...spatialIndices].reverse();
  const matrix = reversed.map((row) =>
    reversed.map((col) => homogeneous[row][col])
  );
  const offset = reversed.map((row) => homogeneous[row][ndim]);

  return { matrix, offset };
}

/**
 * Convert an RFC-5 transformation to an ITK-Wasm transform list.
 *
 * The transformation is collapsed into a single `Affine` entry, so the result
 * is independent of ITK's own list-composition order.
 *
 * @param transform An RFC-5 coordinate transformation describing a linear map.
 * @param dims The coordinate system's axis names, in RFC-5 (Zarr) order.
 * @returns A single-entry ITK-Wasm `TransformList`.
 */
export function ngffTransformToItkTransform(
  transform: V06Transform,
  dims: string[],
): TransformList {
  const { matrix, offset } = ngffTransformToItkMatrix(transform, dims);
  const dimension = offset.length;

  // ITK's MatrixOffsetTransformBase packs the row-major matrix followed by the
  // translation, with the center of rotation as the fixed parameters. An
  // RFC-5 affine has no center, so it stays at the origin.
  const parameters = new Float64Array(dimension * dimension + dimension);
  for (let row = 0; row < dimension; row++) {
    for (let col = 0; col < dimension; col++) {
      parameters[row * dimension + col] = matrix[row][col];
    }
  }
  parameters.set(offset, dimension * dimension);

  const itkTransform: Transform = {
    transformType: {
      transformParameterization: "Affine",
      parametersValueType: "float64",
      inputDimension: dimension,
      outputDimension: dimension,
    },
    name: "AffineTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: dimension,
    numberOfParameters: parameters.length,
    fixedParameters: new Float64Array(dimension),
    parameters,
    metadata: new Map(),
  } as unknown as Transform;

  return [itkTransform];
}
