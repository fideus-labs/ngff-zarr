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
 *   fastest-axis-first by name: x, then y, then z. Writing `P` for the
 *   permutation sending an ITK-order vector to the `dims`-order vector, an
 *   RFC-5 affine `q = M p + b` becomes `A = P^T M P` and `t = P^T b` in
 *   ITK. For the canonical `zyx` order `P` is the axis reversal.
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
import type { NgffImage } from "../types/ngff_image.ts";
import {
  changeOfFrame,
  optionalFrameGeometry,
  transposed,
} from "./itk_direction.ts";
import type {
  ByDimension,
  ByDimensionItem,
  V06Transform,
} from "../types/zarr_metadata.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

/** A square matrix stored as an array of rows. */
type Matrix = number[][];

function zeroMatrix(size: number): Matrix {
  return Array.from(
    { length: size },
    () => Array.from({ length: size }, () => 0),
  );
}

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

/**
 * A matrix's shape for an error message. Naming row 0's length alone would
 * describe a ragged matrix as its own requirement ("is 2x3 but requires
 * 2x3"), hiding the short row the message exists to point at.
 */
function describeShape(rows: number[][]): string {
  const lengths = new Set(rows.map((row) => row.length));
  if (lengths.size <= 1) {
    return `${rows.length}x${rows[0]?.length ?? 0}`;
  }
  return `${rows.length} rows of lengths ${
    rows.map((row) => row.length).join(", ")
  }`;
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
          `rotation transformation is ${describeShape(rotation)} but the ` +
            `coordinate system has ${ndim} axes`,
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
          `affine transformation is ${describeShape(affine)} but ` +
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

    case "mapAxis": {
      if (transform.mapAxis.length !== ndim) {
        throw new Error(
          `mapAxis transformation permutes ${transform.mapAxis.length} axes ` +
            `but the coordinate system has ${ndim}`,
        );
      }
      const matrix = zeroMatrix(ndim + 1);
      matrix[ndim][ndim] = 1;
      // The value at position i names the input axis that becomes output
      // axis i, so the row for output i selects that input.
      transform.mapAxis.forEach((inputAxis, outputAxis) => {
        matrix[outputAxis][inputAxis] = 1;
      });
      return matrix;
    }

    case "byDimension":
      return homogeneousFromByDimension(transform, ndim);

    case "bijection":
      // The forward direction is the mapping; RFC-5 keeps the inverse
      // alongside it so a reader need not invert the forward one, which is
      // what ITK does for an affine anyway.
      return homogeneousFromTransform(transform.forward, ndim);

    default:
      throw new Error(
        `transformation type '${
          (transform as { type: string }).type
        }' cannot be converted to an ITK transform. identity, scale, ` +
          `translation, rotation, affine, mapAxis, byDimension, bijection ` +
          `and sequences of them describe a linear mapping that ITK can ` +
          `represent as a single affine transform; displacements and ` +
          `coordinates convert to a displacement field, given the field ` +
          `they point at.`,
      );
  }
}

/**
 * Assemble a byDimension transformation into one homogeneous matrix.
 *
 * Each item is a lower-dimensional transformation between two subsets of
 * axes, so its own matrix is written into the rows its `outputAxes` name and
 * the columns its `inputAxes` name. Axes no item produces would leave a zero
 * row, which collapses the image rather than transforming it, so a gap is
 * refused here rather than resampled.
 */
function homogeneousFromByDimension(
  transform: ByDimension,
  ndim: number,
): Matrix {
  const matrix = zeroMatrix(ndim + 1);
  matrix[ndim][ndim] = 1;
  const produced = new Set<number>();

  for (const item of transform.transformations) {
    const block = byDimensionItemBlock(item, ndim);
    item.outputAxes.forEach((outputAxis, row) => {
      item.inputAxes.forEach((inputAxis, col) => {
        matrix[outputAxis][inputAxis] = block[row][col];
      });
      matrix[outputAxis][ndim] = block[row][item.inputAxes.length];
      produced.add(outputAxis);
    });
  }

  const missing = Array.from({ length: ndim }, (_, axis) => axis)
    .filter((axis) => !produced.has(axis));
  if (missing.length > 0) {
    throw new Error(
      `byDimension transformation produces output axes ` +
        `[${Array.from(produced).sort((a, b) => a - b).join(", ")}], leaving ` +
        `[${missing.join(", ")}] of the ${ndim} axes of the coordinate ` +
        `system unset; every output axis must be produced by exactly one item`,
    );
  }
  return matrix;
}

/** One byDimension item as a homogeneous matrix over its own axes. */
function byDimensionItemBlock(item: ByDimensionItem, ndim: number): Matrix {
  if (item.inputAxes.length !== item.outputAxes.length) {
    throw new Error(
      `byDimension item of type '${item.transformation.type}' maps ` +
        `${item.inputAxes.length} input axes to ${item.outputAxes.length} ` +
        `output axes; only a square mapping converts to an ITK transform`,
    );
  }
  const beyond = [...item.inputAxes, ...item.outputAxes]
    .filter((axis) => axis >= ndim);
  if (beyond.length > 0) {
    throw new Error(
      `byDimension axis indices [${beyond.join(", ")}] exceed the ${ndim} ` +
        `axes of the coordinate system`,
    );
  }
  if (new Set(item.inputAxes).size !== item.inputAxes.length) {
    throw new Error(
      `byDimension input axes [${item.inputAxes.join(", ")}] name an axis ` +
        `twice`,
    );
  }
  return homogeneousFromTransform(item.transformation, item.inputAxes.length);
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
 * Internal: {@link ngffTransformToItkTransform} is the public entry point, and
 * nothing outside this package needs the raw numbers, since ITK is what the
 * caller wants to hand them to. Exported for the tests, which use it as a fine
 * probe on the axis and composition conventions.
 *
 * @internal
 * @param transform An RFC-5 (OME-Zarr v0.6) coordinate transformation. It must
 *   describe a linear mapping: `identity`, `scale`, `translation`, `rotation`,
 *   `affine`, `mapAxis`, `byDimension`, `bijection`, or a `sequence` of those.
 * @param dims The axis names of the coordinate system the transformation is
 *   defined on, in RFC-5 (Zarr) order, e.g. `["z", "y", "x"]`.
 * @returns The matrix and offset for the spatial axes, in ITK order. ITK has no
 *   notion of a non-spatial axis, so a component acting purely on `t` or `c` (a
 *   frame interval, say) is *projected away*. That is lossless for the spatial
 *   mapping, which is all an ITK transform describes, but the result is not a
 *   faithful copy of the input. A component that *couples* the two kinds of
 *   axis is refused instead, because dropping it would move the image.
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

  // RFC-5 (`dims`) order -> ITK (fastest-axis-first) order. ITK orders
  // components by name (x, then y, then z), so the mapping is built by name
  // rather than by reversing, which is only equivalent for the canonical
  // (z, y, x) order.
  const spatialNames = spatialIndices.map((index) => dims[index]);
  const itkIndices = SPATIAL_DIMS
    .filter((dim) => spatialNames.includes(dim))
    .map((dim) => spatialIndices[spatialNames.indexOf(dim)]);
  const matrix = itkIndices.map((row) =>
    itkIndices.map((col) => homogeneous[row][col])
  );
  const offset = itkIndices.map((row) => homogeneous[row][ndim]);

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
  frames: { fixed?: NgffImage; moving?: NgffImage } = {},
): TransformList {
  if (transform.type === "displacements" || transform.type === "coordinates") {
    throw new Error(
      `the ${transform.type} transform points at '${transform.path}'; ` +
        "convert it with ngffDisplacementFieldToItkTransform, passing the " +
        "field loaded from that path",
    );
  }
  let { matrix, offset } = ngffTransformToItkMatrix(transform, dims);
  const dimension = offset.length;

  const spatial = dims.filter((dim) => SPATIAL_DIMS.includes(dim));
  const itkDims = SPATIAL_DIMS.filter((dim) => spatial.includes(dim));
  const geometry = optionalFrameGeometry(frames.fixed, frames.moving, itkDims);
  if (geometry !== undefined) {
    // The intrinsic systems -> ITK physical space: phi_m . T . phi_f^-1,
    // which is the same change of frame with the directions inverted.
    ({ matrix, offset } = changeOfFrame(
      matrix,
      offset,
      transposed(geometry.directionFixed),
      transposed(geometry.directionMoving),
      geometry.originFixed,
      geometry.originMoving,
    ));
  }

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
