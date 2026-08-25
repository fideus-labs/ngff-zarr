// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * The ITK direction matrix an `NgffImage`'s RFC-4 orientation induces.
 *
 * Shared between the resample bounding box and the transform converters,
 * which both need the exact matrix `ngffImageToItkImage` would build.
 */

import type { NgffImage } from "../types/ngff_image.ts";
import { anatomicalOrientationToItkDirection } from "../types/rfc4.ts";

export function identityDirection(dimension: number): Float64Array {
  const direction = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension; i++) direction[i * dimension + i] = 1.0;
  return direction;
}

/**
 * Direction matrix from RFC-4 orientation, matching `ngffImageToItkImage`.
 *
 * All-or-nothing: unless every spatial axis carries an orientation that maps
 * onto an LPS axis, the direction falls back to identity.
 */
export function itkDirection(
  image: NgffImage,
  itkDims: string[],
): Float64Array {
  const dimension = itkDims.length;
  const direction = identityDirection(dimension);

  const orientations = image.axesOrientations;
  if (!orientations) return direction;

  const columns: number[][] = [];
  const seenAxes = new Set<number>();
  for (const dim of itkDims) {
    const orientation = orientations[dim];
    if (orientation === undefined) return direction;
    const column = anatomicalOrientationToItkDirection(orientation.value);
    if (column === undefined) return direction;
    // A column pointing outside the matrix's dimension (e.g. a
    // superior/inferior orientation on a 2D image), or two dims mapping onto
    // the same LPS axis, would truncate to a singular matrix; keep the
    // identity fallback instead.
    if (column.slice(dimension).some((component) => component !== 0)) {
      return direction;
    }
    let axis = 0;
    for (let i = 1; i < column.length; i++) {
      if (Math.abs(column[i]) > Math.abs(column[axis])) axis = i;
    }
    if (seenAxes.has(axis)) return direction;
    seenAxes.add(axis);
    columns.push(column);
  }

  for (let col = 0; col < dimension; col++) {
    for (let row = 0; row < dimension; row++) {
      direction[row * dimension + col] = columns[col][row];
    }
  }
  return direction;
}

/** Row-major `Float64Array` direction as an array of rows. */
export function directionRows(
  direction: Float64Array,
  dimension: number,
): number[][] {
  return Array.from(
    { length: dimension },
    (_, row) =>
      Array.from(
        { length: dimension },
        (_, col) => direction[row * dimension + col],
      ),
  );
}

/** RFC-4 directions are orthogonal, so the inverse is the transpose. */
export function transposed(matrix: number[][]): number[][] {
  return matrix.map((row, i) => row.map((_, j) => matrix[j][i]));
}

function matmul(left: number[][], right: number[][]): number[][] {
  return left.map((row) =>
    right[0].map((_, col) =>
      row.reduce((sum, value, k) => sum + value * right[k][col], 0)
    )
  );
}

function matvec(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) =>
    row.reduce((sum, value, col) => sum + value * vector[col], 0)
  );
}

/** The direction matrices and origins of an image pair, in ITK order. */
export interface FrameGeometry {
  directionFixed: number[][];
  directionMoving: number[][];
  originFixed: number[];
  originMoving: number[];
}

/**
 * The directions and origins `ngffImageToItkImage` gives the two images.
 *
 * Throws when an image lacks a translation entry for one of `itkDims`.
 */
export function frameGeometry(
  fixed: NgffImage,
  moving: NgffImage,
  itkDims: string[],
): FrameGeometry {
  for (
    const [label, image] of [["fixed", fixed], ["moving", moving]] as const
  ) {
    const missing = itkDims.filter((dim) => !(dim in image.translation));
    if (missing.length > 0) {
      throw new Error(
        `the ${label} image has no translation entry for spatial ` +
          `dimension(s) [${missing.join(", ")}]; its dims do not cover dims`,
      );
    }
  }
  const dimension = itkDims.length;
  return {
    directionFixed: directionRows(itkDirection(fixed, itkDims), dimension),
    directionMoving: directionRows(itkDirection(moving, itkDims), dimension),
    originFixed: itkDims.map((dim) => fixed.translation[dim]),
    originMoving: itkDims.map((dim) => moving.translation[dim]),
  };
}

/**
 * The frame geometry of an optional image pair, or `undefined` for neither.
 *
 * The counterpart of Python's `_check_frame_images`: every converter takes the
 * pair as two optional arguments and every one of them owes the same rule, so
 * it lives here rather than in each of them.
 */
export function optionalFrameGeometry(
  fixed: NgffImage | undefined,
  moving: NgffImage | undefined,
  itkDims: string[],
): FrameGeometry | undefined {
  if ((fixed === undefined) !== (moving === undefined)) {
    throw new Error("pass both fixed and moving, or neither");
  }
  if (fixed === undefined || moving === undefined) return undefined;
  return frameGeometry(fixed, moving, itkDims);
}

/**
 * Re-express `y = A x + t` through a change of frame on each side.
 *
 * `ngffImageToItkImage` builds each image with `origin = translation` and the
 * RFC-4 direction matrix `D`, so an intrinsic point `p` sits at the physical
 * point `phi(p) = D (p - o) + o`. Given a mapping `T` between two frames,
 * this returns `phi_out^-1 . T . phi_in` for the directions and origins
 * supplied:
 *
 *     M = D_out^-1 A D_in
 *     b = D_out^-1 (A (I - D_in) o_in + t - o_out) + o_out
 *
 * `phi^-1` has the same shape as `phi` with `D^-1` in place of `D`, so the
 * same formula serves both conversions: ITK to RFC-5 passes the images'
 * directions, RFC-5 to ITK passes their inverses. The scales cancel on both
 * sides; the translations do not, which is why orientations alone would not
 * be enough. With no anatomical orientation every direction is the identity
 * and the mapping comes back unchanged.
 */
export function changeOfFrame(
  matrix: number[][],
  offset: number[],
  directionIn: number[][],
  directionOut: number[][],
  originIn: number[],
  originOut: number[],
): { matrix: number[][]; offset: number[] } {
  const inverseOut = transposed(directionOut);
  const rotatedIn = matvec(directionIn, originIn);
  const shifted = originIn.map((value, row) => value - rotatedIn[row]);
  const inner = matvec(matrix, shifted).map(
    (value, row) => value + offset[row] - originOut[row],
  );
  return {
    matrix: matmul(matmul(inverseOut, matrix), directionIn),
    offset: matvec(inverseOut, inner).map(
      (value, row) => value + originOut[row],
    ),
  };
}
