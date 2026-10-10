// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * A moving-image box that every point of a block lands in, for an ITK
 * transform list.
 *
 * The box is carried through the list one stage at a time, ITK's last entry
 * first. A linear stage maps the box's corners. A B-spline stage samples the
 * transform on a lattice over the box and widens the lattice's box by the
 * interpolation error bound, read off the second differences of the control
 * points, then intersects the result with the signed range of the control
 * points; where the box leaves the B-spline's domain, ITK displaces nothing,
 * so the signed range with zero in it is taken alone. A displacement-field
 * stage takes the box of the landing points of the samples under the box.
 * Every rule holds because a point moves by a convex combination of the
 * samples or control points around it.
 *
 * @module
 */

import type { Transform, TransformList } from "itk-wasm";
import {
  asNumbers,
  linearStage,
} from "../utils/itk_transform_to_ngff_transform.ts";

/** Upper bound on lattice points per stage; the step grows past it. */
const MAX_POINTS = 1 << 14;
/** Tolerance, in samples, for a box edge that lies on a field's lattice. */
const ON_LATTICE = 1e-6;

export interface Box {
  low: number[];
  high: number[];
}

interface Lattice {
  size: number[];
  origin: number[];
  spacing: number[];
  /** Row-major `dimension x dimension`. */
  direction: number[];
}

function latticeGeometry(
  entry: Transform,
  dimension: number,
): Lattice | undefined {
  const fixed = asNumbers(entry.fixedParameters);
  if (!entry.parameters || fixed.length !== dimension * (3 + dimension)) {
    return undefined;
  }
  const size = fixed.slice(0, dimension).map((value) => Math.trunc(value));
  const spacing = fixed.slice(2 * dimension, 3 * dimension);
  if (size.some((value) => value < 1) || spacing.some((value) => value <= 0)) {
    return undefined;
  }
  return {
    size,
    origin: fixed.slice(dimension, 2 * dimension),
    spacing,
    direction: fixed.slice(3 * dimension),
  };
}

/** The 2^d corners of a box. */
export function corners(low: number[], high: number[]): number[][] {
  const dimension = low.length;
  const result: number[][] = [];
  for (let bit = 0; bit < 1 << dimension; bit++) {
    result.push(
      low.map((value, axis) => (bit & (1 << axis) ? high[axis] : value)),
    );
  }
  return result;
}

export function boxOf(points: number[][]): Box {
  const dimension = points[0].length;
  const low = new Array(dimension).fill(Infinity);
  const high = new Array(dimension).fill(-Infinity);
  for (const point of points) {
    for (let axis = 0; axis < dimension; axis++) {
      if (point[axis] < low[axis]) low[axis] = point[axis];
      if (point[axis] > high[axis]) high[axis] = point[axis];
    }
  }
  return { low, high };
}

/** Inverse of a small row-major matrix by Gauss-Jordan elimination. */
export function inverse(matrix: number[], dimension: number): number[] {
  const a = Array.from({ length: dimension }, (_, row) =>
    matrix.slice(row * dimension, (row + 1) * dimension),
  );
  const b = Array.from({ length: dimension }, (_, row) =>
    Array.from({ length: dimension }, (_, col) => (row === col ? 1 : 0)),
  );
  for (let col = 0; col < dimension; col++) {
    let pivot = col;
    for (let row = col + 1; row < dimension; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    [b[col], b[pivot]] = [b[pivot], b[col]];
    const scale = a[col][col];
    if (scale === 0) throw new Error("singular direction matrix");
    for (let k = 0; k < dimension; k++) {
      a[col][k] /= scale;
      b[col][k] /= scale;
    }
    for (let row = 0; row < dimension; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let k = 0; k < dimension; k++) {
        a[row][k] -= factor * a[col][k];
        b[row][k] -= factor * b[col][k];
      }
    }
  }
  return b.flat();
}

/** Per row, the sum of `|matrix| * weights` over the columns. */
export function absRowSums(
  matrix: number[],
  dimension: number,
  weights: number[] = new Array(dimension).fill(1),
): number[] {
  return Array.from({ length: dimension }, (_, row) => {
    let sum = 0;
    for (let col = 0; col < dimension; col++) {
      sum += Math.abs(matrix[row * dimension + col]) * weights[col];
    }
    return sum;
  });
}

/** `matrix @ vector` for a row-major matrix. */
export function apply(matrix: number[], vector: number[]): number[] {
  const dimension = vector.length;
  return Array.from({ length: dimension }, (_, row) => {
    let sum = 0;
    for (let col = 0; col < dimension; col++) {
      sum += matrix[row * dimension + col] * vector[col];
    }
    return sum;
  });
}

/** A physical point's continuous index on a lattice. */
function toIndex(point: number[], lattice: Lattice, inv: number[]): number[] {
  const shifted = point.map((value, axis) => value - lattice.origin[axis]);
  return apply(inv, shifted).map(
    (value, axis) => value / lattice.spacing[axis],
  );
}

/** A continuous index's physical point on a lattice. */
function toPoint(index: number[], lattice: Lattice): number[] {
  const scaled = index.map((value, axis) => value * lattice.spacing[axis]);
  return apply(lattice.direction, scaled).map(
    (value, axis) => value + lattice.origin[axis],
  );
}

/**
 * ITK's cubic B-spline weights for the four support points of a cell, from
 * the fractional position `t` in `[0, 1)`. The support starts one control
 * point below the cell.
 */
export function bsplineWeights(t: number): [number, number, number, number] {
  const one = 1 - t;
  return [
    (one * one * one) / 6,
    (3 * t * t * t - 6 * t * t + 4) / 6,
    (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6,
    (t * t * t) / 6,
  ];
}

/**
 * Displacement of one point at continuous grid `index` through a cubic
 * B-spline whose `parameters` are ITK's, one component after the other, x
 * fastest within each. Zero outside ITK's valid region, where ITK returns the
 * point unchanged.
 */
export function bsplineDisplacement(
  parameters: ArrayLike<number>,
  size: number[],
  index: number[],
): number[] {
  const dimension = size.length;
  const perComponent = size.reduce((product, extent) => product * extent, 1);
  const out = new Array(dimension).fill(0);
  // ITK's valid region: one control point in from the low edge, two from the
  // high edge, the high limit itself nudged inside.
  for (let axis = 0; axis < dimension; axis++) {
    if (index[axis] < 1 || index[axis] > size[axis] - 2) return out;
  }
  const first: number[] = [];
  const weights: number[][] = [];
  for (let axis = 0; axis < dimension; axis++) {
    const cell = Math.floor(Math.min(index[axis], size[axis] - 2 - 1e-9));
    first.push(cell - 1);
    weights.push(bsplineWeights(Math.min(Math.max(index[axis] - cell, 0), 1)));
  }
  const strides = size.map((_, axis) =>
    size.slice(0, axis).reduce((product, extent) => product * extent, 1),
  );
  const taps = 4 ** dimension;
  for (let tap = 0; tap < taps; tap++) {
    let weight = 1;
    let offset = 0;
    let rest = tap;
    for (let axis = 0; axis < dimension; axis++) {
      const k = rest % 4;
      rest = Math.floor(rest / 4);
      weight *= weights[axis][k];
      offset += (first[axis] + k) * strides[axis];
    }
    for (let c = 0; c < dimension; c++) {
      out[c] += weight * Number(parameters[c * perComponent + offset]);
    }
  }
  return out;
}

/** Linear index of a grid point, x fastest, from an index `at` in grid order. */
function flatIndex(at: number[], size: number[]): number {
  let offset = 0;
  let stride = 1;
  for (let axis = 0; axis < size.length; axis++) {
    offset += at[axis] * stride;
    stride *= size[axis];
  }
  return offset;
}

/** Every index of the box `[first, last]`, in grid order. */
function* gridIndices(first: number[], last: number[]): Generator<number[]> {
  const at = [...first];
  const dimension = first.length;
  while (true) {
    yield [...at];
    let axis = 0;
    for (; axis < dimension; axis++) {
      at[axis]++;
      if (at[axis] <= last[axis]) break;
      at[axis] = first[axis];
    }
    if (axis === dimension) return;
  }
}

/** `max |Δ_i Δ_m c|` of one component over a band of control points. */
function secondDifference(
  parameters: ArrayLike<number>,
  size: number[],
  component: number,
  first: number[],
  last: number[],
  i: number,
  m: number,
): number {
  const perComponent = size.reduce((product, extent) => product * extent, 1);
  const base = component * perComponent;
  const value = (at: number[]) =>
    Number(parameters[base + flatIndex(at, size)]);
  let largest = 0;
  const stop = last.map((value, axis) => {
    let end = value;
    if (axis === i) end -= i === m ? 2 : 1;
    if (axis === m && i !== m) end -= 1;
    return end;
  });
  if (stop.some((value, axis) => value < first[axis])) return 0;
  for (const at of gridIndices(first, stop)) {
    let difference: number;
    if (i === m) {
      const up = [...at];
      up[i] += 1;
      const upper = [...at];
      upper[i] += 2;
      difference = value(upper) - 2 * value(up) + value(at);
    } else {
      const ai = [...at];
      ai[i] += 1;
      const am = [...at];
      am[m] += 1;
      const both = [...at];
      both[i] += 1;
      both[m] += 1;
      difference = value(both) - value(ai) - value(am) + value(at);
    }
    largest = Math.max(largest, Math.abs(difference));
  }
  return largest;
}

/** A linear map as a row-major matrix and an offset. */
interface Affine {
  matrix: number[];
  offset: number[];
}

function mapPoint(point: number[], affine: Affine): number[] {
  return apply(affine.matrix, point).map(
    (value, row) => value + affine.offset[row],
  );
}

/** The box of a box's corners through a linear map. */
function mappedBox(box: Box, affine: Affine): Box {
  return boxOf(
    corners(box.low, box.high).map((corner) => mapPoint(corner, affine)),
  );
}

/** `left @ right` for row-major matrices. */
function compose(left: number[], right: number[], dimension: number): number[] {
  const result = new Array(dimension * dimension).fill(0);
  for (let row = 0; row < dimension; row++) {
    for (let col = 0; col < dimension; col++) {
      let sum = 0;
      for (let k = 0; k < dimension; k++) {
        sum += left[row * dimension + k] * right[k * dimension + col];
      }
      result[row * dimension + col] = sum;
    }
  }
  return result;
}

/**
 * The box through a cubic B-spline stage. `box` bounds the block in the fixed
 * image and `affine` is the linear stages applied since, so the lattice is laid
 * over the block itself and mapped exactly: the block's rotation does not widen
 * the box.
 */
function bsplineStage(
  entry: Transform,
  dimension: number,
  box: Box,
  affine: Affine,
  voxel: number[],
): Box | undefined {
  const lattice = latticeGeometry(entry, dimension);
  if (lattice === undefined || !entry.parameters) return undefined;
  const { size, spacing, direction } = lattice;
  const perComponent = size.reduce((product, extent) => product * extent, 1);
  if (entry.parameters.length !== dimension * perComponent) return undefined;
  const parameters = entry.parameters as unknown as ArrayLike<number>;
  const inv = inverse(direction, dimension);
  const mapped = mappedBox(box, affine);
  const indexBox = boxOf(
    corners(mapped.low, mapped.high).map((corner) =>
      toIndex(corner, lattice, inv),
    ),
  );
  const inside =
    indexBox.low.every((value) => value >= 1) &&
    indexBox.high.every((value, axis) => value <= size[axis] - 2);

  // The signed range of the control points whose support meets the box.
  const first = indexBox.low.map((value, axis) =>
    Math.min(Math.max(Math.floor(value) - 1, 0), size[axis] - 1),
  );
  const last = indexBox.high.map((value, axis) =>
    Math.min(Math.max(Math.floor(value) + 2, 0), size[axis] - 1),
  );
  const rangeLow = new Array(dimension).fill(Infinity);
  const rangeHigh = new Array(dimension).fill(-Infinity);
  for (const at of gridIndices(first, last)) {
    const offset = flatIndex(at, size);
    for (let c = 0; c < dimension; c++) {
      const value = Number(parameters[c * perComponent + offset]);
      if (value < rangeLow[c]) rangeLow[c] = value;
      if (value > rangeHigh[c]) rangeHigh[c] = value;
    }
  }
  if (!inside) {
    return {
      low: mapped.low.map((value, c) => value + Math.min(rangeLow[c], 0)),
      high: mapped.high.map((value, c) => value + Math.max(rangeHigh[c], 0)),
    };
  }

  // The second derivative of the displacement along a fixed axis, bounded by
  // the second differences of the control points in the band: the index of a
  // fixed point moves by `derivative` per unit of fixed coordinate.
  const derivative = compose(inv, affine.matrix, dimension).map(
    (value, flat) => value / spacing[Math.floor(flat / dimension)],
  );
  const factor: number[][] = Array.from({ length: dimension }, () =>
    new Array(dimension).fill(0),
  );
  for (let c = 0; c < dimension; c++) {
    const bound: number[][] = Array.from({ length: dimension }, () =>
      new Array(dimension).fill(0),
    );
    for (let i = 0; i < dimension; i++) {
      for (let m = i; m < dimension; m++) {
        const value = secondDifference(parameters, size, c, first, last, i, m);
        bound[i][m] = value;
        bound[m][i] = value;
      }
    }
    for (let j = 0; j < dimension; j++) {
      let sum = 0;
      for (let i = 0; i < dimension; i++) {
        for (let m = 0; m < dimension; m++) {
          sum +=
            Math.abs(derivative[i * dimension + j]) *
            Math.abs(derivative[m * dimension + j]) *
            bound[i][m];
        }
      }
      factor[c][j] = sum;
    }
  }

  // The lattice step: half a control cell at most, the voxel at least, and in
  // between the coarsest whose margin stays within a voxel.
  const cell = absRowSums(inverse(derivative, dimension), dimension);
  const nominal = cell.map((value, axis) => Math.max(value / 2, voxel[axis]));
  const budget = Math.min(...voxel);
  const worst = nominal.map((_, j) => Math.max(...factor.map((row) => row[j])));
  const total = nominal.reduce(
    (sum, value, j) => sum + value * value * worst[j],
    0,
  );
  const scale = total <= 0 ? 1 : Math.min(1, Math.sqrt((8 * budget) / total));
  const step = nominal.map((value, axis) =>
    Math.max(voxel[axis], scale * value),
  );
  const span = box.high.map((value, axis) => value - box.low[axis]);
  let count = span.map((value, axis) =>
    value > 0 ? Math.max(1, Math.ceil(value / step[axis] - 1e-9)) : 0,
  );
  while (
    count.reduce((product, value) => product * (value + 1), 1) > MAX_POINTS
  ) {
    count = count.map((value) => Math.max(1, Math.floor(value / 2)));
  }
  const actual = count.map((value, axis) =>
    value > 0 ? span[axis] / value : 0,
  );
  const latticeLow = new Array(dimension).fill(Infinity);
  const latticeHigh = new Array(dimension).fill(-Infinity);
  for (const at of gridIndices(new Array(dimension).fill(0), count)) {
    const point = mapPoint(
      at.map((value, axis) => box.low[axis] + value * actual[axis]),
      affine,
    );
    const displacement = bsplineDisplacement(
      parameters,
      size,
      toIndex(point, lattice, inv),
    );
    for (let axis = 0; axis < dimension; axis++) {
      const landing = point[axis] + displacement[axis];
      if (landing < latticeLow[axis]) latticeLow[axis] = landing;
      if (landing > latticeHigh[axis]) latticeHigh[axis] = landing;
    }
  }
  // Interpolation error between lattice points: s^2/8 times the bound.
  const margin = factor.map((row) =>
    row.reduce(
      (sum, value, j) => sum + ((actual[j] * actual[j]) / 8) * value,
      0,
    ),
  );
  return {
    low: mapped.low.map((value, c) =>
      Math.max(latticeLow[c] - margin[c], value + rangeLow[c]),
    ),
    high: mapped.high.map((value, c) =>
      Math.min(latticeHigh[c] + margin[c], value + rangeHigh[c]),
    ),
  };
}

function fieldStage(
  entry: Transform,
  dimension: number,
  box: Box,
): Box | undefined {
  const lattice = latticeGeometry(entry, dimension);
  if (lattice === undefined || !entry.parameters) return undefined;
  const { size, spacing, direction } = lattice;
  const samples = size.reduce((product, extent) => product * extent, 1);
  if (entry.parameters.length !== dimension * samples) return undefined;
  const parameters = entry.parameters as unknown as ArrayLike<number>;
  const inv = inverse(direction, dimension);
  const indexBox = boxOf(
    corners(box.low, box.high).map((corner) => toIndex(corner, lattice, inv)),
  );
  // A box on the field's own lattice computes to within rounding of whole
  // indices; without the tolerance each side would pick up a sample the box
  // does not reach.
  const first = indexBox.low.map((value, axis) =>
    Math.min(Math.max(Math.floor(value + ON_LATTICE), 0), size[axis] - 1),
  );
  const last = indexBox.high.map((value, axis) =>
    Math.min(Math.max(Math.ceil(value - ON_LATTICE), 0), size[axis] - 1),
  );
  const outside =
    indexBox.low.some((value) => value < -ON_LATTICE) ||
    indexBox.high.some((value, axis) => value > size[axis] - 1 + ON_LATTICE);
  // Two boxes hold every point of the block: the landing points of the samples
  // around it, exact when the field is at the block's resolution, and the box
  // plus the signed range of those samples, exact when the field is coarse and
  // smooth. Their intersection holds every point too.
  const landLow = new Array(dimension).fill(Infinity);
  const landHigh = new Array(dimension).fill(-Infinity);
  let rangeLow = new Array(dimension).fill(Infinity);
  let rangeHigh = new Array(dimension).fill(-Infinity);
  for (const at of gridIndices(first, last)) {
    const point = toPoint(at, lattice);
    const offset = flatIndex(at, size) * dimension;
    for (let axis = 0; axis < dimension; axis++) {
      const displacement = Number(parameters[offset + axis]);
      const landing = point[axis] + displacement;
      if (landing < landLow[axis]) landLow[axis] = landing;
      if (landing > landHigh[axis]) landHigh[axis] = landing;
      if (displacement < rangeLow[axis]) rangeLow[axis] = displacement;
      if (displacement > rangeHigh[axis]) rangeHigh[axis] = displacement;
    }
  }
  let low = landLow;
  let high = landHigh;
  if (outside) {
    // ITK displaces a point up to half a sample past the lattice by what the
    // nearest sample holds, and any further point by nothing.
    const half = absRowSums(direction, dimension, spacing).map((v) => v / 2);
    low = low.map((value, axis) => Math.min(box.low[axis], value - half[axis]));
    high = high.map((value, axis) =>
      Math.max(box.high[axis], value + half[axis]),
    );
    rangeLow = rangeLow.map((value) => Math.min(value, 0));
    rangeHigh = rangeHigh.map((value) => Math.max(value, 0));
  }
  return {
    low: low.map((value, axis) =>
      Math.max(value, box.low[axis] + rangeLow[axis]),
    ),
    high: high.map((value, axis) =>
      Math.min(value, box.high[axis] + rangeHigh[axis]),
    ),
  };
}

/**
 * The box in the last stage's output space, or `undefined` for a stage of a
 * kind this cannot size. ITK applies the last entry of a list first. Linear
 * stages are composed and applied to the points a B-spline stage samples, so
 * a rotation ahead of a B-spline does not widen its box; they map the box's
 * corners otherwise.
 *
 * @param voxel The fixed image's voxel extent per physical axis, which caps
 *   how fine a lattice gets.
 */
export function transformBox(
  transformList: TransformList,
  dimension: number,
  low: number[],
  high: number[],
  voxel: number[],
): Box | undefined {
  const identity = (): Affine => ({
    matrix: Array.from(
      { length: dimension ** 2 },
      (_, i) => +(i % (dimension + 1) === 0),
    ),
    offset: new Array(dimension).fill(0),
  });
  let box: Box = { low: [...low], high: [...high] };
  let affine = identity();
  for (let position = transformList.length - 1; position >= 0; position--) {
    const entry = transformList[position];
    const name = String(entry.transformType.transformParameterization);
    // A parameterless 'Composite' entry is a pipeline grouping header or a
    // nested composite whose children the serialization dropped
    // (InsightSoftwareConsortium/ITK#6792); the two cannot be told apart, so
    // the pipeline sizes such a list.
    if (name === "Composite") return undefined;
    let next: Box | undefined;
    if (name === "BSpline") {
      next = bsplineStage(entry, dimension, box, affine, voxel);
    } else if (name === "DisplacementField") {
      next = fieldStage(entry, dimension, mappedBox(box, affine));
    } else {
      const stage = linearStage(entry, dimension);
      if (stage === undefined) return undefined;
      const matrix = stage.matrix.flat();
      affine = {
        matrix: compose(matrix, affine.matrix, dimension),
        offset: apply(matrix, affine.offset).map(
          (value, row) => value + stage.offset[row],
        ),
      };
      continue;
    }
    if (next === undefined) return undefined;
    box = next;
    affine = identity();
  }
  return mappedBox(box, affine);
}
