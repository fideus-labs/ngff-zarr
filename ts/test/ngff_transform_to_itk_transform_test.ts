// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-5 transformations that are not an affine, converted to ITK.
 *
 * Mirrors `py/test/test_ngff_transform_to_itk_transform.py`. `mapAxis`,
 * `byDimension` and `bijection` describe a linear mapping the way the RFC lets
 * a writer describe it -- as a permutation, as one transformation per group of
 * axes, or alongside its inverse -- rather than as a matrix. Each is folded
 * into the single affine ITK gets. There is no `itk` here to evaluate the
 * result against, so the anchor is an oracle that reads each convention from
 * the RFC rather than from the matrix builder.
 */

import { assertAlmostEquals, assertThrows } from "@std/assert";
import {
  type ByDimensionItem,
  createAffine,
  createBijection,
  createByDimension,
  createIdentity,
  createMapAxis,
  createScale,
  createTransformSequence,
  createTranslation,
  type V06Transform,
} from "../src/mod.ts";
import { ngffTransformToItkMatrix } from "../src/utils/ngff_transform_to_itk_transform.ts";

const ITK_ORDER = ["x", "y", "z"];

/**
 * Evaluate an RFC-5 transformation, independently of the module under test.
 *
 * Every convention it encodes -- the direction a `mapAxis` permutation runs
 * in, the order a `sequence` applies its entries in, which side of a
 * `bijection` is the mapping -- is read straight from the RFC.
 */
function apply(transform: V06Transform, point: number[]): number[] {
  switch (transform.type) {
    case "identity":
      return [...point];
    case "scale":
      return point.map((value, i) => value * transform.scale[i]);
    case "translation":
      return point.map((value, i) => value + transform.translation[i]);
    case "affine":
      return transform.affine.map((row) =>
        row.slice(0, -1).reduce((total, v, i) => total + v * point[i], 0) +
        row[row.length - 1]
      );
    case "mapAxis":
      // The value at position i names the input axis that becomes output i.
      return transform.mapAxis.map((inputAxis) => point[inputAxis]);
    case "byDimension": {
      const result = new Array(point.length).fill(0);
      for (const item of transform.transformations) {
        const sub = apply(
          item.transformation,
          item.input_axes.map((axis) => point[axis]),
        );
        item.output_axes.forEach((axis, position) => {
          result[axis] = sub[position];
        });
      }
      return result;
    }
    case "bijection":
      return apply(transform.forward, point);
    case "sequence":
      return transform.transformations.reduce(
        (current, sub) => apply(sub, current),
        point,
      );
    default:
      throw new Error(`the oracle does not handle ${transform.type}`);
  }
}

/** `values`, given in `dims` order, reordered fastest-axis-first. */
function itkOrder(values: number[], dims: string[]): number[] {
  return ITK_ORDER.filter((dim) => dims.includes(dim))
    .map((dim) => values[dims.indexOf(dim)]);
}

function samplePoints(ndim: number): number[][] {
  const points = [new Array(ndim).fill(0)];
  for (let row = 0; row < 4; row++) {
    points.push(
      Array.from({ length: ndim }, (_, axis) => (row + 1) * 3.5 - axis * 2.25),
    );
  }
  return points;
}

function assertMapsLikeTheOracle(transform: V06Transform, dims: string[]) {
  const { matrix, offset } = ngffTransformToItkMatrix(transform, dims);
  for (const point of samplePoints(dims.length)) {
    const input = itkOrder(point, dims);
    const expected = itkOrder(apply(transform, point), dims);
    for (let row = 0; row < offset.length; row++) {
      const mapped = matrix[row].reduce(
        (total, value, col) => total + value * input[col],
        offset[row],
      );
      assertAlmostEquals(mapped, expected[row], 1e-9);
    }
  }
}

const item = (
  transformation: V06Transform,
  input_axes: number[],
  output_axes: number[],
): ByDimensionItem => ({ transformation, input_axes, output_axes });

Deno.test("a mapAxis reversal maps like the oracle", () => {
  assertMapsLikeTheOracle(createMapAxis([2, 1, 0]), ["z", "y", "x"]);
});

Deno.test("a mapAxis on non-canonical dims binds axes by name", () => {
  assertMapsLikeTheOracle(createMapAxis([1, 2, 0]), ["x", "z", "y"]);
});

Deno.test("a mapAxis rotation is not read backwards", () => {
  // [1, 2, 0] and [2, 0, 1] are inverse permutations, so a conversion that
  // reads the transpose vector backwards passes every symmetric case and
  // fails this one.
  const dims = ["z", "y", "x"];
  const { matrix, offset } = ngffTransformToItkMatrix(
    createMapAxis([1, 2, 0]),
    dims,
  );
  const input = itkOrder([1, 2, 3], dims);
  const expected = itkOrder([2, 3, 1], dims);
  for (let row = 0; row < offset.length; row++) {
    const mapped = matrix[row].reduce(
      (total, value, col) => total + value * input[col],
      offset[row],
    );
    assertAlmostEquals(mapped, expected[row], 1e-9);
  }
});

Deno.test("a byDimension of a translation and a scale maps like the oracle", () => {
  assertMapsLikeTheOracle(
    createByDimension([
      item(createTranslation([7]), [0], [0]),
      item(createScale([2, 3]), [1, 2], [1, 2]),
    ]),
    ["z", "y", "x"],
  );
});

Deno.test("a byDimension that permutes maps like the oracle", () => {
  assertMapsLikeTheOracle(
    createByDimension([
      item(createScale([4]), [0], [2]),
      item(createAffine([[1, 0.5, 3], [0, 2, -1]]), [1, 2], [0, 1]),
    ]),
    ["z", "y", "x"],
  );
});

Deno.test("a byDimension holding a mapAxis maps like the oracle", () => {
  assertMapsLikeTheOracle(
    createByDimension([
      item(createMapAxis([1, 0]), [0, 1], [0, 1]),
      item(createTranslation([-5]), [2], [2]),
    ]),
    ["z", "y", "x"],
  );
});

Deno.test("a bijection maps like its forward direction", () => {
  assertMapsLikeTheOracle(
    createBijection(createScale([2, 4]), createScale([0.5, 0.25])),
    ["y", "x"],
  );
});

Deno.test("a bijection ignores an inverse that disagrees", () => {
  // RFC-5 does not require the two directions to be consistent, and the
  // forward one is the mapping.
  const { matrix } = ngffTransformToItkMatrix(
    createBijection(createScale([2, 2]), createScale([9, 9])),
    ["y", "x"],
  );
  assertAlmostEquals(matrix[0][0], 2, 1e-12);
});

Deno.test("a sequence of the new types maps like the oracle", () => {
  assertMapsLikeTheOracle(
    createTransformSequence([
      createMapAxis([1, 0]),
      createTranslation([10, -20]),
      createBijection(
        createAffine([[0.8, -0.6, 1], [0.6, 0.8, 2]]),
        createIdentity(),
      ),
    ]),
    ["y", "x"],
  );
});

Deno.test("a mapAxis length must match the coordinate system", () => {
  assertThrows(
    () => ngffTransformToItkMatrix(createMapAxis([1, 0]), ["z", "y", "x"]),
    Error,
    "permutes 2 axes",
  );
});

Deno.test("a mapAxis that couples a channel axis is refused", () => {
  assertThrows(
    () => ngffTransformToItkMatrix(createMapAxis([2, 1, 0]), ["c", "y", "x"]),
    Error,
    "couples spatial and non-spatial",
  );
});

Deno.test("a byDimension may leave a non-spatial axis to itself", () => {
  // A time scale is projected away, exactly as it is for an affine: it is not
  // part of the spatial mapping an ITK transform describes.
  const { matrix, offset } = ngffTransformToItkMatrix(
    createByDimension([
      item(createScale([0.25]), [0], [0]),
      item(createTranslation([3, -4]), [1, 2], [1, 2]),
    ]),
    ["t", "y", "x"],
  );
  assertAlmostEquals(matrix[0][0], 1, 1e-12);
  assertAlmostEquals(offset[0], -4, 1e-12);
  assertAlmostEquals(offset[1], 3, 1e-12);
});

Deno.test("a byDimension that leaves an output axis unset is refused", () => {
  assertThrows(
    () =>
      ngffTransformToItkMatrix(
        createByDimension([item(createScale([2]), [0], [0])]),
        ["z", "y", "x"],
      ),
    Error,
    "leaving [1, 2]",
  );
});

Deno.test("a byDimension item of unequal arity is refused", () => {
  assertThrows(
    () =>
      ngffTransformToItkMatrix(
        createByDimension([
          item(createAffine([[1, 0, 2, 0]]), [0, 1, 2], [0]),
          item(createIdentity(), [1, 2], [1, 2]),
        ]),
        ["z", "y", "x"],
      ),
    Error,
    "only a square mapping",
  );
});

Deno.test("a byDimension axis index beyond the coordinate system is refused", () => {
  assertThrows(
    () =>
      ngffTransformToItkMatrix(
        createByDimension([item(createScale([2, 2]), [0, 5], [0, 1])]),
        ["y", "x"],
      ),
    Error,
    "exceed the 2 axes",
  );
});

Deno.test("a byDimension matches the affine it is equivalent to", () => {
  const dims = ["z", "y", "x"];
  const byDimension = ngffTransformToItkMatrix(
    createByDimension([
      item(createTranslation([7]), [0], [0]),
      item(createScale([2, 3]), [1, 2], [1, 2]),
    ]),
    dims,
  );
  const affine = ngffTransformToItkMatrix(
    createAffine([[1, 0, 0, 7], [0, 2, 0, 0], [0, 0, 3, 0]]),
    dims,
  );
  for (let row = 0; row < 3; row++) {
    assertAlmostEquals(byDimension.offset[row], affine.offset[row], 1e-12);
    for (let col = 0; col < 3; col++) {
      assertAlmostEquals(
        byDimension.matrix[row][col],
        affine.matrix[row][col],
        1e-12,
      );
    }
  }
});

Deno.test("an unsupported type names the ones that convert", () => {
  assertThrows(
    () =>
      ngffTransformToItkMatrix(
        { type: "someFutureType" } as unknown as V06Transform,
        ["y", "x"],
      ),
    Error,
    "mapAxis, byDimension, bijection",
  );
});
