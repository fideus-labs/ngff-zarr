// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Converting ITK transforms back to RFC-5 transformations.
 *
 * Mirrors `py/test/test_itk_transform_to_ngff_transform.py`. The strongest
 * check available here is the round trip: converting an RFC-5 transformation
 * to ITK and back must preserve the mapping exactly. Anything wrong with the
 * axis reversal or the center-of-rotation algebra breaks it.
 */

import { assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import type { Transform } from "itk-wasm";
import {
  createAffine,
  createIdentity,
  createScale,
  createTransformSequence,
  createTranslation,
  itkTransformToNgffMatrix,
  itkTransformToNgffTransform,
  ngffTransformToItkMatrix,
  ngffTransformToItkTransform,
  type V06Transform,
} from "../src/mod.ts";

const ROUND_TRIP_CASES: Array<[string, V06Transform, string[]]> = [
  ["identity", createIdentity(), ["y", "x"]],
  ["translation", createTranslation([5, 10]), ["y", "x"]],
  ["scale", createScale([2, 3, 4]), ["z", "y", "x"]],
  ["affine 2d", createAffine([[0.8, -0.6, 10], [0.6, 0.8, -4]]), ["y", "x"]],
  [
    "sheared affine 3d",
    createAffine([
      [1, 0.2, 0, 4],
      [0, 2, 0.3, -6],
      [0.5, 0, 1, 11],
    ]),
    ["z", "y", "x"],
  ],
  [
    "sequence",
    createTransformSequence([
      createTranslation([0, 10]),
      createScale([1, 2]),
    ]),
    ["y", "x"],
  ],
  [
    "affine with non-spatial axes",
    createAffine([
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 0.9, 0.1, 3],
      [0, 0, 0.2, 1.1, -2],
    ]),
    ["t", "c", "y", "x"],
  ],
];

for (const [name, transform, dims] of ROUND_TRIP_CASES) {
  Deno.test(`round trip preserves the mapping: ${name}`, () => {
    const expected = ngffTransformToItkMatrix(transform, dims);

    const converted = itkTransformToNgffTransform(
      ngffTransformToItkTransform(transform, dims),
      dims,
    );

    const actual = ngffTransformToItkMatrix(converted, dims);
    actual.matrix.forEach((row, i) =>
      row.forEach((value, j) =>
        assertAlmostEquals(value, expected.matrix[i][j], 1e-9)
      )
    );
    actual.offset.forEach((value, i) =>
      assertAlmostEquals(value, expected.offset[i], 1e-9)
    );
  });
}

const SIMPLIFY_CASES: Array<[V06Transform, string]> = [
  [createIdentity(), "identity"],
  [createTranslation([5, 10]), "translation"],
  [createScale([2, 3]), "scale"],
  [
    createTransformSequence([createScale([2, 3]), createTranslation([1, 2])]),
    "sequence",
  ],
  [createAffine([[0.8, -0.6, 0], [0.6, 0.8, 0]]), "affine"],
];

for (const [transform, expectedType] of SIMPLIFY_CASES) {
  Deno.test(`simplify returns the least expressive form: ${expectedType}`, () => {
    const dims = ["y", "x"];
    const converted = itkTransformToNgffTransform(
      ngffTransformToItkTransform(transform, dims),
      dims,
    );
    assertEquals(converted.type, expectedType);
  });
}

Deno.test("simplify can be disabled", () => {
  const dims = ["y", "x"];
  const converted = itkTransformToNgffTransform(
    ngffTransformToItkTransform(createIdentity(), dims),
    dims,
    false,
  );
  assertEquals(converted.type, "affine");
  if (converted.type !== "affine") throw new Error("expected an affine");
  assertEquals(converted.affine, [[1, 0, 0], [0, 1, 0]]);
});

/** Build an ITK-Wasm transform entry by hand. */
function entry(
  parameterization: string,
  parameters: number[],
  fixed: number[] = [],
): Transform {
  return {
    transformType: {
      transformParameterization: parameterization,
      parametersValueType: "float64",
      inputDimension: 2,
      outputDimension: 2,
    },
    name: parameterization,
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: fixed.length,
    numberOfParameters: parameters.length,
    fixedParameters: new Float64Array(fixed),
    parameters: new Float64Array(parameters),
    metadata: new Map(),
  } as unknown as Transform;
}

Deno.test("a transform list composes its last entry first", () => {
  // ITK order (x, y): shift x by 10, and scale by 2.
  const shift = entry("Translation", [10, 0]);
  const scaling = entry("Affine", [2, 0, 0, 2, 0, 0], [0, 0]);

  // [shift, scaling] means shift(scaling(p)) = 2p + 10.
  const forward = itkTransformToNgffMatrix([shift, scaling], ["y", "x"]);
  assertEquals(forward.matrix, [[2, 0], [0, 2]]);
  assertEquals(forward.offset, [0, 10]); // reversed to (y, x)

  // The other order means scaling(shift(p)) = 2(p + 10) = 2p + 20.
  const reversed = itkTransformToNgffMatrix([scaling, shift], ["y", "x"]);
  assertEquals(reversed.offset, [0, 20]);
});

Deno.test("the ITK center of rotation is folded into the offset", () => {
  // An RFC-5 affine has no center, so ITK's must be absorbed:
  // b = t + c - A c.
  const matrix = [[0.8, -0.6], [0.6, 0.8]];
  const center = [9.5, 4.5];
  const translation = [10, 10];
  const affine = entry(
    "Affine",
    [...matrix[0], ...matrix[1], ...translation],
    center,
  );

  const { offset } = itkTransformToNgffMatrix(affine, ["y", "x"]);

  const expectedItk = translation.map((value, row) =>
    value + center[row] -
    (matrix[row][0] * center[0] + matrix[row][1] * center[1])
  );
  // ITK order (x, y) reverses to NGFF order (y, x).
  assertAlmostEquals(offset[0], expectedItk[1], 1e-9);
  assertAlmostEquals(offset[1], expectedItk[0], 1e-9);
});

Deno.test("an angle-based parameterization is refused with guidance", () => {
  const euler = entry("Euler2D", [0.3, 5, -2]);
  assertThrows(
    () => itkTransformToNgffMatrix(euler, ["y", "x"]),
    Error,
    "Convert it to an Affine transform first",
  );
});

Deno.test("a scale transform decodes", () => {
  const scaling = entry("Scale", [2, 3]);
  const { matrix, offset } = itkTransformToNgffMatrix(scaling, ["y", "x"]);
  // Reversed from ITK (x, y) to NGFF (y, x).
  assertEquals(matrix, [[3, 0], [0, 2]]);
  assertEquals(offset, [0, 0]);
});

Deno.test("an empty transform list is rejected", () => {
  assertThrows(
    () => itkTransformToNgffMatrix([], ["y", "x"]),
    Error,
    "transform list is empty",
  );
});

Deno.test("a coordinate system without spatial axes is rejected", () => {
  assertThrows(
    () => itkTransformToNgffMatrix(entry("Translation", [1, 2]), ["t", "c"]),
    Error,
    "no spatial axes",
  );
});
