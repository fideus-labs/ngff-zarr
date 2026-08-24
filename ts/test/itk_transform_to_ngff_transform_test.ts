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
  ngffTransformToItkTransform,
  type V06Transform,
} from "../src/mod.ts";
import { ngffTransformToItkMatrix } from "../src/utils/ngff_transform_to_itk_transform.ts";

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

Deno.test("a mirror does not simplify to a scale", () => {
  // RFC-5 requires strictly positive scale factors. A flip is diagonal, so a
  // sign-blind simplifier calls it a `scale` and writes metadata the 0.6
  // schema rejects. An LPS to RAS flip is an ordinary registration result.
  const flip = entry("Scale", [-1, 1]);
  const converted = itkTransformToNgffTransform(flip, ["y", "x"]);
  assertEquals(converted.type, "affine");

  // A mirror with a translation must not become a sequence either.
  const withShift = entry("Affine", [-2, 0, 0, 3, 5, 7], [0, 0]);
  assertEquals(
    itkTransformToNgffTransform(withShift, ["y", "x"]).type,
    "affine",
  );

  // ... while a strictly positive diagonal still simplifies.
  assertEquals(
    itkTransformToNgffTransform(entry("Scale", [2, 3]), ["y", "x"]).type,
    "scale",
  );
});

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
  dimension = 2,
): Transform {
  return {
    transformType: {
      transformParameterization: parameterization,
      parametersValueType: "float64",
      inputDimension: dimension,
      outputDimension: dimension,
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

Deno.test("mismatched ITK-Wasm dimensionality is rejected", () => {
  // Without the parameter-count check, a 3D affine decoded against 2D dims
  // silently projects, and a 2D affine against 3D dims fills the result with
  // NaN, which JSON.stringify writes into the store as null.
  const cases: Array<[string, number[], number[], string[]]> = [
    ["Affine", [2, 0, 0, 0, 3, 0, 0, 0, 4, 10, 20, 30], [0, 0, 0], ["y", "x"]],
    ["Translation", [10, 20, 30], [], ["y", "x"]],
    ["Affine", [2, 0, 0, 3, 10, 20], [0, 0], ["z", "y", "x"]],
    ["Scale", [2, 3], [], ["z", "y", "x"]],
    ["Translation", [5], [], ["y", "x"]],
    ["Affine", [], [], ["y", "x"]],
  ];
  for (const [parameterization, parameters, fixed, dims] of cases) {
    assertThrows(
      () =>
        itkTransformToNgffMatrix(
          entry(parameterization, parameters, fixed),
          dims,
        ),
      Error,
      "does not match",
    );
  }
});

Deno.test("a composite entry is refused at any position", () => {
  // A parameterless 'Composite' entry is ambiguous: the pipeline writes one
  // as a grouping header, but itk.dict_from_transform emits one only for a
  // NESTED composite whose children it dropped. Skipping a leading one
  // silently returns the mapping minus the lost children.
  for (const position of [0, 1]) {
    const entries = [
      entry("Translation", [10, 0], []),
      entry("Affine", [2, 0, 0, 2, 0, 0], [0, 0]),
    ];
    entries.splice(position, 0, entry("Composite", [], []));
    assertThrows(
      () => itkTransformToNgffMatrix(entries, ["y", "x"]),
      Error,
      "'Composite' entry",
    );
  }
});

Deno.test("non-canonical spatial order converts by name", () => {
  // ITK's first component is x by name, not whichever axis dims lists last.
  const wasm = entry("Translation", [7, 5, 1], [], 3);

  for (const dims of [["z", "y", "x"], ["z", "x", "y"], ["x", "y", "z"]]) {
    const { offset } = itkTransformToNgffMatrix(wasm, dims);
    const named = Object.fromEntries(dims.map((dim, i) => [dim, offset[i]]));
    assertEquals(named, { x: 7, y: 5, z: 1 });
  }

  // The reverse direction inverts it exactly, in every spelling.
  const byName: Record<string, number> = { x: 7, y: 5, z: 1 };
  for (const dims of [["z", "y", "x"], ["z", "x", "y"], ["x", "y", "z"]]) {
    const translation = createTranslation(dims.map((dim) => byName[dim]));
    const { offset } = ngffTransformToItkMatrix(translation, dims);
    assertEquals(offset, [7, 5, 1]);
  }
});

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

Deno.test("a deformation is refused as a deformation, not as a matrix gap", () => {
  // "Convert it to an Affine transform first" is not advice that applies to a
  // displacement field: it has no affine equivalent at all. Saying so sends
  // the caller to the RFC-5 field types instead of on a wild goose chase.
  for (const parameterization of ["DisplacementField", "BSpline"]) {
    const deformation = entry(parameterization, [0, 0, 0, 0]);
    assertThrows(
      () => itkTransformToNgffMatrix(deformation, ["y", "x"]),
      Error,
      "describes a deformation",
    );
  }
});

Deno.test("a scale transform decodes", () => {
  const scaling = entry("Scale", [2, 3]);
  const { matrix, offset } = itkTransformToNgffMatrix(scaling, ["y", "x"]);
  // Reversed from ITK (x, y) to NGFF (y, x).
  assertEquals(matrix, [[3, 0], [0, 2]]);
  assertEquals(offset, [0, 0]);
});

Deno.test("a scale transform's center is folded into the offset", () => {
  // itk.ScaleTransform scales about its center like an affine does, so its
  // fixed parameters cannot be ignored: y = S (x - c) + c, which is
  // y = S x + (c - S c). Reading the parameters alone would put the image at
  // the wrong place with no error.
  const scale = [2, 3];
  const center = [10, 20];
  const scaling = entry("Scale", scale, center);

  const { matrix, offset } = itkTransformToNgffMatrix(scaling, ["y", "x"]);

  // ITK order (x, y): (10 - 2*10, 20 - 3*20) = (-10, -40).
  assertEquals(matrix, [[3, 0], [0, 2]]);
  assertEquals(offset, [-40, -10]); // reversed to NGFF order (y, x)
});

Deno.test("a component on a non-spatial axis alone is projected away", () => {
  // ITK has no non-spatial axis, so a frame interval on t has nowhere to go.
  // Dropping it leaves the spatial mapping exact, which is all an ITK
  // transform describes. This pins that as a decision rather than an accident;
  // a component that couples t to a spatial axis is refused instead.
  const { matrix, offset } = ngffTransformToItkMatrix(
    createTransformSequence([
      createScale([0.5, 1, 2, 2]),
      createTranslation([7, 0, 3, -4]),
    ]),
    ["t", "c", "y", "x"],
  );

  assertEquals(matrix, [[2, 0], [0, 2]]); // ITK order (x, y)
  assertEquals(offset, [-4, 3]);

  assertThrows(
    () =>
      ngffTransformToItkMatrix(
        createAffine([
          [1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0.5, 0, 1, 0, 0], // y would depend on t
          [0, 0, 0, 1, 0],
        ]),
        ["t", "c", "y", "x"],
      ),
    Error,
    "couples spatial and non-spatial axes",
  );
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
