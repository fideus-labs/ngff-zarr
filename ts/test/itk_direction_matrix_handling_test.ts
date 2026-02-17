// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Tests for direction matrix handling in itkImageToNgffImage.
 * Verifies anatomical orientation extraction from ITK direction matrices.
 */

import { assertEquals } from "@std/assert";
import type { Image } from "itk-wasm";

import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import { AnatomicalOrientationValues } from "../src/types/rfc4.ts";

/**
 * Create identity direction matrix for ITK-Wasm Image
 */
function identityMatrix(dimension: number): Float64Array {
  const m = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension; i++) {
    m[i * dimension + i] = 1;
  }
  return m;
}

/**
 * Build a mock ITK-Wasm Image with custom direction matrix.
 */
function mockItkImageWithDirection(
  size: number[],
  direction: Float64Array,
): Image {
  const totalElements = size.reduce((a, b) => a * b, 1);
  const data = new Uint8Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    data[i] = i % 251;
  }

  return {
    imageType: {
      dimension: size.length,
      componentType: "uint8",
      pixelType: "Scalar",
      components: 1,
    },
    name: "test",
    origin: size.map(() => 0),
    spacing: size.map(() => 1),
    direction,
    size,
    data,
    metadata: new Map(),
  };
}

// Test identity direction matrices (should fall back to simple LPS lookup)

Deno.test("3D identity direction → standard LPS orientations", async () => {
  const direction = identityMatrix(3);
  const image = mockItkImageWithDirection([10, 20, 30], direction);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["z", "y", "x"]);
  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
  assertEquals(
    ngff.axesOrientations?.["z"]?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("2D identity direction → standard LPS orientations", async () => {
  const direction = identityMatrix(2);
  const image = mockItkImageWithDirection([10, 20], direction);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["y", "x"]);
  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

// Test permuted axes (common in medical imaging)

Deno.test("3D permuted axes: X→Y, Y→Z, Z→X", async () => {
  // Direction matrix:
  // [0  0  1]   (X maps to Z: inferior-superior)
  // [1  0  0]   (Y maps to X: right-left)
  // [0  1  0]   (Z maps to Y: anterior-posterior)
  const direction = new Float64Array([
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
  ]);
  const image = mockItkImageWithDirection([10, 20, 30], direction);
  const ngff = await itkImageToNgffImage(image);

  // dims are still ["z", "y", "x"] in array order
  assertEquals(ngff.dims, ["z", "y", "x"]);
  // But orientations should reflect the permutation:
  // x (ITK axis 0) → column 0 → [0, 1, 0] → Y component → A/P
  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
  // y (ITK axis 1) → column 1 → [0, 0, 1] → Z component → I/S
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
  // z (ITK axis 2) → column 2 → [1, 0, 0] → X component → R/L
  assertEquals(
    ngff.axesOrientations?.["z"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
});

Deno.test("3D permuted with negation: flipped Y axis", async () => {
  // Direction matrix with Y axis flipped:
  // [1   0   0]
  // [0  -1   0]
  // [0   0   1]
  const direction = new Float64Array([
    1,
    0,
    0,
    0,
    -1,
    0,
    0,
    0,
    1,
  ]);
  const image = mockItkImageWithDirection([10, 20, 30], direction);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  // Negative Y component → Posterior to Anterior
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
  assertEquals(
    ngff.axesOrientations?.["z"]?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("2D permuted axes: X↔Y swap", async () => {
  // Direction matrix:
  // [0  1]   (X maps to Y: A/P)
  // [1  0]   (Y maps to X: R/L)
  const direction = new Float64Array([
    0,
    1,
    1,
    0,
  ]);
  const image = mockItkImageWithDirection([10, 20], direction);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["y", "x"]);
  // x (ITK axis 0) → column 0 → [0, 1] → Y component → A/P
  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
  // y (ITK axis 1) → column 1 → [1, 0] → X component → R/L
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
});

// Test oblique/rotated direction matrices

Deno.test("3D oblique: 45-degree rotation in XY plane", async () => {
  // Rotation around Z axis by 45 degrees:
  // [0.707  -0.707  0]
  // [0.707   0.707  0]
  // [0       0      1]
  const cos45 = 0.707;
  const direction = new Float64Array([
    cos45,
    -cos45,
    0,
    cos45,
    cos45,
    0,
    0,
    0,
    1,
  ]);
  const image = mockItkImageWithDirection([10, 20, 30], direction);
  const ngff = await itkImageToNgffImage(image);

  // x (ITK axis 0) → column 0 → [cos45, cos45, 0]
  // Dominant: equal X and Y, so first (X) wins → R/L
  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  // y (ITK axis 1) → column 1 → [-cos45, cos45, 0]
  // Dominant: Y component (0.707 > |-0.707|) → A/P
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
  // z (ITK axis 2) → column 2 → [0, 0, 1]
  // Clearly Z component → I/S
  assertEquals(
    ngff.axesOrientations?.["z"]?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("3D oblique with dominant components", async () => {
  // Direction where each column has a clearly dominant component
  // Column 0: [0.9, 0.3, 0.2] → dominant X
  // Column 1: [0.2, 0.85, 0.3] → dominant Y
  // Column 2: [0.1, 0.2, 0.9] → dominant Z
  const direction = new Float64Array([
    0.9,
    0.2,
    0.1,
    0.3,
    0.85,
    0.2,
    0.2,
    0.3,
    0.9,
  ]);
  const image = mockItkImageWithDirection([10, 20, 30], direction);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
  assertEquals(
    ngff.axesOrientations?.["z"]?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("2D oblique rotation", async () => {
  // 30-degree rotation:
  // [0.866  -0.5]
  // [0.5    0.866]
  const cos30 = 0.866;
  const sin30 = 0.5;
  const direction = new Float64Array([
    cos30,
    -sin30,
    sin30,
    cos30,
  ]);
  const image = mockItkImageWithDirection([10, 20], direction);
  const ngff = await itkImageToNgffImage(image);

  // x (ITK axis 0) → column 0 → [cos30, sin30]
  // Dominant: X component (0.866 > 0.5) → R/L
  assertEquals(
    ngff.axesOrientations?.["x"]?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  // y (ITK axis 1) → column 1 → [-sin30, cos30]
  // Dominant: Y component (0.866 > |-0.5|) → A/P
  assertEquals(
    ngff.axesOrientations?.["y"]?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

// Test edge case: axes aligned at 45 degrees

Deno.test(
  "2D 45-degree rotation: first component wins on equal magnitude",
  async () => {
    // Both axes at 45 degrees:
    // [0.707  -0.707]
    // [0.707   0.707]
    const direction = new Float64Array([
      0.707,
      -0.707,
      0.707,
      0.707,
    ]);
    const image = mockItkImageWithDirection([10, 20], direction);
    const ngff = await itkImageToNgffImage(image);

    // When components are equal, first one wins
    // x → [0.707, 0.707] → X wins → R/L
    assertEquals(
      ngff.axesOrientations?.["x"]?.value,
      AnatomicalOrientationValues.RightToLeft,
    );
    // y → [-0.707, 0.707] → Y wins (0.707 > |-0.707|) → A/P
    assertEquals(
      ngff.axesOrientations?.["y"]?.value,
      AnatomicalOrientationValues.AnteriorToPosterior,
    );
  },
);

// Test that disabling anatomical orientation works

Deno.test("anatomical orientation disabled → no axesOrientations", async () => {
  const direction = identityMatrix(3);
  const image = mockItkImageWithDirection([10, 20, 30], direction);
  const ngff = await itkImageToNgffImage(image, {
    addAnatomicalOrientation: false,
  });

  assertEquals(ngff.axesOrientations, undefined);
});
