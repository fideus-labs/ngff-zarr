// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for itkDirectionToAnatomicalOrientation function.
 * Tests the conversion from ITK direction cosine vectors to anatomical orientations.
 */

import { assertEquals } from "@std/assert";
import {
  AnatomicalOrientationValues,
  itkDirectionToAnatomicalOrientation,
} from "../src/types/rfc4.ts";

// Test primary axis directions for 3D

Deno.test("itkDirectionToAnatomicalOrientation - positive X axis [1,0,0] → RightToLeft", () => {
  const orientation = itkDirectionToAnatomicalOrientation([1, 0, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

Deno.test("itkDirectionToAnatomicalOrientation - negative X axis [-1,0,0] → LeftToRight", () => {
  const orientation = itkDirectionToAnatomicalOrientation([-1, 0, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

Deno.test("itkDirectionToAnatomicalOrientation - positive Y axis [0,1,0] → AnteriorToPosterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 1, 0]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - negative Y axis [0,-1,0] → PosteriorToAnterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, -1, 0]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - positive Z axis [0,0,1] → InferiorToSuperior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 0, 1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - negative Z axis [0,0,-1] → SuperiorToInferior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 0, -1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.SuperiorToInferior,
  );
});

// Test oblique directions with clearly dominant component

Deno.test("itkDirectionToAnatomicalOrientation - oblique with dominant positive X [0.9, 0.3, 0.2] → RightToLeft", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.9, 0.3, 0.2]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

Deno.test("itkDirectionToAnatomicalOrientation - oblique with dominant negative X [-0.8, 0.3, 0.1] → LeftToRight", () => {
  const orientation = itkDirectionToAnatomicalOrientation([-0.8, 0.3, 0.1]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

Deno.test("itkDirectionToAnatomicalOrientation - oblique with dominant positive Y [0.2, 0.85, 0.3] → AnteriorToPosterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.2, 0.85, 0.3]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - oblique with dominant negative Y [0.1, -0.9, 0.2] → PosteriorToAnterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.1, -0.9, 0.2]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - oblique with dominant positive Z [0.2, 0.3, 0.88] → InferiorToSuperior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.2, 0.3, 0.88]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - oblique with dominant negative Z [0.1, 0.2, -0.95] → SuperiorToInferior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.1, 0.2, -0.95]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.SuperiorToInferior,
  );
});

// Test 2D cases (2-element direction vectors)

Deno.test("itkDirectionToAnatomicalOrientation - 2D positive X [1,0] → RightToLeft", () => {
  const orientation = itkDirectionToAnatomicalOrientation([1, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

Deno.test("itkDirectionToAnatomicalOrientation - 2D negative X [-1,0] → LeftToRight", () => {
  const orientation = itkDirectionToAnatomicalOrientation([-1, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

Deno.test("itkDirectionToAnatomicalOrientation - 2D positive Y [0,1] → AnteriorToPosterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - 2D negative Y [0,-1] → PosteriorToAnterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, -1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
});

Deno.test("itkDirectionToAnatomicalOrientation - 2D oblique with dominant X [0.8, 0.4] → RightToLeft", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.8, 0.4]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

Deno.test("itkDirectionToAnatomicalOrientation - 2D oblique with dominant Y [0.3, 0.9] → AnteriorToPosterior", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.3, 0.9]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

// Test edge cases with equal or near-equal components

Deno.test("itkDirectionToAnatomicalOrientation - equal X/Y favors first component [0.707, 0.707, 0] → RightToLeft", () => {
  // When components are equal, the first one wins
  const orientation = itkDirectionToAnatomicalOrientation([0.707, 0.707, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

Deno.test("itkDirectionToAnatomicalOrientation - equal X/Y with negative X [-0.707, 0.707, 0] → LeftToRight", () => {
  // First component still wins when its absolute value is equal
  const orientation = itkDirectionToAnatomicalOrientation([
    -0.707,
    0.707,
    0,
  ]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

Deno.test("itkDirectionToAnatomicalOrientation - all near-zero with tiny X bias [0.001, 0, 0] → RightToLeft", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.001, 0, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});
