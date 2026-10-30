// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for itkDirectionToAnatomicalOrientation function.
 * Tests conversion from ITK direction cosine vectors to anatomical
 * orientations.
 */

import { test } from "vite-plus/test";
import { assertEquals } from "./_assert.ts";
import {
  AnatomicalOrientationValues,
  itkDirectionToAnatomicalOrientation,
} from "../src/types/rfc4.ts";

// Test primary axis directions for 3D

test("itkDirectionToAnatomicalOrientation - positive X [1,0,0] → RightToLeft", () => {
  const orientation = itkDirectionToAnatomicalOrientation([1, 0, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

test("itkDirectionToAnatomicalOrientation - negative X [-1,0,0] → LeftToRight", () => {
  const orientation = itkDirectionToAnatomicalOrientation([-1, 0, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

test("itkDirectionToAnatomicalOrientation - positive Y [0,1,0] → A→P", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 1, 0]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

test("itkDirectionToAnatomicalOrientation - negative Y [0,-1,0] → P→A", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, -1, 0]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
});

test("itkDirectionToAnatomicalOrientation - positive Z [0,0,1] → I→S", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 0, 1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

test("itkDirectionToAnatomicalOrientation - negative Z [0,0,-1] → S→I", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 0, -1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.SuperiorToInferior,
  );
});

// Test oblique directions with clearly dominant component

test("itkDirectionToAnatomicalOrientation - oblique dominant +X [0.9, 0.3, 0.2]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.9, 0.3, 0.2]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

test("itkDirectionToAnatomicalOrientation - oblique dominant -X [-0.8, 0.3, 0.1]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([-0.8, 0.3, 0.1]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

test("itkDirectionToAnatomicalOrientation - oblique dominant +Y [0.2, 0.85, 0.3]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.2, 0.85, 0.3]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

test("itkDirectionToAnatomicalOrientation - oblique dominant -Y [0.1, -0.9, 0.2]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.1, -0.9, 0.2]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
});

test("itkDirectionToAnatomicalOrientation - oblique dominant +Z [0.2, 0.3, 0.88]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.2, 0.3, 0.88]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

test("itkDirectionToAnatomicalOrientation - oblique dominant -Z [0.1, 0.2, -0.95]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.1, 0.2, -0.95]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.SuperiorToInferior,
  );
});

// Test 2D cases (2-element direction vectors)

test("itkDirectionToAnatomicalOrientation - 2D positive X [1,0]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([1, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

test("itkDirectionToAnatomicalOrientation - 2D negative X [-1,0]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([-1, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

test("itkDirectionToAnatomicalOrientation - 2D positive Y [0,1]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, 1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

test("itkDirectionToAnatomicalOrientation - 2D negative Y [0,-1]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0, -1]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
});

test("itkDirectionToAnatomicalOrientation - 2D oblique dominant X [0.8, 0.4]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.8, 0.4]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

test("itkDirectionToAnatomicalOrientation - 2D oblique dominant Y [0.3, 0.9]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.3, 0.9]);
  assertEquals(
    orientation.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

// Test edge cases with equal or near-equal components

test("itkDirectionToAnatomicalOrientation - equal X/Y favors first", () => {
  // When components are equal, the first one wins
  const orientation = itkDirectionToAnatomicalOrientation([0.707, 0.707, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});

test("itkDirectionToAnatomicalOrientation - equal magnitude with -X", () => {
  // First component still wins when its absolute value is equal
  const orientation = itkDirectionToAnatomicalOrientation([-0.707, 0.707, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

test("itkDirectionToAnatomicalOrientation - near-zero with tiny X [0.001, 0, 0]", () => {
  const orientation = itkDirectionToAnatomicalOrientation([0.001, 0, 0]);
  assertEquals(orientation.value, AnatomicalOrientationValues.RightToLeft);
});
