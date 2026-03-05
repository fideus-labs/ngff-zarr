// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Tests for itkImageToNgffImage conversion, covering scalar and
 * multi-component (vector/RGB) ITK-Wasm images.
 */

import { assertEquals } from "@std/assert";
import type { Image } from "itk-wasm";
import * as zarr from "zarrita";

import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";

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
 * Build a mock ITK-Wasm Image.
 *
 * `size` is the spatial-only size in physical order [x, y] or [x, y, z],
 * exactly as ITK-Wasm reports it.  `components` defaults to 1 (scalar).
 * The data buffer is sized `product(size) * components`.
 */
function mockItkImage(
  size: number[],
  componentType:
    | "uint8"
    | "int8"
    | "uint16"
    | "int16"
    | "uint32"
    | "int32"
    | "float32"
    | "float64",
  components = 1,
): Image {
  const totalElements = size.reduce((a, b) => a * b, 1) * components;

  const ArrayCtor: {
    new (
      n: number,
    ):
      | Uint8Array
      | Int8Array
      | Uint16Array
      | Int16Array
      | Uint32Array
      | Int32Array
      | Float32Array
      | Float64Array;
  } = {
    uint8: Uint8Array,
    int8: Int8Array,
    uint16: Uint16Array,
    int16: Int16Array,
    uint32: Uint32Array,
    int32: Int32Array,
    float32: Float32Array,
    float64: Float64Array,
  }[componentType];

  const data = new ArrayCtor(totalElements);
  for (let i = 0; i < totalElements; i++) {
    data[i] = i % 251; // prime modulus for a non-trivial pattern
  }

  return {
    imageType: {
      dimension: size.length,
      componentType,
      pixelType: components > 1 ? "VariableLengthVector" : "Scalar",
      components,
    },
    name: "test",
    origin: size.map(() => 0),
    spacing: size.map(() => 1),
    direction: identityMatrix(size.length),
    size,
    data,
    metadata: new Map(),
  };
}

// ---------------------------------------------------------------------------
// Scalar (single-component) images
// ---------------------------------------------------------------------------

Deno.test("2D scalar image: dims and shape", async () => {
  // ITK-Wasm size is in physical order [x, y]
  const image = mockItkImage([100, 80], "uint8");
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["y", "x"]);
  // shape is size reversed → [y, x] = [80, 100]
  assertEquals(ngff.data.shape, [80, 100]);
});

Deno.test("3D scalar image: dims and shape", async () => {
  const image = mockItkImage([30, 40, 20], "uint16");
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["z", "y", "x"]);
  assertEquals(ngff.data.shape, [20, 40, 30]);
});

// ---------------------------------------------------------------------------
// Multi-component (vector / RGB) images
// ---------------------------------------------------------------------------

Deno.test("2D RGB image: dims, shape, and data length", async () => {
  // 2D spatial [x=100, y=80] with 3 components (RGB)
  const image = mockItkImage([100, 80], "uint8", 3);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["y", "x", "c"]);
  assertEquals(ngff.data.shape, [80, 100, 3]);

  // Verify zarr array contains the full RGB data
  const result = await zarr.get(ngff.data);
  assertEquals(result.data.length, 80 * 100 * 3);
});

Deno.test("2D RGBA image: dims, shape, and data length", async () => {
  const image = mockItkImage([64, 48], "uint8", 4);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["y", "x", "c"]);
  assertEquals(ngff.data.shape, [48, 64, 4]);

  const result = await zarr.get(ngff.data);
  assertEquals(result.data.length, 48 * 64 * 4);
});

Deno.test("3D vector image: dims, shape, and data length", async () => {
  // 3D spatial [x=16, y=20, z=10] with 3 components
  const image = mockItkImage([16, 20, 10], "float32", 3);
  const ngff = await itkImageToNgffImage(image);

  assertEquals(ngff.dims, ["z", "y", "x", "c"]);
  assertEquals(ngff.data.shape, [10, 20, 16, 3]);

  const result = await zarr.get(ngff.data);
  assertEquals(result.data.length, 10 * 20 * 16 * 3);
});

// ---------------------------------------------------------------------------
// Spatial metadata
// ---------------------------------------------------------------------------

Deno.test("2D RGB image: scale and translation are spatial-only", async () => {
  const image = mockItkImage([100, 80], "uint8", 3);
  const ngff = await itkImageToNgffImage(image);

  // Scale and translation should only have spatial keys
  assertEquals(Object.keys(ngff.scale).sort(), ["x", "y"]);
  assertEquals(Object.keys(ngff.translation).sort(), ["x", "y"]);
});

Deno.test("2D RGB image: data round-trips correctly", async () => {
  const image = mockItkImage([4, 3], "uint8", 3);
  const ngff = await itkImageToNgffImage(image);

  const result = await zarr.get(ngff.data);
  const actual = Array.from(result.data as Uint8Array);
  const expected = Array.from(image.data as Uint8Array);
  assertEquals(actual, expected);
});
