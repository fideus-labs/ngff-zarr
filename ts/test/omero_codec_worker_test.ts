// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Tests for omero_codec_worker utilities, specifically copySubRegion.
 */

import { assertEquals } from "@std/assert";

/**
 * Copy a sub-region from a padded C-order buffer into a compact destination.
 *
 * For each dimension, copies only the first `subShape[d]` elements along
 * that axis from the source (which has strides `srcStrides`).
 */
function copySubRegion(
  src: { readonly [i: number]: unknown; readonly length: number },
  srcStrides: number[],
  dst: { [i: number]: unknown; length: number },
  subShape: number[],
  srcOffset = 0,
  dstOffset = 0,
  dim = 0,
): void {
  if (dim === subShape.length - 1) {
    // Innermost dimension — copy contiguous run
    for (let i = 0; i < subShape[dim]; i++) {
      dst[dstOffset + i] = src[srcOffset + i];
    }
    return;
  }
  const dstStride = subShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  for (let i = 0; i < subShape[dim]; i++) {
    copySubRegion(
      src,
      srcStrides,
      dst,
      subShape,
      srcOffset + i * srcStrides[dim],
      dstOffset + i * dstStride,
      dim + 1,
    );
  }
}

/**
 * Compute C-order strides from shape.
 */
function getStrides(shape: number[]): number[] {
  const strides: number[] = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// ============================================================================
// Tests for 1D arrays
// ============================================================================

Deno.test("copySubRegion: 1D array - full copy", () => {
  const src = [1, 2, 3, 4, 5];
  const dst = new Array(5).fill(0);
  const srcStrides = getStrides([5]);
  const subShape = [5];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 2, 3, 4, 5]);
});

Deno.test("copySubRegion: 1D array - partial copy", () => {
  const src = [1, 2, 3, 4, 5];
  const dst = new Array(3).fill(0);
  const srcStrides = getStrides([5]);
  const subShape = [3];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 2, 3]);
});

Deno.test("copySubRegion: 1D array - single element", () => {
  const src = [1, 2, 3];
  const dst = new Array(1).fill(0);
  const srcStrides = getStrides([3]);
  const subShape = [1];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1]);
});

// ============================================================================
// Tests for 2D arrays
// ============================================================================

Deno.test("copySubRegion: 2D array - full copy", () => {
  // 3x4 array in C-order
  const src = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
  ];
  const dst = new Array(12).fill(0);
  const srcStrides = getStrides([3, 4]);
  const subShape = [3, 4];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, src);
});

Deno.test("copySubRegion: 2D array - extract sub-region", () => {
  // 3x4 padded array, extract 2x3 region
  const src = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
  ];
  const dst = new Array(6).fill(0);
  const srcStrides = getStrides([3, 4]);
  const subShape = [2, 3];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 2, 3, 5, 6, 7]);
});

Deno.test("copySubRegion: 2D array - extract single row", () => {
  const src = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
  ];
  const dst = new Array(3).fill(0);
  const srcStrides = getStrides([3, 3]);
  const subShape = [1, 3];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 2, 3]);
});

Deno.test("copySubRegion: 2D array - extract single column", () => {
  const src = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
  ];
  const dst = new Array(3).fill(0);
  const srcStrides = getStrides([3, 3]);
  const subShape = [3, 1];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 4, 7]);
});

Deno.test("copySubRegion: 2D array - extract single element", () => {
  const src = [1, 2, 3, 4];
  const dst = new Array(1).fill(0);
  const srcStrides = getStrides([2, 2]);
  const subShape = [1, 1];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1]);
});

// ============================================================================
// Tests for 3D arrays
// ============================================================================

Deno.test("copySubRegion: 3D array - full copy", () => {
  // 2x3x4 array
  const src = Array.from({ length: 24 }, (_, i) => i);
  const dst = new Array(24).fill(-1);
  const srcStrides = getStrides([2, 3, 4]);
  const subShape = [2, 3, 4];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, src);
});

Deno.test("copySubRegion: 3D array - extract sub-region", () => {
  // 2x3x4 padded array, extract 2x2x3 region
  // Shape: [z, y, x] = [2, 3, 4]
  const src = [
    // z=0
    0,
    1,
    2,
    3, // y=0
    4,
    5,
    6,
    7, // y=1
    8,
    9,
    10,
    11, // y=2
    // z=1
    12,
    13,
    14,
    15, // y=0
    16,
    17,
    18,
    19, // y=1
    20,
    21,
    22,
    23, // y=2
  ];
  const dst = new Array(12).fill(-1);
  const srcStrides = getStrides([2, 3, 4]);
  const subShape = [2, 2, 3]; // Extract first 2x2x3

  copySubRegion(src, srcStrides, dst, subShape);

  // Expected: z=0: [0,1,2], [4,5,6]  z=1: [12,13,14], [16,17,18]
  assertEquals(dst, [0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]);
});

Deno.test("copySubRegion: 3D array - extract single plane", () => {
  // 3x2x2 array, extract first plane (1x2x2)
  const src = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
  ];
  const dst = new Array(4).fill(-1);
  const srcStrides = getStrides([3, 2, 2]);
  const subShape = [1, 2, 2];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 2, 3, 4]);
});

// ============================================================================
// Tests for edge chunk scenarios
// ============================================================================

Deno.test("copySubRegion: padded edge chunk - 2D example", () => {
  // Simulate a padded edge chunk: chunk_shape=[3,3], actual_shape=[2,2]
  // The padded region is filled with zeros
  const src = [
    1,
    2,
    0, // y=0: valid data, then padding
    3,
    4,
    0, // y=1: valid data, then padding
    0,
    0,
    0, // y=2: all padding
  ];
  const dst = new Array(4).fill(-1);
  const srcStrides = getStrides([3, 3]);
  const subShape = [2, 2]; // Extract only valid region

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [1, 2, 3, 4]);
});

Deno.test("copySubRegion: padded edge chunk - 3D example", () => {
  // chunk_shape=[2,2,2], actual_shape=[1,2,1]
  const src = [
    10,
    0, // z=0, y=0: one valid, one pad
    20,
    0, // z=0, y=1: one valid, one pad
    0,
    0, // z=1, y=0: all padding
    0,
    0, // z=1, y=1: all padding
  ];
  const dst = new Array(2).fill(-1);
  const srcStrides = getStrides([2, 2, 2]);
  const subShape = [1, 2, 1]; // Extract only valid region

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(dst, [10, 20]);
});

// ============================================================================
// Tests with typed arrays
// ============================================================================

Deno.test("copySubRegion: Float32Array", () => {
  const src = new Float32Array([1.5, 2.5, 3.5, 4.5]);
  const dst = new Float32Array(2);
  const srcStrides = getStrides([2, 2]);
  const subShape = [1, 2];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(Array.from(dst), [1.5, 2.5]);
});

Deno.test("copySubRegion: Uint8Array", () => {
  const src = new Uint8Array([255, 128, 64, 32, 16, 8, 4, 2]);
  const dst = new Uint8Array(4);
  const srcStrides = getStrides([2, 2, 2]);
  const subShape = [1, 2, 2];

  copySubRegion(src, srcStrides, dst, subShape);

  assertEquals(Array.from(dst), [255, 128, 64, 32]);
});

// ============================================================================
// Tests for different dimension orderings
// ============================================================================

Deno.test("copySubRegion: 4D array (c, z, y, x)", () => {
  // Shape: [c=2, z=2, y=2, x=2] = 16 elements
  const src = Array.from({ length: 16 }, (_, i) => i);
  const dst = new Array(4).fill(-1);
  const srcStrides = getStrides([2, 2, 2, 2]);
  const subShape = [1, 1, 2, 2]; // Extract c=0, z=0, all y,x

  copySubRegion(src, srcStrides, dst, subShape);

  // c=0, z=0: indices 0,1,2,3
  assertEquals(dst, [0, 1, 2, 3]);
});

Deno.test("copySubRegion: 5D array", () => {
  // Shape: [t=2, c=2, z=2, y=2, x=2] = 32 elements
  const src = Array.from({ length: 32 }, (_, i) => i);
  const dst = new Array(8).fill(-1);
  const srcStrides = getStrides([2, 2, 2, 2, 2]);
  const subShape = [1, 1, 2, 2, 2]; // Extract t=0, c=0, all z,y,x

  copySubRegion(src, srcStrides, dst, subShape);

  // t=0, c=0: indices 0-7
  assertEquals(dst, [0, 1, 2, 3, 4, 5, 6, 7]);
});

// ============================================================================
// Tests with offset parameters
// ============================================================================

Deno.test("copySubRegion: with srcOffset", () => {
  const src = [0, 0, 1, 2, 3, 4]; // First 2 elements are padding
  const dst = new Array(4).fill(-1);
  const srcStrides = getStrides([2, 3]); // but treat as starting at offset 2
  const subShape = [2, 2];

  // Start copy from index 2 in source
  copySubRegion(src, srcStrides, dst, subShape, 2, 0, 0);

  assertEquals(dst, [1, 2, 4, 0]); // Note: srcStrides expects [2,3] shape so stride[0]=3
});

Deno.test("copySubRegion: with dstOffset", () => {
  const src = [1, 2, 3, 4];
  const dst = new Array(6).fill(0);
  const srcStrides = getStrides([2, 2]);
  const subShape = [1, 2];

  // Write to destination starting at offset 2
  copySubRegion(src, srcStrides, dst, subShape, 0, 2, 0);

  assertEquals(dst, [0, 0, 1, 2, 0, 0]);
});
