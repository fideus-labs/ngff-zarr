#!/usr/bin/env -S deno test --allow-read --allow-write
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Normalization of channel-last input to the OME-Zarr axis order.
 *
 * Mirrors the Python port's `_canonical_axis_order` coverage: the pipeline
 * reorders axes to `(t, c, z, y, x)`, moves the data with them, and leaves an
 * axis model outside that vocabulary alone.
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import * as zarr from "zarrita";

import {
  toMultiscales,
  toNgffImage,
} from "../src/process/to_multiscales-node.ts";
import { Methods } from "../src/types/methods.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { canonicalAxisOrder } from "../src/utils/axis_order.ts";
import { zarrGet } from "../src/utils/worker_pool.ts";
import { calculateStride } from "../src/utils/transpose.ts";

/** A ramp image over `dims`/`shape`, values `0..n-1` in row-major order. */
async function rampImage(
  dims: string[],
  shape: number[],
): Promise<NgffImage> {
  const total = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    data[i] = i;
  }
  return await toNgffImage(data, { dims, shape });
}

/** Every element of `array`, read back in row-major order. */
async function readAll(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
): Promise<number[]> {
  const result = await zarrGet(array);
  return Array.from(result.data as ArrayLike<number>);
}

Deno.test("canonicalAxisOrder - channel-last input is reordered", async () => {
  const image = await rampImage(["z", "y", "x", "c"], [2, 3, 4, 2]);
  const normalized = await canonicalAxisOrder(image);

  assertEquals(normalized.dims, ["c", "z", "y", "x"]);
  assertEquals([...normalized.data.shape], [2, 2, 3, 4]);
});

Deno.test("canonicalAxisOrder - the data follows the axes", async () => {
  // A relabelling that left the buffer alone would keep the ramp intact, so
  // compare against the transpose computed independently.
  const dims = ["y", "x", "c"];
  const shape = [2, 3, 4];
  const image = await rampImage(dims, shape);
  const normalized = await canonicalAxisOrder(image);

  assertEquals(normalized.dims, ["c", "y", "x"]);
  const permutation = [2, 0, 1];
  const newShape = permutation.map((i) => shape[i]);
  const sourceStride = calculateStride(shape);
  const expected = new Array(shape.reduce((a, b) => a * b, 1));
  for (let c = 0; c < newShape[0]; c++) {
    for (let y = 0; y < newShape[1]; y++) {
      for (let x = 0; x < newShape[2]; x++) {
        const target = (c * newShape[1] + y) * newShape[2] + x;
        expected[target] = y * sourceStride[0] + x * sourceStride[1] +
          c * sourceStride[2];
      }
    }
  }
  assertEquals(await readAll(normalized.data), expected);
});

Deno.test("canonicalAxisOrder - already ordered input is untouched", async () => {
  const image = await rampImage(["c", "y", "x"], [2, 3, 4]);
  assertEquals(await canonicalAxisOrder(image), image);
});

Deno.test("canonicalAxisOrder - a non-canonical vocabulary is left alone", async () => {
  // RFC-3 axis names carry no spec ordering to normalize to.
  const image = await rampImage(["b", "a"], [2, 3]);
  const normalized = await canonicalAxisOrder(image);

  assertEquals(normalized, image);
  assertEquals(normalized.dims, ["b", "a"]);
});

Deno.test("toMultiscales - generated axes are spec-ordered", async () => {
  const image = await rampImage(["z", "y", "x", "c"], [8, 16, 16, 2]);
  const multiscales = await toMultiscales(image, {
    scaleFactors: [],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  assertEquals(
    multiscales.metadata.axes.map((axis) => axis.name),
    ["c", "z", "y", "x"],
  );
  assertEquals(multiscales.images[0].dims, ["c", "z", "y", "x"]);
  assertNotEquals(multiscales.images[0].dims, image.dims);
});
