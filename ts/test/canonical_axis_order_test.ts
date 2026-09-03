#!/usr/bin/env -S deno test --allow-read --allow-write
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Normalization of a non-canonical axis order to the OME-Zarr axis order.
 *
 * Mirrors the Python port's `_canonical_axis_order` coverage: the pipeline
 * reorders axes to `(t, c, z, y, x)`, moves the data with them, and leaves an
 * axis model outside that vocabulary alone.
 */

import { assertEquals, assertNotEquals, assertRejects } from "@std/assert";
import * as zarr from "zarrita";

import {
  toMultiscales,
  toNgffImage,
} from "../src/process/to_multiscales-node.ts";
import { Methods } from "../src/types/methods.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { canonicalAxisOrder } from "../src/utils/axis_order.ts";
import { zarrGet, zarrSet } from "../src/utils/worker_pool.ts";
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

Deno.test("canonicalAxisOrder - a non-canonical order is normalized", async () => {
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

/** An in-memory image over `dims`/`shape` with an explicit dtype and chunking. */
async function chunkedImage(
  dims: string[],
  shape: number[],
  chunkShape: number[],
  dataType: "float32" | "int64",
): Promise<NgffImage> {
  const total = shape.reduce((a, b) => a * b, 1);
  const store: Map<string, Uint8Array> = new Map();
  const array = await zarr.create(zarr.root(store).resolve("/0"), {
    shape,
    chunk_shape: chunkShape,
    data_type: dataType,
    fill_value: 0,
  });
  const data = dataType === "int64"
    ? BigInt64Array.from({ length: total }, (_, i) => BigInt(i))
    : Float32Array.from({ length: total }, (_, i) => i);
  await zarrSet(array as never, shape.map(() => null), {
    data,
    shape,
    stride: calculateStride(shape),
  } as never);
  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  for (const dim of dims) {
    scale[dim] = 1.0;
    translation[dim] = 0.0;
  }
  return new NgffImage({
    data: array as zarr.Array<zarr.DataType, zarr.Readable>,
    dims,
    scale,
    translation,
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

Deno.test("canonicalAxisOrder - a multi-chunk array transposes correctly", async () => {
  // The copy walks the source chunk grid, so a single-chunk fixture would not
  // exercise the per-chunk region arithmetic.
  const shape = [4, 6, 2];
  const image = await chunkedImage(
    ["y", "x", "c"],
    shape,
    [2, 3, 1],
    "float32",
  );
  const normalized = await canonicalAxisOrder(image);

  assertEquals(normalized.dims, ["c", "y", "x"]);
  assertEquals([...normalized.data.shape], [2, 4, 6]);

  const sourceStride = calculateStride(shape);
  const expected: number[] = [];
  for (let c = 0; c < 2; c++) {
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 6; x++) {
        expected.push(
          y * sourceStride[0] + x * sourceStride[1] + c * sourceStride[2],
        );
      }
    }
  }
  assertEquals(await readAll(normalized.data), expected);
});

Deno.test("canonicalAxisOrder - a 64-bit integer image transposes", async () => {
  // int64 reads back as a BigInt64Array; treating it as float32 would allocate
  // the wrong buffer and throw on the first bigint assignment.
  const image = await chunkedImage(
    ["y", "x", "c"],
    [2, 3, 2],
    [2, 3, 2],
    "int64",
  );
  const normalized = await canonicalAxisOrder(image);

  assertEquals(normalized.dims, ["c", "y", "x"]);
  assertEquals([...normalized.data.shape], [2, 2, 3]);
  const values = (await zarrGet(normalized.data)).data;
  assertEquals(values instanceof BigInt64Array, true);
  assertEquals(
    Array.from(values as BigInt64Array).map(Number),
    [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
  );
});

Deno.test("toMultiscales - preserve keeps the order the caller chose", async () => {
  const image = await rampImage(["z", "y", "x", "c"], [8, 16, 16, 2]);
  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
    axisOrder: "preserve",
  });

  assertEquals(
    multiscales.metadata.axes.map((axis) => axis.name),
    ["z", "y", "x", "c"],
  );
  assertEquals(multiscales.images[0].dims, ["z", "y", "x", "c"]);
  // The pyramid follows the axes it was given: the channel axis is not scaled.
  assertEquals(multiscales.images[1].data.shape, [4, 8, 8, 2]);
});

Deno.test("toMultiscales - an unknown axis order is refused", async () => {
  const image = await rampImage(["z", "y", "x", "c"], [8, 16, 16, 2]);
  await assertRejects(
    () =>
      toMultiscales(image, {
        scaleFactors: [],
        // deno-lint-ignore no-explicit-any
        axisOrder: "spec" as any,
      }),
    Error,
    "axisOrder must be",
  );
});
