// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * ngffImageToItkImage lays the pixels out in ITK's order (spatial axes in
 * reverse ITK order, components last) whatever the image's axis order.
 */
import { test } from "vite-plus/test";
import { assertEquals } from "./_assert.ts";
import * as zarr from "zarrita";

import { ngffImageToItkImage } from "../src/io/ngff_image_to_itk_image.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import type { AnatomicalOrientation } from "../src/types/rfc4.ts";

async function countingImage(
  dims: string[],
  shape: number[],
  axesOrientations?: Record<string, AnatomicalOrientation>,
) {
  const array = await zarr.create(zarr.root(new Map()).resolve("a"), {
    shape,
    chunkShape: shape,
    dtype: "uint8",
    fillValue: 0,
  });
  const data = new Uint8Array(shape.reduce((a, b) => a * b, 1)).map(
    (_, i) => i,
  );
  const stride = shape.map((_, i) =>
    shape.slice(i + 1).reduce((a, b) => a * b, 1),
  );
  await zarr.set(array, null, { data, shape, stride });
  return new NgffImage({
    data: array,
    dims,
    scale: Object.fromEntries(dims.map((d) => [d, 1])),
    translation: Object.fromEntries(dims.map((d) => [d, 0])),
    name: "image",
    axesUnits: undefined,
    axesOrientations,
    computedCallbacks: undefined,
  });
}

test("a leading channel axis is moved to the components", async () => {
  const itk = await ngffImageToItkImage(
    await countingImage(["c", "y", "x"], [2, 3, 4]),
  );
  assertEquals(itk.size, [4, 3]);
  assertEquals(itk.imageType.components, 2);
  const expected: number[] = [];
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < 4; x++) {
      for (let c = 0; c < 2; c++) expected.push(c * 12 + y * 4 + x);
    }
  }
  assertEquals(Array.from(itk.data as Uint8Array), expected);
});

test("x before y is transposed for ITK", async () => {
  const itk = await ngffImageToItkImage(
    await countingImage(["x", "y"], [3, 4]),
  );
  assertEquals(itk.size, [3, 4]);
  const expected: number[] = [];
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 3; x++) expected.push(x * 4 + y);
  }
  assertEquals(Array.from(itk.data as Uint8Array), expected);
});

test("a 2D out-of-plane orientation keeps an identity direction", async () => {
  const itk = await ngffImageToItkImage(
    await countingImage(["y", "x"], [4, 5], {
      x: { type: "anatomical", value: "left-to-right" },
      y: { type: "anatomical", value: "superior-to-inferior" },
    } as Record<string, AnatomicalOrientation>),
  );
  assertEquals(Array.from(itk.direction as Float64Array), [1, 0, 0, 1]);
});
