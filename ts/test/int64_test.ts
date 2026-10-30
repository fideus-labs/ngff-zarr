// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/** int64/uint64 images (e.g. Python-written label images) downsample and get omero. */
import { test } from "vite-plus/test";
import { assertEquals } from "./_assert.ts";
import * as zarr from "zarrita";

import { toMultiscales } from "../src/process/to_multiscales-node.ts";
import { Methods } from "../src/types/methods.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { computeOmeroFromNgffImage } from "../src/utils/compute_omero.ts";

async function int64Image(): Promise<NgffImage> {
  const shape = [16, 16];
  const array = await zarr.create(zarr.root(new Map()).resolve("labels"), {
    shape,
    chunkShape: shape,
    dtype: "int64",
  });
  const data = new BigInt64Array(256).map((_, i) => BigInt(i % 7));
  await zarr.set(array, null, { data, shape, stride: [16, 1] });
  return new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1, x: 1 },
    translation: { y: 0, x: 0 },
    name: "labels",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

for (const method of [
  Methods.ITKWASM_LABEL_IMAGE,
  Methods.ITKWASM_BIN_SHRINK,
  Methods.ITKWASM_GAUSSIAN,
]) {
  test(`int64 images downsample with ${method}`, async () => {
    const multiscales = await toMultiscales(await int64Image(), {
      scaleFactors: [2],
      method,
    });
    assertEquals(multiscales.images[1].data.shape, [8, 8]);
    assertEquals(multiscales.images[1].data.dtype, "int64");
  });
}

test("int64 images get omero channel statistics", async () => {
  const omero = await computeOmeroFromNgffImage(await int64Image());
  assertEquals(omero.channels.length, 1);
  assertEquals(omero.channels[0].window.max, 6);
});
