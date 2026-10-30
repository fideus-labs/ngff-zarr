// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/** A channel axis changes neither the downsampled values nor the chunking. */
import { test } from "vite-plus/test";
import { assertEquals } from "./_assert.ts";
import * as zarr from "zarrita";

import {
  toMultiscales,
  toNgffImage,
} from "../src/process/to_multiscales-node.ts";
import { Methods } from "../src/types/methods.ts";

function pixels(size: number): Uint8Array {
  const data = new Uint8Array(size * size);
  for (let i = 0; i < data.length; i++) data[i] = (i * 7919) % 251;
  return data;
}

test("a size-1 channel axis does not change the pyramid", async () => {
  const size = 64;
  const plain = await toNgffImage(pixels(size), {
    dims: ["y", "x"],
    shape: [size, size],
  });
  const withChannel = await toNgffImage(pixels(size), {
    dims: ["c", "y", "x"],
    shape: [1, size, size],
  });
  const options = { scaleFactors: [2, 4], method: Methods.ITKWASM_GAUSSIAN };
  const a = await toMultiscales(plain, options);
  const b = await toMultiscales(withChannel, options);

  for (const level of [1, 2]) {
    assertEquals(b.images[level].data.shape[0], 1, "the channel axis stays");
    const left = await zarr.get(a.images[level].data);
    const right = await zarr.get(b.images[level].data);
    assertEquals(
      Array.from(right.data as Uint8Array),
      Array.from(left.data as Uint8Array),
    );
  }
});

test("chunks keyed by name land on their axes with a channel axis", async () => {
  const size = 64;
  const image = await toNgffImage(new Uint8Array(3 * size * size), {
    dims: ["c", "y", "x"],
    shape: [3, size, size],
  });
  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
    chunks: { c: 1, y: 16, x: 16 },
  });
  assertEquals(multiscales.images[1].data.chunks.slice(1), [16, 16]);
});

test("label images with a few channels downsample per channel", async () => {
  const size = 16;
  const image = await toNgffImage(new Uint8Array(3 * size * size).fill(2), {
    dims: ["c", "y", "x"],
    shape: [3, size, size],
  });
  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_LABEL_IMAGE,
  });
  assertEquals(multiscales.images[1].data.shape, [3, 8, 8]);
});
