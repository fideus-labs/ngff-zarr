// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertThrows } from "@std/assert";
import * as zarr from "zarrita";
import { NgffImage } from "../src/types/ngff_image.ts";
import { Multiscales } from "../src/types/multiscales.ts";
import { Methods } from "../src/types/methods.ts";
import { validateColor } from "../src/types/zarr_metadata.ts";

Deno.test("zarr.Array creation for NgffImage", async () => {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  const zarrArray = await zarr.create(root.resolve("test_array"), {
    shape: [256, 256],
    chunk_shape: [256, 256],
    data_type: "uint8",
  });

  assertEquals(zarrArray.shape, [256, 256]);
  assertEquals(zarrArray.dtype, "uint8");
  assertEquals(zarrArray.chunks, [256, 256]);
  assertEquals(zarrArray.path, "/test_array");
});

Deno.test("NgffImage creation", async () => {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  const zarrArray = await zarr.create(root.resolve("image"), {
    shape: [256, 256],
    chunk_shape: [64, 64],
    data_type: "uint8",
  });

  const ngffImage = new NgffImage({
    data: zarrArray,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test_image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  assertEquals(ngffImage.dims, ["y", "x"]);
  assertEquals(ngffImage.scale, { y: 1.0, x: 1.0 });
  assertEquals(ngffImage.translation, { y: 0.0, x: 0.0 });
  assertEquals(ngffImage.name, "test_image");
  assertEquals(ngffImage.axesUnits, undefined);
  assertEquals(ngffImage.computedCallbacks.length, 0);
});

Deno.test("NgffImage with axes units", async () => {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  const zarrArray = await zarr.create(root.resolve("volume"), {
    shape: [100, 100, 100],
    chunk_shape: [50, 50, 50],
    data_type: "uint16",
  });

  const ngffImage = new NgffImage({
    data: zarrArray,
    dims: ["z", "y", "x"],
    scale: { z: 2.5, y: 1.0, x: 1.0 },
    translation: { z: 0.0, y: 0.0, x: 0.0 },
    name: "image",
    axesUnits: { z: "micrometer", y: "micrometer", x: "micrometer" },
    computedCallbacks: undefined,
  });

  assertEquals(ngffImage.axesUnits, {
    z: "micrometer",
    y: "micrometer",
    x: "micrometer",
  });
});

Deno.test("Multiscales creation", async () => {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  const zarrArray = await zarr.create(root.resolve("image"), {
    shape: [256, 256],
    chunk_shape: [64, 64],
    data_type: "uint8",
  });

  const ngffImage = new NgffImage({
    data: zarrArray,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const metadata = {
    axes: [
      { name: "y" as const, type: "space" as const, unit: undefined },
      { name: "x" as const, type: "space" as const, unit: undefined },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          { type: "scale" as const, scale: [1.0, 1.0] },
          { type: "translation" as const, translation: [0.0, 0.0] },
        ],
      },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "test",
    version: "0.4",
  };

  const multiscales = new Multiscales({
    images: [ngffImage],
    metadata,
    scaleFactors: [2, 4],
    method: Methods.ITKWASM_GAUSSIAN,
    chunks: undefined,
  });

  assertEquals(multiscales.images.length, 1);
  assertEquals(multiscales.scaleFactors, [2, 4]);
  assertEquals(multiscales.method, Methods.ITKWASM_GAUSSIAN);
});

Deno.test("validateColor function", () => {
  validateColor("FF0000");
  validateColor("00FF00");
  validateColor("0000FF");
  validateColor("123ABC");

  assertThrows(() => validateColor("GG0000"), Error, "Invalid color");
  assertThrows(() => validateColor("FF00"), Error, "Invalid color");
  assertThrows(() => validateColor("FF00000"), Error, "Invalid color");
  assertThrows(() => validateColor("#FF0000"), Error, "Invalid color");
});
