// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertThrows } from "@std/assert";
import {
  isValidDimension,
  isValidUnit,
  validateColor,
  validateMetadata,
} from "../src/utils/validation.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
  createNgffImage,
  createScale,
  createTranslation,
} from "../src/utils/factory.ts";
import { Methods } from "../src/types/methods.ts";

Deno.test("validateMetadata - valid metadata", () => {
  const validMetadata = {
    axes: [
      { name: "y", type: "space" },
      { name: "x", type: "space" },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          { type: "scale", scale: [1.0, 1.0] },
          { type: "translation", translation: [0.0, 0.0] },
        ],
      },
    ],
    name: "test_image",
    version: "0.4",
  };

  const result = validateMetadata(validMetadata);
  assertEquals(result.name, "test_image");
  assertEquals(result.version, "0.4");
});

Deno.test("validateMetadata - invalid metadata", () => {
  const invalidMetadata = {
    axes: [{ name: "invalid", type: "space" }],
    datasets: [],
  };

  assertThrows(
    () => validateMetadata(invalidMetadata),
    Error,
    "Invalid metadata",
  );
});

Deno.test("validateColor - valid colors", () => {
  validateColor("FF0000");
  validateColor("00ff00");
  validateColor("123ABC");
});

Deno.test("validateColor - invalid colors", () => {
  assertThrows(() => validateColor("GG0000"), Error, "Invalid color");
  assertThrows(() => validateColor("FF00"), Error, "Invalid color");
  assertThrows(() => validateColor("#FF0000"), Error, "Invalid color");
});

Deno.test("isValidDimension", () => {
  assertEquals(isValidDimension("x"), true);
  assertEquals(isValidDimension("y"), true);
  assertEquals(isValidDimension("z"), true);
  assertEquals(isValidDimension("c"), true);
  assertEquals(isValidDimension("t"), true);
  assertEquals(isValidDimension("invalid"), false);
});

Deno.test("isValidUnit", () => {
  assertEquals(isValidUnit("micrometer"), true);
  assertEquals(isValidUnit("second"), true);
  assertEquals(isValidUnit("meter"), true);
  assertEquals(isValidUnit("invalid"), false);
});

Deno.test("createAxis", () => {
  const axis = createAxis("x", "space", "micrometer");
  assertEquals(axis.name, "x");
  assertEquals(axis.type, "space");
  assertEquals(axis.unit, "micrometer");
});

Deno.test("createScale", () => {
  const scale = createScale([1.0, 2.0, 3.0]);
  assertEquals(scale.type, "scale");
  assertEquals(scale.scale, [1.0, 2.0, 3.0]);
});

Deno.test("createTranslation", () => {
  const translation = createTranslation([0.0, 1.0, 2.0]);
  assertEquals(translation.type, "translation");
  assertEquals(translation.translation, [0.0, 1.0, 2.0]);
});

Deno.test("createDataset", () => {
  const dataset = createDataset("0", [1.0, 1.0], [0.0, 0.0]);
  assertEquals(dataset.path, "0");
  assertEquals(dataset.coordinateTransformations.length, 2);
  assertEquals(dataset.coordinateTransformations[0].type, "scale");
  assertEquals(dataset.coordinateTransformations[1].type, "translation");
});

Deno.test("createMetadata", () => {
  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("0", [1.0, 1.0], [0.0, 0.0])];

  const metadata = createMetadata(axes, datasets, "test", "0.4");
  assertEquals(metadata.name, "test");
  assertEquals(metadata.version, "0.4");
  assertEquals(metadata.axes.length, 2);
  assertEquals(metadata.datasets.length, 1);
});

Deno.test("createNgffImage", async () => {
  const image = await createNgffImage(
    new ArrayBuffer(256 * 256),
    [256, 256],
    "uint8",
    ["y", "x"],
    { y: 1.0, x: 1.0 },
    { y: 0.0, x: 0.0 },
    "test_image",
  );

  assertEquals(image.name, "test_image");
  assertEquals(image.dims, ["y", "x"]);
  assertEquals(image.data.shape, [256, 256]);
  assertEquals(image.data.dtype, "uint8");
});

Deno.test("createMultiscales", async () => {
  const image = await createNgffImage(
    new ArrayBuffer(256 * 256),
    [256, 256],
    "uint8",
    ["y", "x"],
    { y: 1.0, x: 1.0 },
    { y: 0.0, x: 0.0 },
  );

  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("0", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets);

  const multiscales = createMultiscales(
    [image],
    metadata,
    [2, 4],
    Methods.ITKWASM_GAUSSIAN,
  );

  assertEquals(multiscales.images.length, 1);
  assertEquals(multiscales.scaleFactors, [2, 4]);
  assertEquals(multiscales.method, Methods.ITKWASM_GAUSSIAN);
});
