// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertThrows } from "./_assert.ts";
import * as zarr from "zarrita";
import {
  isValidDimension,
  isValidUnit,
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

test("validateMetadata - valid metadata", () => {
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

test("validateMetadata - axes the spec allows", () => {
  // A vector-component axis of a displacement field, and an axis without a
  // type (valid at 0.4).
  for (const c of [{ name: "c", type: "displacement" }, { name: "c" }]) {
    validateMetadata({
      axes: [c, { name: "y", type: "space" }, { name: "x", type: "space" }],
      datasets: [
        {
          path: "0",
          coordinateTransformations: [{ type: "scale", scale: [1, 1, 1] }],
        },
      ],
      name: "field",
      version: "0.4",
    });
  }
});

test("validateMetadata - invalid metadata", () => {
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

test("isValidDimension", () => {
  assertEquals(isValidDimension("x"), true);
  assertEquals(isValidDimension("y"), true);
  assertEquals(isValidDimension("z"), true);
  assertEquals(isValidDimension("c"), true);
  assertEquals(isValidDimension("t"), true);
  assertEquals(isValidDimension("invalid"), false);
});

test("isValidUnit", () => {
  assertEquals(isValidUnit("micrometer"), true);
  assertEquals(isValidUnit("second"), true);
  assertEquals(isValidUnit("meter"), true);
  assertEquals(isValidUnit("invalid"), false);
});

test("createAxis", () => {
  const axis = createAxis("x", "space", "micrometer");
  assertEquals(axis.name, "x");
  assertEquals(axis.type, "space");
  assertEquals(axis.unit, "micrometer");
});

test("createScale", () => {
  const scale = createScale([1.0, 2.0, 3.0]);
  assertEquals(scale.type, "scale");
  assertEquals(scale.scale, [1.0, 2.0, 3.0]);
});

test("createTranslation", () => {
  const translation = createTranslation([0.0, 1.0, 2.0]);
  assertEquals(translation.type, "translation");
  assertEquals(translation.translation, [0.0, 1.0, 2.0]);
});

test("createDataset", () => {
  const dataset = createDataset("0", [1.0, 1.0], [0.0, 0.0]);
  assertEquals(dataset.path, "0");
  assertEquals(dataset.coordinateTransformations.length, 2);
  assertEquals(dataset.coordinateTransformations[0].type, "scale");
  assertEquals(dataset.coordinateTransformations[1].type, "translation");
});

test("createMetadata", () => {
  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("0", [1.0, 1.0], [0.0, 0.0])];

  const metadata = createMetadata(axes, datasets, "test", "0.4");
  assertEquals(metadata.name, "test");
  assertEquals(metadata.version, "0.4");
  assertEquals(metadata.axes.length, 2);
  assertEquals(metadata.datasets.length, 1);
});

test("createNgffImage", async () => {
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

test("createNgffImage keeps the pixel data it is given", async () => {
  const pixels = new Uint16Array([1, 2, 3, 4, 5, 6]);
  const image = await createNgffImage(
    pixels.buffer,
    [2, 3],
    "uint16",
    ["y", "x"],
    { y: 1.0, x: 1.0 },
    { y: 0.0, x: 0.0 },
  );
  const read = await zarr.get(image.data);
  assertEquals(Array.from(read.data as Uint16Array), [1, 2, 3, 4, 5, 6]);

  const fromNumbers = await createNgffImage(
    [0.5, 1.5],
    [2],
    "float32",
    ["x"],
    { x: 1.0 },
    { x: 0.0 },
  );
  const readNumbers = await zarr.get(fromNumbers.data);
  assertEquals(Array.from(readNumbers.data as Float32Array), [0.5, 1.5]);
});

test("createNgffImage refuses a pixel count that does not fill the shape", async () => {
  for (const data of [
    [1, 2, 3],
    [1, 2, 3, 4, 5],
  ]) {
    await assertRejects(
      () => createNgffImage(data, [2, 2], "uint8", ["y", "x"], {}, {}),
      Error,
      "4 pixels",
    );
  }
  // Empty data leaves the fill value, for pixels set afterwards.
  const empty = await createNgffImage([], [2, 2], "uint8", ["y", "x"], {}, {});
  assertEquals(
    Array.from((await zarr.get(empty.data)).data as Uint8Array),
    [0, 0, 0, 0],
  );
});

test("createNgffImage takes int64 values as numbers", async () => {
  const image = await createNgffImage([1, -2], [2], "int64", ["x"], {}, {});
  const read = await zarr.get(image.data);
  assertEquals(Array.from(read.data as BigInt64Array), [1n, -2n]);
});

test("createMultiscales", async () => {
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

// RFC4 validation tests
import {
  hasRfc4OrientationMetadata,
  validateRfc4Orientation,
} from "../src/utils/rfc4_validation.ts";

test("hasRfc4OrientationMetadata - returns true when orientation present", () => {
  const axes = [
    {
      name: "z",
      type: "space",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
    {
      name: "y",
      type: "space",
      orientation: { type: "anatomical", value: "anterior-to-posterior" },
    },
    {
      name: "x",
      type: "space",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
  ];
  assertEquals(hasRfc4OrientationMetadata(axes), true);
});

test("hasRfc4OrientationMetadata - returns false when no orientation", () => {
  const axes = [
    { name: "z", type: "space" },
    { name: "y", type: "space" },
    { name: "x", type: "space" },
  ];
  assertEquals(hasRfc4OrientationMetadata(axes), false);
});

test("hasRfc4OrientationMetadata - returns false for non-space axes", () => {
  const axes = [
    { name: "t", type: "time" },
    { name: "c", type: "channel" },
  ];
  assertEquals(hasRfc4OrientationMetadata(axes), false);
});

test("validateRfc4Orientation - valid LPS orientation", () => {
  const axes = [
    {
      name: "z",
      type: "space",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
    {
      name: "y",
      type: "space",
      orientation: { type: "anatomical", value: "anterior-to-posterior" },
    },
    {
      name: "x",
      type: "space",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
  ];
  // Should not throw
  validateRfc4Orientation(axes);
});

test("validateRfc4Orientation - invalid orientation value", () => {
  const axes = [
    {
      name: "z",
      type: "space",
      orientation: { type: "anatomical", value: "invalid-orientation" },
    },
  ];
  assertThrows(
    () => validateRfc4Orientation(axes),
    Error,
    "Invalid orientation value",
  );
});

test("validateRfc4Orientation - non-anatomical orientation type", () => {
  const axes = [
    {
      name: "z",
      type: "space",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
    {
      name: "y",
      type: "space",
      orientation: { type: "other-type", value: "some-value" },
    },
  ];
  assertThrows(
    () => validateRfc4Orientation(axes),
    Error,
    "must be anatomical",
  );
});

test("validateRfc4Orientation - partial orientation definition", () => {
  const axes = [
    {
      name: "z",
      type: "space",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
    { name: "y", type: "space" }, // missing orientation
    {
      name: "x",
      type: "space",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
  ];
  // RFC 4 makes orientation optional per spatial axis, so this is valid.
  validateRfc4Orientation(axes);
});
