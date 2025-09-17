// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertThrows } from "@std/assert";
import {
  AxisSchema,
  MetadataSchema,
  OmeroChannelSchema,
  ScaleSchema,
  TranslationSchema,
} from "../src/schemas/zarr_metadata.ts";
import { MethodsSchema } from "../src/schemas/methods.ts";

Deno.test("AxisSchema validation", () => {
  const validAxis = {
    name: "x",
    type: "space",
    unit: "micrometer",
  };

  const result = AxisSchema.parse(validAxis);
  assertEquals(result.name, "x");
  assertEquals(result.type, "space");
  assertEquals(result.unit, "micrometer");
});

Deno.test("AxisSchema validation - invalid name", () => {
  const invalidAxis = {
    name: "invalid",
    type: "space",
  };

  assertThrows(() => AxisSchema.parse(invalidAxis));
});

Deno.test("ScaleSchema validation", () => {
  const validScale = {
    type: "scale",
    scale: [1.0, 2.0, 3.0],
  };

  const result = ScaleSchema.parse(validScale);
  assertEquals(result.type, "scale");
  assertEquals(result.scale, [1.0, 2.0, 3.0]);
});

Deno.test("TranslationSchema validation", () => {
  const validTranslation = {
    type: "translation",
    translation: [0.0, 1.0, 2.0],
  };

  const result = TranslationSchema.parse(validTranslation);
  assertEquals(result.type, "translation");
  assertEquals(result.translation, [0.0, 1.0, 2.0]);
});

Deno.test("OmeroChannelSchema validation", () => {
  const validChannel = {
    color: "FF0000",
    window: {
      min: 0,
      max: 255,
      start: 10,
      end: 200,
    },
    label: "Red Channel",
  };

  const result = OmeroChannelSchema.parse(validChannel);
  assertEquals(result.color, "FF0000");
  assertEquals(result.label, "Red Channel");
});

Deno.test("OmeroChannelSchema validation - invalid color", () => {
  const invalidChannel = {
    color: "GG0000",
    window: {
      min: 0,
      max: 255,
      start: 10,
      end: 200,
    },
  };

  assertThrows(() => OmeroChannelSchema.parse(invalidChannel));
});

Deno.test("MetadataSchema validation", () => {
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

  const result = MetadataSchema.parse(validMetadata);
  assertEquals(result.name, "test_image");
  assertEquals(result.version, "0.4");
  assertEquals(result.axes.length, 2);
  assertEquals(result.datasets.length, 1);
});

Deno.test("MethodsSchema validation", () => {
  const validMethod = "itkwasm_gaussian";
  const result = MethodsSchema.parse(validMethod);
  assertEquals(result, "itkwasm_gaussian");

  assertThrows(() => MethodsSchema.parse("invalid_method"));
});
