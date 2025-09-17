// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertThrows } from "@std/assert";
import {
  AffineTransformationSchema,
  AnatomicalOrientationSchema,
  // RFC4 schemas
  AnatomicalOrientationValuesSchema,
  // Enhanced zarr metadata with RFC4 support
  AxisSchema,
  ChannelAxisSchema,
  // RFC5 coordinate systems
  CoordinateSystemSchema,
  CoordinateTransformationSchema,
  MetadataSchema,
  ScaleTransformationSchema,
  SequenceTransformationSchema,
  SpaceAxisSchema,
  TimeAxisSchema,
  TranslationTransformationSchema,
} from "../src/schemas/index.ts";

// RFC4 Tests
Deno.test("AnatomicalOrientationValuesSchema validation", () => {
  const validValues = [
    "left-to-right",
    "right-to-left",
    "anterior-to-posterior",
    "posterior-to-anterior",
    "rostral-to-caudal",
    "proximal-to-distal",
  ];

  validValues.forEach((value) => {
    const result = AnatomicalOrientationValuesSchema.parse(value);
    assertEquals(result, value);
  });

  assertThrows(() =>
    AnatomicalOrientationValuesSchema.parse("invalid-orientation")
  );
});

Deno.test("AnatomicalOrientationSchema validation", () => {
  const validOrientation = {
    type: "anatomical",
    value: "left-to-right",
  };

  const result = AnatomicalOrientationSchema.parse(validOrientation);
  assertEquals(result.type, "anatomical");
  assertEquals(result.value, "left-to-right");

  // Test missing fields
  assertThrows(() => AnatomicalOrientationSchema.parse({ type: "anatomical" }));
  assertThrows(() =>
    AnatomicalOrientationSchema.parse({ value: "left-to-right" })
  );
});

Deno.test("SpaceAxisSchema validation", () => {
  const validSpaceAxis = {
    name: "x",
    type: "space",
    unit: "micrometer",
    orientation: {
      type: "anatomical",
      value: "left-to-right",
    },
  };

  const result = SpaceAxisSchema.parse(validSpaceAxis);
  assertEquals(result.name, "x");
  assertEquals(result.type, "space");
  assertEquals(result.unit, "micrometer");
  assertEquals(result.orientation?.type, "anatomical");
  assertEquals(result.orientation?.value, "left-to-right");

  // Test without orientation (should be optional)
  const validWithoutOrientation = {
    name: "y",
    type: "space",
    unit: "millimeter",
  };

  const result2 = SpaceAxisSchema.parse(validWithoutOrientation);
  assertEquals(result2.name, "y");
  assertEquals(result2.orientation, undefined);
});

Deno.test("SpaceAxisSchema validation - invalid names", () => {
  const invalidSpaceAxis = {
    name: "t", // time axis name not allowed for space
    type: "space",
    unit: "micrometer",
  };

  assertThrows(() => SpaceAxisSchema.parse(invalidSpaceAxis));
});

Deno.test("TimeAxisSchema validation", () => {
  const validTimeAxis = {
    name: "t",
    type: "time",
    unit: "second",
  };

  const result = TimeAxisSchema.parse(validTimeAxis);
  assertEquals(result.name, "t");
  assertEquals(result.type, "time");
  assertEquals(result.unit, "second");

  // Test other time units
  const millisecondAxis = {
    name: "t",
    type: "time",
    unit: "millisecond",
  };

  const result2 = TimeAxisSchema.parse(millisecondAxis);
  assertEquals(result2.unit, "millisecond");
});

Deno.test("ChannelAxisSchema validation", () => {
  const validChannelAxis = {
    name: "c",
    type: "channel",
  };

  const result = ChannelAxisSchema.parse(validChannelAxis);
  assertEquals(result.name, "c");
  assertEquals(result.type, "channel");

  // Test invalid name
  assertThrows(() =>
    ChannelAxisSchema.parse({
      name: "x",
      type: "channel",
    })
  );
});

// RFC5 Tests
Deno.test("ScaleTransformationSchema validation", () => {
  const validScale = {
    type: "scale",
    scale: [0.5, 1.2, 0.8],
    input: "array_coords",
    output: "physical_coords",
  };

  const result = ScaleTransformationSchema.parse(validScale);
  assertEquals(result.type, "scale");
  assertEquals(result.scale, [0.5, 1.2, 0.8]);
  assertEquals(result.input, "array_coords");
  assertEquals(result.output, "physical_coords");

  // Test with path instead of scale array
  const validWithPath = {
    type: "scale",
    path: "/path/to/scale/data",
    input: "array_coords",
    output: "physical_coords",
  };

  const result2 = ScaleTransformationSchema.parse(validWithPath);
  assertEquals(result2.path, "/path/to/scale/data");

  // Test that either scale or path is required
  assertThrows(() =>
    ScaleTransformationSchema.parse({
      type: "scale",
      input: "array_coords",
      output: "physical_coords",
    })
  );
});

Deno.test("TranslationTransformationSchema validation", () => {
  const validTranslation = {
    type: "translation",
    translation: [10.0, 20.0, 0.0],
    input: "array_coords",
    output: "physical_coords",
  };

  const result = TranslationTransformationSchema.parse(validTranslation);
  assertEquals(result.type, "translation");
  assertEquals(result.translation, [10.0, 20.0, 0.0]);
});

Deno.test("AffineTransformationSchema validation", () => {
  const validAffine = {
    type: "affine",
    affine: [
      [1.0, 0.0, 0.0, 10.0],
      [0.0, 1.0, 0.0, 20.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ],
    input: "array_coords",
    output: "physical_coords",
  };

  const result = AffineTransformationSchema.parse(validAffine);
  assertEquals(result.type, "affine");
  assertEquals(result.affine?.length, 4);
  assertEquals(result.affine?.[0].length, 4);
});

Deno.test("CoordinateSystemSchema validation", () => {
  const validCoordinateSystem = {
    name: "physical_coords",
    axes: [
      { name: "x", type: "space", unit: "micrometer" },
      { name: "y", type: "space", unit: "micrometer" },
      { name: "z", type: "space", unit: "micrometer" },
    ],
  };

  const result = CoordinateSystemSchema.parse(validCoordinateSystem);
  assertEquals(result.name, "physical_coords");
  assertEquals(result.axes.length, 3);

  // Test empty name (should fail)
  assertThrows(() =>
    CoordinateSystemSchema.parse({
      name: "",
      axes: [],
    })
  );
});

Deno.test("SequenceTransformationSchema validation", () => {
  const validSequence = {
    type: "sequence",
    transformations: [
      { type: "scale", scale: [0.5, 1.0, 2.0] },
      { type: "translation", translation: [10.0, 20.0, 0.0] },
    ],
    input: "array_coords",
    output: "physical_coords",
  };

  const result = SequenceTransformationSchema.parse(validSequence);
  assertEquals(result.type, "sequence");
  assertEquals(result.transformations.length, 2);
});

Deno.test("CoordinateTransformationSchema union validation", () => {
  // Test that union accepts scale transformation
  const scaleTransform = {
    type: "scale",
    scale: [1.0, 2.0, 3.0],
  };

  const result1 = CoordinateTransformationSchema.parse(scaleTransform);
  assertEquals(result1.type, "scale");

  // Test that union accepts affine transformation
  const affineTransform = {
    type: "affine",
    affine: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
  };

  const result2 = CoordinateTransformationSchema.parse(affineTransform);
  assertEquals(result2.type, "affine");

  // Test invalid transformation type
  assertThrows(() =>
    CoordinateTransformationSchema.parse({
      type: "invalid_type",
      data: [1, 2, 3],
    })
  );
});

// Enhanced metadata tests with RFC4 support
Deno.test("Enhanced AxisSchema with orientation", () => {
  const spaceAxisWithOrientation = {
    name: "x",
    type: "space",
    unit: "micrometer",
    orientation: {
      type: "anatomical",
      value: "left-to-right",
    },
  };

  const result = AxisSchema.parse(spaceAxisWithOrientation);
  assertEquals(result.name, "x");
  assertEquals(result.type, "space");
  assertEquals(result.unit, "micrometer");
  assertEquals(result.orientation?.type, "anatomical");
  assertEquals(result.orientation?.value, "left-to-right");
});

Deno.test("Enhanced MetadataSchema with RFC4 and RFC5 features", () => {
  const enhancedMetadata = {
    axes: [
      { name: "c", type: "channel" },
      { name: "t", type: "time", unit: "second" },
      {
        name: "z",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "superior-to-inferior" },
      },
      {
        name: "y",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "anterior-to-posterior" },
      },
      {
        name: "x",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "left-to-right" },
      },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          { type: "scale", scale: [1.0, 0.1, 0.5, 0.5, 0.5] },
          { type: "translation", translation: [0.0, 0.0, 0.0, 0.0, 0.0] },
        ],
      },
    ],
    coordinateTransformations: [
      {
        type: "affine",
        affine: [
          [1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1],
        ],
      },
    ],
    name: "brain_image",
    version: "0.5",
  };

  const result = MetadataSchema.parse(enhancedMetadata);
  assertEquals(result.name, "brain_image");
  assertEquals(result.version, "0.5");
  assertEquals(result.axes.length, 5);

  // Check that space axes have orientation
  const spaceAxes = result.axes.filter((axis) => axis.type === "space");
  assertEquals(spaceAxes.length, 3);
  spaceAxes.forEach((axis) => {
    assertEquals(axis.orientation?.type, "anatomical");
  });
});
