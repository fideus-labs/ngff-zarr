// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import {
  AffineTransformationSchema,
  // RFC4 schemas
  AnatomicalOrientationSchema,
  ChannelAxisSchema,
  // Coordinate systems (RFC5)
  CoordinateSystemSchema,
  CoordinateTransformationSchema,
  // Zarr metadata
  MetadataSchema,
  ScaleTransformationSchema,
  SpaceAxisSchema,
  TimeAxisSchema,
  TranslationTransformationSchema,
} from "../src/schemas/index.ts";

// Test data examples based on NGFF specifications

// RFC4: Test anatomical orientation
const anatomicalOrientationExample = {
  type: "anatomical",
  value: "left-to-right",
};

const spaceAxisExample = {
  name: "x" as const,
  type: "space" as const,
  unit: "micrometer" as const,
  orientation: anatomicalOrientationExample,
};

const timeAxisExample = {
  name: "t" as const,
  type: "time" as const,
  unit: "second" as const,
};

const channelAxisExample = {
  name: "c" as const,
  type: "channel" as const,
};

// RFC5: Test coordinate transformations
const scaleTransformExample = {
  type: "scale" as const,
  scale: [0.5, 1.2, 0.8],
  input: "array_coords",
  output: "physical_coords",
};

const translationTransformExample = {
  type: "translation" as const,
  translation: [10.0, 20.0, 0.0],
  input: "array_coords",
  output: "physical_coords",
};

const affineTransformExample = {
  type: "affine" as const,
  affine: [
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 20.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ],
  input: "array_coords",
  output: "physical_coords",
};

// Coordinate system example
const coordinateSystemExample = {
  name: "physical_coords",
  axes: [
    spaceAxisExample,
    { name: "y" as const, type: "space" as const, unit: "micrometer" as const },
    { name: "z" as const, type: "space" as const, unit: "micrometer" as const },
  ],
};

// Dataset example with coordinate transformations
const datasetExample = {
  path: "0",
  coordinateTransformations: [
    scaleTransformExample,
    translationTransformExample,
  ],
};

// Complete metadata example
const metadataExample = {
  axes: [
    channelAxisExample,
    timeAxisExample,
    { name: "z" as const, type: "space" as const, unit: "micrometer" as const },
    { name: "y" as const, type: "space" as const, unit: "micrometer" as const },
    spaceAxisExample,
  ],
  datasets: [datasetExample],
  coordinateTransformations: [scaleTransformExample],
  name: "example_image",
  version: "0.4",
};

// Test function to validate all schemas
export function testSchemas() {
  console.log("Testing NGFF-Zarr Zod Schemas...\n");

  // Test RFC4 schemas
  console.log("Testing RFC4 (Anatomical Orientation) schemas:");

  try {
    AnatomicalOrientationSchema.parse(anatomicalOrientationExample);
    console.log("✓ AnatomicalOrientationSchema passed");

    SpaceAxisSchema.parse(spaceAxisExample);
    console.log("✓ SpaceAxisSchema passed");

    TimeAxisSchema.parse(timeAxisExample);
    console.log("✓ TimeAxisSchema passed");

    ChannelAxisSchema.parse(channelAxisExample);
    console.log("✓ ChannelAxisSchema passed");
  } catch (error) {
    console.error("✗ RFC4 schema validation failed:", error);
  }

  // Test RFC5 schemas
  console.log("\nTesting RFC5 (Coordinate Systems) schemas:");

  try {
    ScaleTransformationSchema.parse(scaleTransformExample);
    console.log("✓ ScaleTransformationSchema passed");

    TranslationTransformationSchema.parse(translationTransformExample);
    console.log("✓ TranslationTransformationSchema passed");

    AffineTransformationSchema.parse(affineTransformExample);
    console.log("✓ AffineTransformationSchema passed");

    CoordinateSystemSchema.parse(coordinateSystemExample);
    console.log("✓ CoordinateSystemSchema passed");
  } catch (error) {
    console.error("✗ RFC5 schema validation failed:", error);
  }

  // Test integrated metadata schema
  console.log("\nTesting integrated Metadata schema:");

  try {
    const metadataResult = MetadataSchema.parse(metadataExample);
    console.log("✓ MetadataSchema passed");
    console.log("Validated metadata:", JSON.stringify(metadataResult, null, 2));
  } catch (error) {
    console.error("✗ Metadata schema validation failed:", error);
  }

  // Test coordinate transformation union schema
  console.log("\nTesting CoordinateTransformationSchema union:");

  try {
    CoordinateTransformationSchema.parse(scaleTransformExample);
    console.log("✓ Scale transformation parsed successfully");

    CoordinateTransformationSchema.parse(affineTransformExample);
    console.log("✓ Affine transformation parsed successfully");
  } catch (error) {
    console.error("✗ Coordinate transformation union failed:", error);
  }

  console.log("\n✅ All schema tests completed!");
}

// Run tests if this file is executed directly
if (import.meta.main) {
  testSchemas();
}
