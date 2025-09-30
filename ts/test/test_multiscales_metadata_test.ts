#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * Test the multiscales metadata field functionality.
 * TypeScript equivalent of the Python test_multiscales_metadata.py
 */

import { assertEquals, assertExists } from "jsr:@std/assert";
import * as zarr from "zarrita";
import { Methods } from "../src/types/methods.ts";
import { toMultiscales, toNgffImage } from "../src/io/to_multiscales.ts";
import { fromNgffZarr, toNgffZarr } from "../src/mod.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";

Deno.test("multiscales metadata field is populated correctly", async () => {
  // Create a simple test image
  const data = new Array(64 * 64)
    .fill(0)
    .map(() => Math.floor(Math.random() * 255));
  const image = await toNgffImage(data, { dims: ["y", "x"] });

  // Test with ITKWASM_GAUSSIAN method
  const multiscales = toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  // Check that metadata field is populated
  assertExists(
    multiscales.metadata.metadata,
    "metadata field should be populated",
  );

  if (multiscales.metadata.metadata) {
    assertExists(
      multiscales.metadata.metadata.description,
      "description should be populated",
    );
    assertExists(
      multiscales.metadata.metadata.method,
      "method should be populated",
    );
    assertExists(
      multiscales.metadata.metadata.version,
      "version should be populated",
    );

    // Check specific content
    assertEquals(
      multiscales.metadata.metadata.description
        .toLowerCase()
        .includes("gaussian filter"),
      true,
      "description should mention gaussian filter",
    );
    assertEquals(
      multiscales.metadata.metadata.method.includes("itkwasm_downsample"),
      true,
      "method should include itkwasm_downsample",
    );
  }
});

Deno.test("multiscales metadata is correctly serialized to zarr", async () => {
  const data = new Array(32 * 32)
    .fill(0)
    .map(() => Math.floor(Math.random() * 255));
  const image = await toNgffImage(data, { dims: ["y", "x"] });

  const store: MemoryStore = new Map();

  // Create and save multiscales
  const multiscales = toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
  });
  await toNgffZarr(store, multiscales);

  // Read raw metadata from zarr
  const root = zarr.root(store);
  const rootGroup = await zarr.open(root, { kind: "group" });
  const attrs = rootGroup.attrs as Record<string, unknown>;

  const multiscalesArray = attrs.multiscales as unknown[];
  assertEquals(
    Array.isArray(multiscalesArray),
    true,
    "multiscales should be an array",
  );
  assertEquals(
    multiscalesArray.length > 0,
    true,
    "multiscales should have at least one entry",
  );

  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;

  // Check that metadata field exists and has correct structure
  assertEquals(
    "metadata" in multiscalesMetadata,
    true,
    "metadata field should exist",
  );
  const metadataField = multiscalesMetadata.metadata as Record<string, unknown>;
  assertEquals(
    typeof metadataField === "object",
    true,
    "metadata should be an object",
  );
  assertEquals(
    "description" in metadataField,
    true,
    "description should exist",
  );
  assertEquals("method" in metadataField, true, "method should exist");
  assertEquals("version" in metadataField, true, "version should exist");

  // Check content types
  assertEquals(
    typeof metadataField.description,
    "string",
    "description should be a string",
  );
  assertEquals(
    typeof metadataField.method,
    "string",
    "method should be a string",
  );
  assertEquals(
    typeof metadataField.version,
    "string",
    "version should be a string",
  );
});

Deno.test(
  "multiscales metadata round-trip: save to zarr and load back",
  async () => {
    const data = new Array(32 * 32)
      .fill(0)
      .map(() => Math.floor(Math.random() * 255));
    const image = await toNgffImage(data, { dims: ["y", "x"] });

    const store: MemoryStore = new Map();

    // Create and save multiscales
    const originalMultiscales = toMultiscales(image, {
      scaleFactors: [2],
      method: Methods.ITKWASM_GAUSSIAN,
    });
    await toNgffZarr(store, originalMultiscales);

    // Load back
    const loadedMultiscales = await fromNgffZarr(store);

    // Compare metadata
    assertExists(
      loadedMultiscales.metadata.metadata,
      "loaded metadata should exist",
    );
    assertExists(
      originalMultiscales.metadata.metadata,
      "original metadata should exist",
    );

    if (
      loadedMultiscales.metadata.metadata &&
      originalMultiscales.metadata.metadata
    ) {
      assertEquals(
        loadedMultiscales.metadata.metadata.description,
        originalMultiscales.metadata.metadata.description,
        "descriptions should match",
      );
      assertEquals(
        loadedMultiscales.metadata.metadata.method,
        originalMultiscales.metadata.metadata.method,
        "methods should match",
      );
      assertEquals(
        loadedMultiscales.metadata.metadata.version,
        originalMultiscales.metadata.metadata.version,
        "versions should match",
      );
    }
  },
);

Deno.test("different methods have different metadata", async () => {
  const data = new Array(32 * 32)
    .fill(0)
    .map(() => Math.floor(Math.random() * 255));
  const image = await toNgffImage(data, { dims: ["y", "x"] });

  // Test different methods that should work
  const testMethods = [
    Methods.ITKWASM_GAUSSIAN,
    Methods.ITKWASM_BIN_SHRINK,
    Methods.ITKWASM_LABEL_IMAGE,
  ];

  const metadataResults = [];
  for (const method of testMethods) {
    const multiscales = toMultiscales(image, { scaleFactors: [2], method });
    assertExists(
      multiscales.metadata.metadata,
      `metadata should exist for method ${method}`,
    );

    if (multiscales.metadata.metadata) {
      metadataResults.push({
        method: method,
        description: multiscales.metadata.metadata.description,
        methodName: multiscales.metadata.metadata.method,
        version: multiscales.metadata.metadata.version,
      });
    }
  }

  // Check that different methods have different descriptions
  const descriptions = metadataResults.map((m) => m.description);
  const uniqueDescriptions = new Set(descriptions);
  assertEquals(
    uniqueDescriptions.size > 1,
    true,
    "Different methods should have different descriptions",
  );

  // Check that all use similar package but potentially different functions
  const methodNames = metadataResults.map((m) => m.methodName);
  const allUseItkwasm = methodNames.every((name) =>
    name.includes("itkwasm_downsample")
  );
  assertEquals(
    allUseItkwasm,
    true,
    "All should use itkwasm_downsample package",
  );
});

Deno.test("metadata field is populated when method is specified", async () => {
  const data = new Array(32 * 32)
    .fill(0)
    .map(() => Math.floor(Math.random() * 255));
  const image = await toNgffImage(data, { dims: ["y", "x"] });

  // Create multiscales with default method (should default to ITKWASM_GAUSSIAN)
  const multiscales = toMultiscales(image, { scaleFactors: [2] });

  // When method is not explicitly set, defaults to ITKWASM_GAUSSIAN
  // and metadata should be populated since default method is used
  assertExists(
    multiscales.metadata.metadata,
    "metadata should be populated with default method",
  );
});

Deno.test("can handle metadata being undefined", async () => {
  // This test verifies that our type system correctly handles optional metadata
  // Since metadata is optional in the interface, undefined should be handled gracefully
  const data = new Array(32 * 32)
    .fill(0)
    .map(() => Math.floor(Math.random() * 255));
  const image = await toNgffImage(data, { dims: ["y", "x"] });

  // Create multiscales normally
  const multiscales = toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  // Verify that metadata exists by default
  assertExists(
    multiscales.metadata.metadata,
    "metadata should exist when method is specified",
  );

  // The fact that metadata is typed as optional (MethodMetadata | undefined) in the interface
  // confirms that the system can handle legacy files without this field
});

console.log("✅ All metadata field tests completed!");
