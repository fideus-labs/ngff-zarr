#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * Test data layout correctness with small, non-uniform shapes
 */

import { assertEquals } from "@std/assert";
import { readImageNode, writeImageNode } from "@itk-wasm/image-io";
import type { Image as ItkWasmImage } from "itk-wasm";
import { join } from "@std/path";
import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import { ngffImageToItkImage } from "../src/io/ngff_image_to_itk_image.ts";
import * as zarr from "zarrita";

const TEST_OUTPUT_DIR = join(Deno.cwd(), "test", "output");
await Deno.mkdir(TEST_OUTPUT_DIR, { recursive: true });

/**
 * Create a simple test image with known pattern
 * Pattern: value at (x, y, z) = x + y*10 + z*100
 */
function createTestImage3D(): ItkWasmImage {
  const xSize = 3;
  const ySize = 4;
  const zSize = 2;
  const totalSize = xSize * ySize * zSize;

  // Create data in ITK column-major order (x varies fastest)
  const data = new Int16Array(totalSize);
  let idx = 0;
  for (let z = 0; z < zSize; z++) {
    for (let y = 0; y < ySize; y++) {
      for (let x = 0; x < xSize; x++) {
        const value = x + y * 10 + z * 100;
        data[idx++] = value;
      }
    }
  }

  console.log("Created 3D test data:");
  console.log("  Size: [x=%d, y=%d, z=%d]", xSize, ySize, zSize);
  console.log("  First few values:", Array.from(data.slice(0, 12)));
  console.log("  Pattern: value(x,y,z) = x + y*10 + z*100");

  const image: ItkWasmImage = {
    imageType: {
      dimension: 3,
      componentType: "int16",
      pixelType: "Scalar",
      components: 1,
    },
    name: "test-3d",
    origin: [0.5, 1.0, 1.5], // Different origins to test
    spacing: [1.0, 2.0, 3.0], // Different spacings to test
    direction: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), // Identity
    size: [xSize, ySize, zSize], // ITK physical order: [x, y, z]
    metadata: new Map(),
    data: data,
  };

  return image;
}

/**
 * Verify image has expected values at specific coordinates
 */
function verifyImageData(
  data: Int16Array,
  size: number[],
  testName: string,
): void {
  // ITK size is [x, y, z], data is column-major
  const [xSize, ySize, zSize] = size;

  console.log(`\nVerifying ${testName}:`);
  console.log(`  Size: [${xSize}, ${ySize}, ${zSize}]`);

  // Test some specific coordinates
  const testCases = [
    { x: 0, y: 0, z: 0, expected: 0 },
    { x: 1, y: 0, z: 0, expected: 1 },
    { x: 2, y: 0, z: 0, expected: 2 },
    { x: 0, y: 1, z: 0, expected: 10 },
    { x: 1, y: 1, z: 0, expected: 11 },
    { x: 0, y: 0, z: 1, expected: 100 },
    { x: 1, y: 2, z: 1, expected: 121 },
  ];

  for (const { x, y, z, expected } of testCases) {
    // Column-major indexing: x + y*xSize + z*xSize*ySize
    const idx = x + y * xSize + z * xSize * ySize;
    const actual = data[idx];
    console.log(
      `  [x=%d, y=%d, z=%d] -> idx=%d: expected=%d, actual=%d %s`,
      x,
      y,
      z,
      idx,
      expected,
      actual,
      actual === expected ? "✓" : "✗",
    );
    assertEquals(
      actual,
      expected,
      `Value at (${x},${y},${z}) should be ${expected} but got ${actual}`,
    );
  }
}

Deno.test("1. Write to disk and read back - verify pixel data", async () => {
  const originalImage = createTestImage3D();

  // Write to NRRD file
  const outputPath = join(TEST_OUTPUT_DIR, "test_layout_3d.nrrd");
  await writeImageNode(originalImage, outputPath);
  console.log(`\nWrote image to: ${outputPath}`);

  // Read back
  const readImage = await readImageNode(outputPath);
  console.log("\nRead image back from disk");

  // Verify metadata
  assertEquals(readImage.size, originalImage.size, "Size should match");
  console.log("✓ Size matches:", readImage.size);

  // Note: spacing/origin may have floating point differences, check approximately
  for (let i = 0; i < 3; i++) {
    const diff = Math.abs(readImage.spacing[i] - originalImage.spacing[i]);
    if (diff > 0.0001) {
      throw new Error(
        `Spacing[${i}] mismatch: ${readImage.spacing[i]} vs ${
          originalImage.spacing[i]
        }`,
      );
    }
    const originDiff = Math.abs(readImage.origin[i] - originalImage.origin[i]);
    if (originDiff > 0.0001) {
      throw new Error(
        `Origin[${i}] mismatch: ${readImage.origin[i]} vs ${
          originalImage.origin[i]
        }`,
      );
    }
  }
  console.log("✓ Spacing matches:", readImage.spacing);
  console.log("✓ Origin matches:", readImage.origin);

  // Verify pixel data
  const readData = readImage.data as Int16Array;
  verifyImageData(readData, readImage.size, "Read from disk");
});

Deno.test(
  "2. ITK → NGFF → ITK round-trip - verify metadata and pixel data",
  async () => {
    const originalImage = createTestImage3D();

    console.log("\n=== Original ITK Image ===");
    console.log("  size (ITK [x,y,z]):", originalImage.size);
    console.log("  spacing (ITK [x,y,z]):", originalImage.spacing);
    console.log("  origin (ITK [x,y,z]):", originalImage.origin);
    verifyImageData(
      originalImage.data as Int16Array,
      originalImage.size,
      "Original ITK",
    );

    // Convert to NGFF
    const ngffImage = await itkImageToNgffImage(originalImage, {
      addAnatomicalOrientation: false,
    });

    console.log("\n=== NGFF Image ===");
    console.log("  dims:", ngffImage.dims);
    console.log("  scale:", ngffImage.scale);
    console.log("  translation:", ngffImage.translation);

    // Read zarr data
    const zarrData = await zarr.get(ngffImage.data);
    console.log("  shape (NGFF [z,y,x]):", zarrData.shape);

    // Expected NGFF shape should be reversed: [z, y, x] = [2, 4, 3]
    assertEquals(
      zarrData.shape,
      [2, 4, 3],
      "NGFF shape should be [z,y,x] = [2,4,3]",
    );

    // Verify NGFF data can be indexed correctly
    // In NGFF with shape [z,y,x], element at [z,y,x] should have value x + y*10 + z*100
    const ngffTypedData = zarrData.data as Int16Array;
    console.log("\nVerifying NGFF data indexing:");

    const ngffTestCases = [
      { z: 0, y: 0, x: 0, expected: 0 },
      { z: 0, y: 0, x: 1, expected: 1 },
      { z: 0, y: 1, x: 0, expected: 10 },
      { z: 1, y: 0, x: 0, expected: 100 },
      { z: 1, y: 2, x: 1, expected: 121 },
    ];

    for (const { z, y, x, expected } of ngffTestCases) {
      // C-order indexing for shape [2, 4, 3]: z*4*3 + y*3 + x
      const idx = z * 4 * 3 + y * 3 + x;
      const actual = ngffTypedData[idx];
      console.log(
        `  [z=%d, y=%d, x=%d] -> idx=%d: expected=%d, actual=%d %s`,
        z,
        y,
        x,
        idx,
        expected,
        actual,
        actual === expected ? "✓" : "✗",
      );
      assertEquals(
        actual,
        expected,
        `NGFF value at [${z},${y},${x}] should be ${expected} but got ${actual}`,
      );
    }

    // Convert back to ITK
    const roundtripImage = await ngffImageToItkImage(ngffImage);

    console.log("\n=== Round-trip ITK Image ===");
    console.log("  size (ITK [x,y,z]):", roundtripImage.size);
    console.log("  spacing (ITK [x,y,z]):", roundtripImage.spacing);
    console.log("  origin (ITK [x,y,z]):", roundtripImage.origin);

    // Verify metadata
    assertEquals(
      roundtripImage.size,
      originalImage.size,
      "Round-trip size should match",
    );
    console.log("✓ Size matches");

    for (let i = 0; i < 3; i++) {
      const diff = Math.abs(
        roundtripImage.spacing[i] - originalImage.spacing[i],
      );
      if (diff > 0.0001) {
        throw new Error(
          `Spacing[${i}] mismatch: ${roundtripImage.spacing[i]} vs ${
            originalImage.spacing[i]
          }`,
        );
      }
      const originDiff = Math.abs(
        roundtripImage.origin[i] - originalImage.origin[i],
      );
      if (originDiff > 0.0001) {
        throw new Error(
          `Origin[${i}] mismatch: ${roundtripImage.origin[i]} vs ${
            originalImage.origin[i]
          }`,
        );
      }
    }
    console.log("✓ Spacing matches");
    console.log("✓ Origin matches");

    // Verify pixel data
    const roundtripData = roundtripImage.data as Int16Array;
    verifyImageData(roundtripData, roundtripImage.size, "Round-trip ITK");

    // Also verify all elements match
    for (let i = 0; i < roundtripData.length; i++) {
      assertEquals(
        roundtripData[i],
        (originalImage.data as Int16Array)[i],
        `All data elements should match at index ${i}`,
      );
    }
    console.log("✓ All pixel data matches");
  },
);

Deno.test("3. Test 2D image (no z dimension)", async () => {
  const xSize = 5;
  const ySize = 3;
  const totalSize = xSize * ySize;

  // Create 2D data in column-major order
  const data = new Uint8Array(totalSize);
  let idx = 0;
  for (let y = 0; y < ySize; y++) {
    for (let x = 0; x < xSize; x++) {
      data[idx++] = x + y * 10;
    }
  }

  const image2D: ItkWasmImage = {
    imageType: {
      dimension: 2,
      componentType: "uint8",
      pixelType: "Scalar",
      components: 1,
    },
    name: "test-2d",
    origin: [0.0, 0.0],
    spacing: [1.0, 1.0],
    direction: new Float64Array([1, 0, 0, 1]),
    size: [xSize, ySize], // [x, y]
    metadata: new Map(),
    data: data,
  };

  console.log("\n=== 2D Image Test ===");
  console.log("Size [x,y]:", image2D.size);

  // Convert to NGFF
  const ngffImage = await itkImageToNgffImage(image2D, {
    addAnatomicalOrientation: false,
  });
  console.log("NGFF dims:", ngffImage.dims);

  const zarrData = await zarr.get(ngffImage.data);
  console.log("NGFF shape [y,x]:", zarrData.shape);

  // Should be reversed: [y, x] = [3, 5]
  assertEquals(zarrData.shape, [3, 5], "2D shape should be [y,x] = [3,5]");

  // Verify a few values
  const ngff2DData = zarrData.data as Uint8Array;
  // [y=0, x=0] -> 0
  assertEquals(ngff2DData[0 * 5 + 0], 0);
  // [y=0, x=1] -> 1
  assertEquals(ngff2DData[0 * 5 + 1], 1);
  // [y=1, x=0] -> 10
  assertEquals(ngff2DData[1 * 5 + 0], 10);
  // [y=2, x=4] -> 24
  assertEquals(ngff2DData[2 * 5 + 4], 24);

  console.log("✓ 2D data layout correct");

  // Round-trip
  const roundtrip2D = await ngffImageToItkImage(ngffImage);
  assertEquals(roundtrip2D.size, image2D.size);

  const rt2DData = roundtrip2D.data as Uint8Array;
  for (let i = 0; i < data.length; i++) {
    assertEquals(rt2DData[i], data[i], `2D round-trip data at index ${i}`);
  }
  console.log("✓ 2D round-trip successful");
});
