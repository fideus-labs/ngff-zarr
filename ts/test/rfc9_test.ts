// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Tests for RFC-9: Zipped OME-Zarr (.ozx) support
 *
 * @see https://ngff.openmicroscopy.org/rfc/9/index.html
 */

import { assertEquals, assertExists, assertRejects } from "@std/assert";
import { join } from "@std/path";
import { readImageNode } from "@itk-wasm/image-io";

import {
  getZipFileCompressionMethod,
  getZipFileList,
  isOzxPath,
  memoryStoreToZip,
  orderFilesForRfc9,
  readOzxVersion,
} from "../src/io/rfc9_zip.ts";
import { toNgffZarr, toNgffZarrOzx } from "../src/io/to_ngff_zarr.ts";
import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";
import { Methods } from "../src/types/methods.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import * as zarr from "zarrita";

// Path to Python test data directory
const TEST_DATA_DIR = join(Deno.cwd(), "..", "py", "test", "data");
const INPUT_DIR = join(TEST_DATA_DIR, "input");
const OUTPUT_DIR = join(Deno.cwd(), "test", "output");

// Ensure output directory exists
await Deno.mkdir(OUTPUT_DIR, { recursive: true });

// ============================================================================
// isOzxPath tests
// ============================================================================

Deno.test("isOzxPath - detects .ozx extension", () => {
  assertEquals(isOzxPath("test.ozx"), true);
  assertEquals(isOzxPath("/path/to/file.ozx"), true);
  assertEquals(isOzxPath("C:\\path\\to\\file.ozx"), true);
  assertEquals(isOzxPath("file.OZX"), false); // Case sensitive
});

Deno.test("isOzxPath - rejects non-.ozx extensions", () => {
  assertEquals(isOzxPath("test.zarr"), false);
  assertEquals(isOzxPath("test.ome.zarr"), false);
  assertEquals(isOzxPath("test.zip"), false);
  assertEquals(isOzxPath("test.ozx.bak"), false);
  assertEquals(isOzxPath("ozx"), false);
});

// ============================================================================
// orderFilesForRfc9 tests
// ============================================================================

Deno.test("orderFilesForRfc9 - root zarr.json first", () => {
  const files = ["scale0/zarr.json", "zarr.json", "scale1/zarr.json"];
  const ordered = orderFilesForRfc9(files);

  assertEquals(ordered[0], "zarr.json");
});

Deno.test("orderFilesForRfc9 - breadth-first zarr.json ordering", () => {
  const files = [
    "scale0/image/zarr.json",
    "zarr.json",
    "scale0/zarr.json",
    "scale1/image/zarr.json",
    "scale1/zarr.json",
    "c/0/0",
  ];
  const ordered = orderFilesForRfc9(files);

  // zarr.json files should come first, in breadth-first order
  assertEquals(ordered[0], "zarr.json");
  // Level 1 (depth 2) before level 2 (depth 3)
  const zarrJsons = ordered.filter((f) => f.endsWith("zarr.json"));
  assertEquals(zarrJsons[0], "zarr.json");
  // scale0/zarr.json and scale1/zarr.json before deeper ones
  assertEquals(zarrJsons[1].split("/").length, 2);
  assertEquals(zarrJsons[2].split("/").length, 2);

  // Data files should come after all zarr.json files
  const dataFileIndex = ordered.indexOf("c/0/0");
  const lastZarrJsonIndex = ordered.lastIndexOf(
    ordered.filter((f) => f.endsWith("zarr.json")).pop()!,
  );
  assertEquals(dataFileIndex > lastZarrJsonIndex, true);
});

// ============================================================================
// memoryStoreToZip tests
// ============================================================================

Deno.test("memoryStoreToZip - creates valid ZIP", () => {
  // Create a simple memory store
  const store = new Map<string, Uint8Array>();
  store.set("zarr.json", new TextEncoder().encode('{"zarr_format": 3}'));
  store.set("scale0/zarr.json", new TextEncoder().encode('{"zarr_format": 3}'));
  store.set("c/0/0", new Uint8Array([1, 2, 3, 4]));

  const zipData = memoryStoreToZip(store, { version: "0.5" });

  // Verify it's a valid ZIP (magic number)
  assertEquals(zipData[0], 0x50); // 'P'
  assertEquals(zipData[1], 0x4b); // 'K'
  assertEquals(zipData[2], 0x03); // Local file header
  assertEquals(zipData[3], 0x04);
});

Deno.test("memoryStoreToZip - adds ZIP comment with version", () => {
  const store = new Map<string, Uint8Array>();
  store.set("zarr.json", new TextEncoder().encode('{"zarr_format": 3}'));

  const zipData = memoryStoreToZip(store, { version: "0.5" });
  const version = readOzxVersion(zipData);

  assertEquals(version, "0.5");
});

Deno.test("memoryStoreToZip - orders files correctly", () => {
  const store = new Map<string, Uint8Array>();
  // Add in wrong order
  store.set("c/0/0", new Uint8Array([1, 2, 3, 4]));
  store.set("scale0/zarr.json", new TextEncoder().encode("{}"));
  store.set("zarr.json", new TextEncoder().encode("{}"));

  const zipData = memoryStoreToZip(store);
  const files = getZipFileList(zipData);

  // root zarr.json should be first
  assertEquals(files[0], "zarr.json");
  // Then other zarr.json files
  assertEquals(files[1], "scale0/zarr.json");
  // Data files last
  assertEquals(files[2], "c/0/0");
});

Deno.test("memoryStoreToZip - uses no compression (ZIP_STORED)", () => {
  const store = new Map<string, Uint8Array>();
  store.set("zarr.json", new TextEncoder().encode('{"zarr_format": 3}'));
  store.set("data", new Uint8Array(1000).fill(42)); // Compressible data

  const zipData = memoryStoreToZip(store);

  // Check compression method for each file (should be 0 = stored)
  const zarrJsonCompression = getZipFileCompressionMethod(zipData, "zarr.json");
  const dataCompression = getZipFileCompressionMethod(zipData, "data");

  assertEquals(zarrJsonCompression, 0); // ZIP_STORED
  assertEquals(dataCompression, 0); // ZIP_STORED
});

// ============================================================================
// toNgffZarrOzx tests with synthetic data
// ============================================================================

Deno.test("toNgffZarrOzx - creates valid .ozx file", async () => {
  // Create synthetic test data
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const shape = [10, 20];
  const chunks = [5, 10];

  const array = await zarr.create(root.resolve("data"), {
    shape,
    chunk_shape: chunks,
    data_type: "uint8",
    fill_value: 0,
  });

  // Fill with test data
  const testData = new Uint8Array(shape[0] * shape[1]);
  for (let i = 0; i < testData.length; i++) {
    testData[i] = i % 256;
  }
  await zarr.set(array, null, {
    data: testData,
    shape,
    stride: [shape[1], 1],
  });

  // Create NgffImage
  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  // Create multiscales
  const axes = [
    createAxis("y", "space", "micrometer"),
    createAxis("x", "space", "micrometer"),
  ];
  const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets, "test", "0.5");
  const multiscales = createMultiscales([image], metadata);

  // Write to .ozx file
  const ozxPath = join(OUTPUT_DIR, "test_synthetic.ozx");
  await toNgffZarrOzx(ozxPath, multiscales);

  // Verify file exists
  const fileInfo = await Deno.stat(ozxPath);
  assertEquals(fileInfo.isFile, true);

  // Read the file and verify ZIP structure
  const zipData = await Deno.readFile(ozxPath);

  // Verify version comment
  const version = readOzxVersion(zipData);
  assertEquals(version, "0.5");

  // Verify file ordering
  const files = getZipFileList(zipData);
  assertEquals(files[0], "zarr.json");

  console.log("  Created synthetic .ozx file:", ozxPath);
});

// ============================================================================
// toNgffZarr with .ozx path tests
// ============================================================================

Deno.test("toNgffZarr - detects .ozx path and writes correctly", async () => {
  // Create synthetic test data
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const shape = [8, 8];
  const chunks = [4, 4];

  const array = await zarr.create(root.resolve("data"), {
    shape,
    chunk_shape: chunks,
    data_type: "uint8",
    fill_value: 0,
  });

  const testData = new Uint8Array(shape[0] * shape[1]).fill(42);
  await zarr.set(array, null, {
    data: testData,
    shape,
    stride: [shape[1], 1],
  });

  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const axes = [
    createAxis("y", "space", "micrometer"),
    createAxis("x", "space", "micrometer"),
  ];
  const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets, "test", "0.5");
  const multiscales = createMultiscales([image], metadata);

  // Use toNgffZarr with .ozx path - should auto-detect and use RFC-9
  const ozxPath = join(OUTPUT_DIR, "test_auto_detect.ozx");
  await toNgffZarr(ozxPath, multiscales, { version: "0.5" });

  // Verify file was created as valid .ozx
  const zipData = await Deno.readFile(ozxPath);
  const version = readOzxVersion(zipData);
  assertEquals(version, "0.5");

  console.log("  Auto-detected .ozx path:", ozxPath);
});

Deno.test("toNgffZarr - requires version 0.5 for .ozx", async () => {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const array = await zarr.create(root.resolve("data"), {
    shape: [4, 4],
    chunk_shape: [4, 4],
    data_type: "uint8",
    fill_value: 0,
  });

  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets, "test", "0.5");
  const multiscales = createMultiscales([image], metadata);

  const ozxPath = join(OUTPUT_DIR, "test_version_error.ozx");

  // Should throw error for version 0.4
  await assertRejects(
    async () => {
      await toNgffZarr(ozxPath, multiscales, { version: "0.4" });
    },
    Error,
    "RFC-9 (.ozx) requires OME-Zarr version 0.5",
  );
});

Deno.test("toNgffZarr - allows .ozx without specifying version", async () => {
  // This test verifies that .ozx files work correctly when no version is specified
  // The function should silently default to version 0.5
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const array = await zarr.create(root.resolve("data"), {
    shape: [4, 4],
    chunk_shape: [4, 4],
    data_type: "uint8",
    fill_value: 0,
  });

  const testData = new Uint8Array(16).fill(42);
  await zarr.set(array, null, {
    data: testData,
    shape: [4, 4],
    stride: [4, 1],
  });

  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets, "test", "0.5");
  const multiscales = createMultiscales([image], metadata);

  const ozxPath = join(OUTPUT_DIR, "test_no_version.ozx");

  // Should succeed without specifying version (defaults to 0.5 for .ozx)
  await toNgffZarr(ozxPath, multiscales);

  // Verify the file was created correctly
  const zipData = await Deno.readFile(ozxPath);
  const version = readOzxVersion(zipData);
  assertEquals(version, "0.5");

  console.log("  Created .ozx file without version option:", ozxPath);
});

Deno.test("toNgffZarr - allows .ozx with explicit version 0.5", async () => {
  // This test verifies that explicitly specifying version 0.5 works correctly
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const array = await zarr.create(root.resolve("data"), {
    shape: [4, 4],
    chunk_shape: [4, 4],
    data_type: "uint8",
    fill_value: 0,
  });

  const testData = new Uint8Array(16).fill(42);
  await zarr.set(array, null, {
    data: testData,
    shape: [4, 4],
    stride: [4, 1],
  });

  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets, "test", "0.5");
  const multiscales = createMultiscales([image], metadata);

  const ozxPath = join(OUTPUT_DIR, "test_explicit_version.ozx");

  // Should succeed with explicit version 0.5
  await toNgffZarr(ozxPath, multiscales, { version: "0.5" });

  // Verify the file was created correctly
  const zipData = await Deno.readFile(ozxPath);
  const version = readOzxVersion(zipData);
  assertEquals(version, "0.5");

  console.log("  Created .ozx file with explicit version 0.5:", ozxPath);
});

Deno.test("toNgffZarr - throws error on chunksPerShard for .ozx", async () => {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const array = await zarr.create(root.resolve("data"), {
    shape: [4, 4],
    chunk_shape: [4, 4],
    data_type: "uint8",
    fill_value: 0,
  });

  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const axes = [createAxis("y", "space"), createAxis("x", "space")];
  const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
  const metadata = createMetadata(axes, datasets, "test", "0.5");
  const multiscales = createMultiscales([image], metadata);

  const ozxPath = join(OUTPUT_DIR, "test_sharding_error.ozx");

  // Should throw error for chunksPerShard
  await assertRejects(
    async () => {
      await toNgffZarr(ozxPath, multiscales, {
        version: "0.5",
        chunksPerShard: 2,
      });
    },
    Error,
    "RFC-9 (.ozx) does not support sharding",
  );
});

// ============================================================================
// Integration tests with real Python test data
// ============================================================================

Deno.test("RFC-9 roundtrip with cthead1.png", async () => {
  // Read input image from Python test data
  const inputPath = join(INPUT_DIR, "cthead1.png");
  const itkImage = await readImageNode(inputPath);
  assertExists(itkImage);

  // Convert to NgffImage
  const ngffImage = await itkImageToNgffImage(itkImage, {
    addAnatomicalOrientation: false,
    path: "scale0/image",
  });

  // Generate multiscales with downsampling
  const multiscales = await toMultiscales(ngffImage, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  // Write to .ozx file
  const ozxPath = join(OUTPUT_DIR, "cthead1_rfc9.ozx");
  await toNgffZarrOzx(ozxPath, multiscales);

  // Read the file and verify
  const zipData = await Deno.readFile(ozxPath);

  // Verify version
  const version = readOzxVersion(zipData);
  assertEquals(version, "0.5");

  // Verify file ordering
  const files = getZipFileList(zipData);
  assertEquals(files[0], "zarr.json");

  // All zarr.json files should come before data files
  const zarrJsonFiles = files.filter((f) => f.endsWith("zarr.json"));
  const otherFiles = files.filter((f) => !f.endsWith("zarr.json"));

  if (otherFiles.length > 0) {
    const lastZarrJsonIndex = files.indexOf(
      zarrJsonFiles[zarrJsonFiles.length - 1],
    );
    const firstOtherIndex = files.indexOf(otherFiles[0]);
    assertEquals(
      lastZarrJsonIndex < firstOtherIndex,
      true,
      "All zarr.json files should come before data files",
    );
  }

  // Verify no compression
  for (const file of files.slice(0, 3)) {
    // Check first few files
    const compression = getZipFileCompressionMethod(zipData, file);
    assertEquals(compression, 0, `File ${file} should use ZIP_STORED`);
  }

  console.log("  Created cthead1 .ozx file:", ozxPath);
  console.log("  File count:", files.length);
  console.log("  zarr.json files:", zarrJsonFiles.length);
});

Deno.test("RFC-9 with MR-head.nrrd (3D data)", async () => {
  // Read input image from Python test data
  const inputPath = join(INPUT_DIR, "MR-head.nrrd");
  const itkImage = await readImageNode(inputPath);
  assertExists(itkImage);

  // Convert to NgffImage
  const ngffImage = await itkImageToNgffImage(itkImage, {
    addAnatomicalOrientation: false,
    path: "scale0/image",
  });

  // Generate multiscales
  const multiscales = await toMultiscales(ngffImage, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  // Write to .ozx file
  const ozxPath = join(OUTPUT_DIR, "mr_head_rfc9.ozx");
  await toNgffZarrOzx(ozxPath, multiscales);

  // Read and verify
  const zipData = await Deno.readFile(ozxPath);
  const version = readOzxVersion(zipData);
  assertEquals(version, "0.5");

  const files = getZipFileList(zipData);
  assertEquals(files[0], "zarr.json");

  console.log("  Created MR-head .ozx file:", ozxPath);
  console.log("  File count:", files.length);
});
