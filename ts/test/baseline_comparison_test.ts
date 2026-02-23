#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * Baseline comparison tests
 *
 * These tests:
 * 1. Read input images from Python test data directory
 * 2. Generate multiscales using TypeScript implementation
 * 3. Write to OME-Zarr format
 * 4. Read back the OME-Zarr and extract first/second scale images
 * 5. Compare against Python baseline zarr stores using @itk-wasm/compare-images
 */

import { assertEquals, assertExists } from "@std/assert";
import { join } from "@std/path";
import { readImageNode, writeImageNode } from "@itk-wasm/image-io";
import { compareImagesNode } from "@itk-wasm/compare-images";
import type { Image as ItkWasmImage } from "itk-wasm";

import { Methods } from "../src/types/methods.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import { fromNgffZarr } from "../src/io/from_ngff_zarr.ts";
import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import { ngffImageToItkImage } from "../src/io/ngff_image_to_itk_image.ts";

// Path to Python test data directory
const TEST_DATA_DIR = join(Deno.cwd(), "..", "py", "test", "data");
const INPUT_DIR = join(TEST_DATA_DIR, "input");
const BASELINE_DIR = join(TEST_DATA_DIR, "baseline");
const OUTPUT_DIR = join(Deno.cwd(), "test", "output");

// Ensure output directory exists
await Deno.mkdir(OUTPUT_DIR, { recursive: true });

/**
 * Helper to read an image from the baseline zarr store at a specific scale
 */
async function readBaselineScaleImage(
  datasetName: string,
  baselineName: string,
  scale: number,
): Promise<ItkWasmImage> {
  const baselinePath = join(BASELINE_DIR, datasetName, baselineName);

  // Read the baseline zarr store - fromNgffZarr will auto-detect and use FileSystemStore
  const multiscales = await fromNgffZarr(baselinePath);

  // Get the image at the specified scale
  const ngffImage = multiscales.images[scale];
  assertExists(ngffImage, `Scale ${scale} not found in baseline`);

  // Convert to ITK-Wasm image for comparison
  return await ngffImageToItkImage(ngffImage);
}

/**
 * Helper to write an ITK-Wasm image to a .nrrd file
 */
async function writeScaleImage(
  image: ItkWasmImage,
  testName: string,
  scale: number,
  suffix: string = "test",
): Promise<void> {
  const filename = `${testName}_s${scale}_${suffix}.nrrd`;
  const outputPath = join(OUTPUT_DIR, filename);
  await writeImageNode(image, outputPath);
  console.log(`  Wrote ${filename}`);
}

/**
 * Helper to compare two ITK-Wasm images with tolerance
 */
async function compareImages(
  testImage: ItkWasmImage,
  baselineImage: ItkWasmImage,
  testName: string,
): Promise<void> {
  try {
    const result = await compareImagesNode(testImage, {
      baselineImages: [baselineImage],
      differenceThreshold: 0.0,
      radiusTolerance: 0,
      numberOfPixelsTolerance: 0,
      ignoreBoundaryPixels: false,
    });

    // Check if images match
    const metrics = result.metrics as {
      almostEqual?: boolean;
      numberOfPixelsWithDifferences?: number;
    };

    if (!metrics.almostEqual) {
      console.error(
        `❌ ${testName} failed: ${metrics.numberOfPixelsWithDifferences} pixels differ`,
      );
    }

    assertEquals(
      metrics.almostEqual,
      true,
      `Images should match for ${testName}`,
    );
    assertEquals(
      metrics.numberOfPixelsWithDifferences,
      0,
      `No pixels should differ for ${testName}`,
    );
  } catch (error) {
    console.error(`Error comparing images for ${testName}:`, error);
    console.error("Test image info:", {
      size: testImage.size,
      spacing: testImage.spacing,
      origin: testImage.origin,
      dataLength: testImage.data?.length,
    });
    console.error("Baseline image info:", {
      size: baselineImage.size,
      spacing: baselineImage.spacing,
      origin: baselineImage.origin,
      dataLength: baselineImage.data?.length,
    });
    throw error;
  }
}

Deno.test("cthead1 - ITKWASM_GAUSSIAN scale factors [2, 4]", async () => {
  // Read input image
  const inputPath = join(INPUT_DIR, "cthead1.png");
  const itkImage = await readImageNode(inputPath);
  assertExists(itkImage);
  const ngffImage = await itkImageToNgffImage(itkImage, {
    addAnatomicalOrientation: false,
    path: "scale0/",
  });

  // Generate multiscales
  const multiscales = await toMultiscales(ngffImage, {
    scaleFactors: [2, 4],
    method: Methods.ITKWASM_GAUSSIAN,
  });
  console.log("Generated multiscales with ITKWASM_GAUSSIAN");
  console.log(multiscales);

  // Write to zarr (both in-memory and to filesystem)
  const testStore = new Map();
  await toNgffZarr(testStore, multiscales);

  // Also write to filesystem for inspection
  const outputZarrPath = join(OUTPUT_DIR, "cthead1_gaussian_2_4.zarr");
  await toNgffZarr(outputZarrPath, multiscales);
  console.log(`  Wrote zarr to ${outputZarrPath}`);

  // Read back from zarr
  const readMultiscales = await fromNgffZarr(testStore);

  // Convert first scale (s0 - original) to ITK image
  const testImageS0 = await ngffImageToItkImage(readMultiscales.images[0]);
  await writeScaleImage(testImageS0, "cthead1_gaussian", 0, "test");
  const baselineImageS0 = await readBaselineScaleImage(
    "cthead1",
    "2_4/ITKWASM_GAUSSIAN.zarr",
    0,
  );
  await writeScaleImage(baselineImageS0, "cthead1_gaussian", 0, "baseline");
  await compareImages(testImageS0, baselineImageS0, "cthead1 s0 (original)");

  // Convert second scale (s1) to ITK image
  const testImageS1 = await ngffImageToItkImage(readMultiscales.images[1]);
  await writeScaleImage(testImageS1, "cthead1_gaussian", 1, "test");
  const baselineImageS1 = await readBaselineScaleImage(
    "cthead1",
    "2_4/ITKWASM_GAUSSIAN.zarr",
    1,
  );
  await writeScaleImage(baselineImageS1, "cthead1_gaussian", 1, "baseline");
  await compareImages(
    testImageS1,
    baselineImageS1,
    "cthead1 s1 (2x downsample)",
  );

  // Convert third scale (s2) to ITK image
  const testImageS2 = await ngffImageToItkImage(readMultiscales.images[2]);
  await writeScaleImage(testImageS2, "cthead1_gaussian", 2, "test");
  const baselineImageS2 = await readBaselineScaleImage(
    "cthead1",
    "2_4/ITKWASM_GAUSSIAN.zarr",
    2,
  );
  await writeScaleImage(baselineImageS2, "cthead1_gaussian", 2, "baseline");
  await compareImages(
    testImageS2,
    baselineImageS2,
    "cthead1 s2 (4x downsample)",
  );
});

Deno.test("cthead1 - ITKWASM_BIN_SHRINK scale factors [2, 4]", async () => {
  // Read input image
  const inputPath = join(INPUT_DIR, "cthead1.png");
  const itkImage = await readImageNode(inputPath);
  assertExists(itkImage);
  const ngffImage = await itkImageToNgffImage(itkImage);

  // Generate multiscales
  const multiscales = await toMultiscales(ngffImage, {
    scaleFactors: [2, 4],
    method: Methods.ITKWASM_BIN_SHRINK,
  });

  // Write to zarr (both in-memory and to filesystem)
  const testStore = new Map();
  await toNgffZarr(testStore, multiscales);

  // Also write to filesystem for inspection
  const outputZarrPath = join(OUTPUT_DIR, "cthead1_bin_shrink_2_4.zarr");
  await toNgffZarr(outputZarrPath, multiscales);
  console.log(`  Wrote zarr to ${outputZarrPath}`);

  // Read back from zarr
  const readMultiscales = await fromNgffZarr(testStore);

  // Convert first scale to ITK image
  const testImageS0 = await ngffImageToItkImage(readMultiscales.images[0]);
  await writeScaleImage(testImageS0, "cthead1_bin_shrink", 0, "test");
  const baselineImageS0 = await readBaselineScaleImage(
    "cthead1",
    "2_4/ITKWASM_BIN_SHRINK.zarr",
    0,
  );
  await writeScaleImage(baselineImageS0, "cthead1_bin_shrink", 0, "baseline");
  await compareImages(testImageS0, baselineImageS0, "cthead1 bin_shrink s0");

  // Convert second scale to ITK image
  const testImageS1 = await ngffImageToItkImage(readMultiscales.images[1]);
  await writeScaleImage(testImageS1, "cthead1_bin_shrink", 1, "test");
  const baselineImageS1 = await readBaselineScaleImage(
    "cthead1",
    "2_4/ITKWASM_BIN_SHRINK.zarr",
    1,
  );
  await writeScaleImage(baselineImageS1, "cthead1_bin_shrink", 1, "baseline");
  await compareImages(testImageS1, baselineImageS1, "cthead1 bin_shrink s1");
});

Deno.test(
  "2th_cthead1 - ITKWASM_LABEL_IMAGE scale factors [2, 4]",
  async () => {
    // Read input image
    const inputPath = join(INPUT_DIR, "2th_cthead1.png");
    const itkImage = await readImageNode(inputPath);
    assertExists(itkImage);
    const ngffImage = await itkImageToNgffImage(itkImage);

    // Generate multiscales
    const multiscales = await toMultiscales(ngffImage, {
      scaleFactors: [2, 4],
      method: Methods.ITKWASM_LABEL_IMAGE,
    });

    // Write to zarr (both in-memory and to filesystem)
    const testStore = new Map();
    await toNgffZarr(testStore, multiscales);

    // Also write to filesystem for inspection
    const outputZarrPath = join(OUTPUT_DIR, "2th_cthead1_label_image_2_4.zarr");
    await toNgffZarr(outputZarrPath, multiscales);
    console.log(`  Wrote zarr to ${outputZarrPath}`);

    // Read back from zarr
    const readMultiscales = await fromNgffZarr(testStore);

    // Convert first scale to ITK image
    const testImageS0 = await ngffImageToItkImage(readMultiscales.images[0]);
    await writeScaleImage(testImageS0, "2th_cthead1_label_image", 0, "test");
    const baselineImageS0 = await readBaselineScaleImage(
      "2th_cthead1",
      "2_4/ITKWASM_LABEL_IMAGE.zarr",
      0,
    );
    await writeScaleImage(
      baselineImageS0,
      "2th_cthead1_label_image",
      0,
      "baseline",
    );
    await compareImages(
      testImageS0,
      baselineImageS0,
      "2th_cthead1 label_image s0",
    );

    // Convert second scale to ITK image
    const testImageS1 = await ngffImageToItkImage(readMultiscales.images[1]);
    await writeScaleImage(testImageS1, "2th_cthead1_label_image", 1, "test");
    const baselineImageS1 = await readBaselineScaleImage(
      "2th_cthead1",
      "2_4/ITKWASM_LABEL_IMAGE.zarr",
      1,
    );
    await writeScaleImage(
      baselineImageS1,
      "2th_cthead1_label_image",
      1,
      "baseline",
    );
    await compareImages(
      testImageS1,
      baselineImageS1,
      "2th_cthead1 label_image s1",
    );
  },
);

Deno.test("MR-head - ITKWASM_GAUSSIAN scale factors [2, 3, 4]", async () => {
  // Read input image
  const inputPath = join(INPUT_DIR, "MR-head.nrrd");
  const itkImage = await readImageNode(inputPath);
  assertExists(itkImage);
  const ngffImage = await itkImageToNgffImage(itkImage);

  // Generate multiscales
  const multiscales = await toMultiscales(ngffImage, {
    scaleFactors: [2, 3, 4],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  // Write to zarr (both in-memory and to filesystem)
  const testStore = new Map();
  await toNgffZarr(testStore, multiscales);

  // Also write to filesystem for inspection
  const outputZarrPath = join(OUTPUT_DIR, "mr_head_gaussian_2_3_4.zarr");
  await toNgffZarr(outputZarrPath, multiscales);
  console.log(`  Wrote zarr to ${outputZarrPath}`);

  // Read back from zarr
  const readMultiscales = await fromNgffZarr(testStore);

  // Convert first three scales to ITK images and compare
  const testImageS0 = await ngffImageToItkImage(readMultiscales.images[0]);
  await writeScaleImage(testImageS0, "mr_head_gaussian", 0, "test");
  const baselineImageS0 = await readBaselineScaleImage(
    "MR-head",
    "2_3_4/ITKWASM_GAUSSIAN.zarr",
    0,
  );
  await writeScaleImage(baselineImageS0, "mr_head_gaussian", 0, "baseline");
  await compareImages(testImageS0, baselineImageS0, "MR-head s0 (original)");

  const testImageS1 = await ngffImageToItkImage(readMultiscales.images[1]);
  await writeScaleImage(testImageS1, "mr_head_gaussian", 1, "test");
  const baselineImageS1 = await readBaselineScaleImage(
    "MR-head",
    "2_3_4/ITKWASM_GAUSSIAN.zarr",
    1,
  );
  await writeScaleImage(baselineImageS1, "mr_head_gaussian", 1, "baseline");
  await compareImages(
    testImageS1,
    baselineImageS1,
    "MR-head s1 (2x downsample)",
  );

  const testImageS2 = await ngffImageToItkImage(readMultiscales.images[2]);
  await writeScaleImage(testImageS2, "mr_head_gaussian", 2, "test");
  const baselineImageS2 = await readBaselineScaleImage(
    "MR-head",
    "2_3_4/ITKWASM_GAUSSIAN.zarr",
    2,
  );
  await writeScaleImage(baselineImageS2, "mr_head_gaussian", 2, "baseline");
  await compareImages(
    testImageS2,
    baselineImageS2,
    "MR-head s2 (3x downsample)",
  );

  const testImageS3 = await ngffImageToItkImage(readMultiscales.images[3]);
  await writeScaleImage(testImageS3, "mr_head_gaussian", 3, "test");
  const baselineImageS3 = await readBaselineScaleImage(
    "MR-head",
    "2_3_4/ITKWASM_GAUSSIAN.zarr",
    3,
  );
  await writeScaleImage(baselineImageS3, "mr_head_gaussian", 3, "baseline");
  await compareImages(
    testImageS3,
    baselineImageS3,
    "MR-head s3 (4x downsample)",
  );
});
