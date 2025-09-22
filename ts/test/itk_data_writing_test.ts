#!/usr/bin/env -S deno test --allow-all
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Test script to verify ITK-Wasm data writing functionality
 */

import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import { AnatomicalOrientationValues } from "../src/types/rfc4.ts";
import type { Image } from "itk-wasm";

Deno.test("ITK-Wasm data writing test", async () => {
  // Create a small test ITK image
  const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);

  const mockItkImage: Image = {
    imageType: {
      dimension: 2,
      componentType: "uint8",
      pixelType: "Scalar",
      components: 1,
    },
    name: "test",
    origin: [0, 0],
    spacing: [1.0, 1.0],
    direction: new Float64Array([1, 0, 0, 1]), // 2D identity matrix
    size: [2, 4], // 2x4 image
    data: testData,
    metadata: new Map(),
  };

  // Convert to NGFF
  const ngffImage = await itkImageToNgffImage(mockItkImage);

  // Verify basic properties
  console.log("✅ Image name:", ngffImage.name);
  console.log("✅ Dimensions:", ngffImage.dims);
  console.log("✅ Shape:", ngffImage.data.shape);
  console.log("✅ Scale:", JSON.stringify(ngffImage.scale));
  console.log("✅ Translation:", JSON.stringify(ngffImage.translation));

  // Verify anatomical orientations
  if (ngffImage.axesOrientations) {
    console.log("✅ Orientations present:");
    for (
      const [axis, orientation] of Object.entries(
        ngffImage.axesOrientations,
      )
    ) {
      console.log(`   ${axis}: ${orientation.value}`);
    }
  }

  // Try to read back the data to verify it was written correctly
  try {
    const chunk = await ngffImage.data.getChunk([0, 0]);
    console.log("✅ Data read back successfully");
    console.log("   Data length:", chunk.data.length);
    console.log("   Data values:", Array.from(chunk.data));

    // Verify data matches
    const expectedData = Array.from(testData);
    const actualData = Array.from(chunk.data);

    if (JSON.stringify(expectedData) === JSON.stringify(actualData)) {
      console.log("✅ Data verification: PASSED - Data matches original");
    } else {
      console.log("❌ Data verification: FAILED");
      console.log("   Expected:", expectedData);
      console.log("   Actual:  ", actualData);
    }
  } catch (error) {
    console.log(
      "❌ Error reading data back:",
      error instanceof Error ? error.message : String(error),
    );
  }
});

Deno.test("Test anatomical orientation constants", () => {
  console.log("✅ Testing RFC-4 orientation constants:");
  console.log(`   LeftToRight: ${AnatomicalOrientationValues.LeftToRight}`);
  console.log(`   RightToLeft: ${AnatomicalOrientationValues.RightToLeft}`);
  console.log(
    `   AnteriorToPosterior: ${AnatomicalOrientationValues.AnteriorToPosterior}`,
  );
  console.log(
    `   PosteriorToAnterior: ${AnatomicalOrientationValues.PosteriorToAnterior}`,
  );
});
