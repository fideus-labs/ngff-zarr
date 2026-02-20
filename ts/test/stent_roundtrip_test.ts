#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * Round-trip test for stent.ome.zarr dataset
 *
 * This test verifies that:
 * 1. The stent.ome.zarr dataset (OME-Zarr 0.5 format) can be read
 * 2. It can be written to a new store
 * 3. The written data can be read back
 * 4. The data remains consistent through the read-write-read cycle
 */

import { assertEquals, assertExists } from "@std/assert";
import { fromNgffZarr } from "../src/io/from_ngff_zarr.ts";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import * as zarr from "zarrita";

Deno.test("Round-trip: stent.ome.zarr (OME-Zarr 0.5)", async () => {
  // Read the original stent.ome.zarr dataset
  const storePath = new URL(
    "../../py/test/data/input/stent.ome.zarr",
    import.meta.url,
  );
  const resolvedPath = storePath.pathname.replace(/^\/([A-Za-z]:)/, "$1"); // Fix Windows paths
  const version = "0.5";

  console.log(`Reading stent.ome.zarr from ${resolvedPath}`);
  const multiscalesOriginal = await fromNgffZarr(resolvedPath, {
    validate: true,
    version: version,
  });

  // Verify that we got valid multiscales data
  assertExists(multiscalesOriginal, "Original multiscales should exist");
  assertExists(multiscalesOriginal.images, "Original images should exist");
  assertEquals(
    Array.isArray(multiscalesOriginal.images),
    true,
    "Images should be an array",
  );
  assertEquals(
    multiscalesOriginal.images.length > 0,
    true,
    "Should have at least one image",
  );

  // Verify metadata exists and has expected structure
  assertExists(multiscalesOriginal.metadata, "Original metadata should exist");
  assertExists(multiscalesOriginal.metadata.axes, "Original axes should exist");
  assertExists(
    multiscalesOriginal.metadata.datasets,
    "Original datasets should exist",
  );

  // Log information about the loaded data
  const firstImage = multiscalesOriginal.images[0];
  assertExists(firstImage, "First image should exist");
  assertExists(firstImage.data, "First image data should exist");
  assertExists(firstImage.dims, "First image dims should exist");
  assertExists(firstImage.scale, "First image scale should exist");
  assertExists(firstImage.translation, "First image translation should exist");

  console.log(
    `Loaded ${multiscalesOriginal.images.length} scales from stent.ome.zarr`,
  );
  console.log(`First image shape: ${firstImage.data.shape}`);
  console.log(`First image dims: ${firstImage.dims}`);
  console.log(`First image scale:`, firstImage.scale);
  console.log(`First image translation:`, firstImage.translation);
  console.log(
    `Metadata axes: ${
      multiscalesOriginal.metadata.axes.map((a) => a.name).join(", ")
    }`,
  );

  // Write to a memory store
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();
  console.log(`Writing to memory store (version ${version})`);
  await toNgffZarr(memoryStore, multiscalesOriginal, { version });

  // Verify the write was successful by checking keys
  const keys = Array.from(memoryStore.keys());
  assertExists(keys, "Memory store keys should exist");
  assertEquals(
    keys.length > 0,
    true,
    `Memory store should contain data after write, found ${keys.length} keys`,
  );

  console.log(`Memory store contains ${keys.length} keys`);

  // Read back from memory store
  console.log("Reading back from memory store");
  const multiscalesRoundtrip = await fromNgffZarr(memoryStore, {
    validate: true,
    version: version,
  });

  // Verify the roundtrip data structure
  assertExists(multiscalesRoundtrip, "Roundtrip multiscales should exist");
  assertExists(multiscalesRoundtrip.images, "Roundtrip images should exist");
  assertEquals(
    multiscalesRoundtrip.images.length,
    multiscalesOriginal.images.length,
    "Should have same number of scales after roundtrip",
  );

  // Verify metadata consistency
  assertExists(
    multiscalesRoundtrip.metadata,
    "Roundtrip metadata should exist",
  );
  assertEquals(
    multiscalesRoundtrip.metadata.axes.length,
    multiscalesOriginal.metadata.axes.length,
    "Should have same number of axes after roundtrip",
  );
  assertEquals(
    multiscalesRoundtrip.metadata.datasets.length,
    multiscalesOriginal.metadata.datasets.length,
    "Should have same number of datasets after roundtrip",
  );

  // Verify axes names match
  for (let i = 0; i < multiscalesOriginal.metadata.axes.length; i++) {
    assertEquals(
      multiscalesRoundtrip.metadata.axes[i].name,
      multiscalesOriginal.metadata.axes[i].name,
      `Axis ${i} name should match`,
    );
    assertEquals(
      multiscalesRoundtrip.metadata.axes[i].type,
      multiscalesOriginal.metadata.axes[i].type,
      `Axis ${i} type should match`,
    );
  }

  // Verify each scale's properties
  for (let i = 0; i < multiscalesOriginal.images.length; i++) {
    const originalImage = multiscalesOriginal.images[i];
    const roundtripImage = multiscalesRoundtrip.images[i];

    console.log(`Verifying scale ${i}`);

    // Check shape
    assertEquals(
      roundtripImage.data.shape.length,
      originalImage.data.shape.length,
      `Scale ${i} should have same number of dimensions`,
    );
    for (let j = 0; j < originalImage.data.shape.length; j++) {
      assertEquals(
        roundtripImage.data.shape[j],
        originalImage.data.shape[j],
        `Scale ${i} dimension ${j} should match`,
      );
    }

    // Check dims
    assertEquals(
      roundtripImage.dims.length,
      originalImage.dims.length,
      `Scale ${i} should have same number of dim labels`,
    );
    for (let j = 0; j < originalImage.dims.length; j++) {
      assertEquals(
        roundtripImage.dims[j],
        originalImage.dims[j],
        `Scale ${i} dim ${j} should match`,
      );
    }

    // Check scale factors
    for (const dim of originalImage.dims) {
      assertExists(
        roundtripImage.scale[dim],
        `Scale ${i} should have scale factor for dimension ${dim}`,
      );
      assertEquals(
        roundtripImage.scale[dim],
        originalImage.scale[dim],
        `Scale ${i} scale factor for ${dim} should match`,
      );
    }

    // Check translations
    for (const dim of originalImage.dims) {
      assertExists(
        roundtripImage.translation[dim],
        `Scale ${i} should have translation for dimension ${dim}`,
      );
      assertEquals(
        roundtripImage.translation[dim],
        originalImage.translation[dim],
        `Scale ${i} translation for ${dim} should match`,
      );
    }
  }

  // Verify actual data consistency by reading a small slice from each scale
  for (let i = 0; i < Math.min(multiscalesOriginal.images.length, 3); i++) {
    const originalImage = multiscalesOriginal.images[i];
    const roundtripImage = multiscalesRoundtrip.images[i];

    console.log(`Verifying data consistency for scale ${i}`);

    // Read a small slice from the center of each array
    const shape = originalImage.data.shape;
    const sliceIndices = shape.map((s) => Math.floor(s / 2));

    // Read single voxel/pixel
    const originalValue = await zarr.get(
      originalImage.data as zarr.Array<zarr.DataType, zarr.Readable>,
      sliceIndices,
    );
    const roundtripValue = await zarr.get(
      roundtripImage.data as zarr.Array<zarr.DataType, zarr.Readable>,
      sliceIndices,
    );

    assertEquals(
      roundtripValue,
      originalValue,
      `Scale ${i} data at ${sliceIndices} should match after roundtrip`,
    );
  }

  console.log("✓ stent.ome.zarr round-trip test passed");
});
