// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertExists, assertNotEquals } from "@std/assert";
import { fromNgffZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { Multiscales } from "../src/types/multiscales.ts";
import * as zarr from "zarrita";

/**
 * Create synthetic test data for TypeScript testing.
 * Equivalent to Python's lung_series data creation.
 */
async function createTestLungSeriesData(): Promise<NgffImage> {
  // Create a synthetic 3D volume similar to lung_series
  const shape = [80, 256, 256]; // z, y, x
  const dtype = "uint16";
  const chunks = [80, 128, 128];

  // Create the zarr array in memory
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const array = await zarr.create(root.resolve("data"), {
    shape,
    chunk_shape: chunks,
    data_type: dtype as zarr.DataType,
    fill_value: 0,
  });

  // Fill with synthetic data
  const chunkData = new Uint16Array(chunks[0] * chunks[1] * chunks[2]);
  for (let i = 0; i < chunkData.length; i++) {
    chunkData[i] = Math.floor(Math.random() * 1000);
  }

  // Write the data to the zarr array
  await zarr.set(array, null, {
    data: chunkData,
    shape: chunks,
    stride: [chunks[1] * chunks[2], chunks[2], 1],
  });

  // Create NgffImage with proper metadata
  return new NgffImage({
    data: array,
    dims: ["z", "y", "x"],
    scale: { z: 2.5, y: 1.40625, x: 1.40625 },
    translation: { z: 332.5, y: 360.0, x: 0.0 },
    name: "LIDC2",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

/**
 * Create multiscales from NgffImage (equivalent to Python's to_multiscales)
 */
function createMultiscalesFromImage(
  image: NgffImage,
  version = "0.4",
): Multiscales {
  const axes = [
    createAxis("z", "space", "micrometer"),
    createAxis("y", "space", "micrometer"),
    createAxis("x", "space", "micrometer"),
  ];

  const datasets = [
    createDataset("0", [2.5, 1.40625, 1.40625], [332.5, 360.0, 0.0]),
  ];

  const metadata = createMetadata(axes, datasets, image.name, version);

  return createMultiscales([image], metadata);
}

Deno.test("from_ngff_zarr basic round trip", async () => {
  // Create synthetic test data
  const image = await createTestLungSeriesData();
  // Create multiscales with undefined optional fields
  const multiscales = createMultiscalesFromImage(image);

  const testStore: MemoryStore = new Map<string, Uint8Array>();
  const version = "0.4";

  // Write to memory store
  await toNgffZarr(testStore, multiscales, { version });

  // Read back from memory store
  const multiscalesBack = await fromNgffZarr(testStore, { version });

  // Verify basic structure
  assertExists(multiscalesBack);
  assertExists(multiscalesBack.images);
  assertEquals(multiscalesBack.images.length, 1);

  const imageBack = multiscalesBack.images[0];
  assertExists(imageBack);
  assertExists(imageBack.data);
  assertEquals(imageBack.dims, ["z", "y", "x"]);
  assertEquals(imageBack.scale.z, 2.5);
  assertEquals(imageBack.scale.y, 1.40625);
  assertEquals(imageBack.scale.x, 1.40625);
  assertEquals(imageBack.translation.z, 332.5);
  assertEquals(imageBack.translation.y, 360.0);
  assertEquals(imageBack.translation.x, 0.0);
  assertEquals(imageBack.name, "LIDC2");

  // Verify metadata
  assertExists(multiscalesBack.metadata);
  assertEquals(multiscalesBack.metadata.axes.length, 3);
  assertEquals(multiscalesBack.metadata.datasets.length, 1);

  console.log("✓ Basic from_ngff_zarr round trip test passed");
});

Deno.test("omero zarr from_ngff_zarr to_ngff_zarr", async () => {
  const datasetName = "13457537";
  // Use better cross-platform path handling
  const storePath = new URL(
    "../../py/test/data/input/13457537.zarr",
    import.meta.url,
  );
  const resolvedPath = storePath.pathname.replace(/^\/([A-Za-z]:)/, "$1"); // Fix Windows paths
  const version = "0.4";

  const multiscales = await fromNgffZarr(resolvedPath, { validate: true });

  // Verify that we got valid multiscales data
  assertExists(multiscales);
  assertExists(multiscales.images);
  assertEquals(Array.isArray(multiscales.images), true);
  assertEquals(multiscales.images.length > 0, true);

  // Verify metadata exists
  assertExists(multiscales.metadata);
  assertExists(multiscales.metadata.axes);
  assertExists(multiscales.metadata.datasets);

  // Check that we have the expected structure
  const firstImage = multiscales.images[0];
  assertExists(firstImage);
  assertExists(firstImage.data);
  assertExists(firstImage.dims);
  assertExists(firstImage.scale);

  // Log some information about the loaded data for debugging
  console.log(`Loaded ${multiscales.images.length} images from ${datasetName}`);
  console.log(`First image shape: ${firstImage.data.shape}`);
  console.log(`First image dims: ${firstImage.dims}`);
  console.log(
    `Metadata axes: ${multiscales.metadata.axes.map((a) => a.name).join(", ")}`,
  );

  // Test round-trip through memory store
  const memoryStore = new Map<string, Uint8Array>();
  await toNgffZarr(memoryStore, multiscales, { version });

  // Verify the write was successful by checking keys
  const keys = Array.from(memoryStore.keys());
  assertNotEquals(
    keys.length,
    0,
    "Memory store should contain data after write",
  );

  console.log("✓ OMERO zarr round trip test passed");
});

Deno.test("from_ngff_zarr with storage_options", async () => {
  // This test is equivalent to Python's test for zarr v3 storage_options
  // For TypeScript/zarrita, we test that various store configurations work

  const image = await createTestLungSeriesData();
  const multiscales = createMultiscalesFromImage(image);

  const testStore: MemoryStore = new Map<string, Uint8Array>();
  const version = "0.4";

  await toNgffZarr(testStore, multiscales, { version });

  // Test with default options (should work as before)
  const multiscalesBack1 = await fromNgffZarr(testStore, { version });
  assertExists(multiscalesBack1);
  assertEquals(multiscalesBack1.images.length, 1);

  // Test with validation enabled
  const multiscalesBack2 = await fromNgffZarr(testStore, {
    version,
    validate: true,
  });
  assertExists(multiscalesBack2);
  assertEquals(multiscalesBack2.images.length, 1);

  // Test with validation disabled
  const multiscalesBack3 = await fromNgffZarr(testStore, {
    version,
    validate: false,
  });
  assertExists(multiscalesBack3);
  assertEquals(multiscalesBack3.images.length, 1);

  console.log("✓ Storage options test passed");
});

Deno.test("omero metadata backward compatibility", async () => {
  // Test that OMERO metadata with only min/max or only start/end is handled correctly
  // This is equivalent to Python's test_omero_metadata_backward_compatibility

  // Create test store with OMERO metadata using only min/max (old format)
  const store: MemoryStore = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  // Create sample data array
  const shape = [1, 1, 10, 100, 100];
  const dtype = "uint16";
  const chunks = [1, 1, 10, 50, 50];

  const array = await zarr.create(root.resolve("0"), {
    shape,
    chunk_shape: chunks,
    data_type: dtype as zarr.DataType,
    fill_value: 0,
  });

  // Fill with synthetic data
  const sampleData = new Uint16Array(chunks.reduce((a, b) => a * b, 1));
  for (let i = 0; i < sampleData.length; i++) {
    sampleData[i] = Math.floor(Math.random() * 1000);
  }
  await zarr.set(
    array,
    [
      zarr.slice(0, 1),
      zarr.slice(0, 1),
      zarr.slice(0, 10),
      zarr.slice(0, 50),
      zarr.slice(0, 50),
    ],
    {
      data: sampleData,
      shape: chunks,
      stride: [
        chunks[1] * chunks[2] * chunks[3] * chunks[4],
        chunks[2] * chunks[3] * chunks[4],
        chunks[3] * chunks[4],
        chunks[4],
        1,
      ],
    },
  );

  // Create root group with OMERO metadata using old min/max format only
  const _group = await zarr.create(root, {
    attributes: {
      multiscales: [
        {
          version: "0.4",
          axes: [
            { name: "t", type: "time", unit: "second" },
            { name: "c", type: "channel" },
            { name: "z", type: "space", unit: "micrometer" },
            { name: "y", type: "space", unit: "micrometer" },
            { name: "x", type: "space", unit: "micrometer" },
          ],
          datasets: [
            {
              path: "0",
              coordinateTransformations: [
                { type: "scale", scale: [1.0, 1.0, 1.0, 0.5, 0.5] },
                { type: "translation", translation: [0.0, 0.0, 0.0, 0.0, 0.0] },
              ],
            },
          ],
        },
      ],
      omero: {
        channels: [
          {
            active: true,
            color: "FFFFFF",
            label: "channel-0",
            window: {
              min: 0,
              max: 1000,
              // Note: no start/end keys - this is the old format
            },
          },
        ],
        version: "0.4",
      },
    },
  });

  // This should work without raising errors
  const multiscales = await fromNgffZarr(store);

  // Verify the result
  assertExists(multiscales);
  assertEquals(multiscales.images.length, 1);
  assertExists(multiscales.metadata.omero);
  assertEquals(multiscales.metadata.omero.channels.length, 1);

  // Check that the window has both min/max and start/end populated
  const channel = multiscales.metadata.omero.channels[0];
  assertEquals(channel.window.min, 0);
  assertEquals(channel.window.max, 1000);
  // For backward compatibility, min/max should be used as start/end
  assertEquals(channel.window.start, 0);
  assertEquals(channel.window.end, 1000);

  // Test with start/end format only (newer format)
  const store2: MemoryStore = new Map<string, Uint8Array>();
  const root2 = zarr.root(store2);

  const array2 = await zarr.create(root2.resolve("0"), {
    shape,
    chunk_shape: chunks,
    data_type: dtype as zarr.DataType,
    fill_value: 0,
  });
  await zarr.set(
    array2,
    [
      zarr.slice(0, 1),
      zarr.slice(0, 1),
      zarr.slice(0, 10),
      zarr.slice(0, 50),
      zarr.slice(0, 50),
    ],
    {
      data: sampleData,
      shape: chunks,
      stride: [
        chunks[1] * chunks[2] * chunks[3] * chunks[4],
        chunks[2] * chunks[3] * chunks[4],
        chunks[3] * chunks[4],
        chunks[4],
        1,
      ],
    },
  );

  const _group2 = await zarr.create(root2, {
    attributes: {
      multiscales: [
        {
          version: "0.4",
          axes: [
            { name: "t", type: "time", unit: "second" },
            { name: "c", type: "channel" },
            { name: "z", type: "space", unit: "micrometer" },
            { name: "y", type: "space", unit: "micrometer" },
            { name: "x", type: "space", unit: "micrometer" },
          ],
          datasets: [
            {
              path: "0",
              coordinateTransformations: [
                { type: "scale", scale: [1.0, 1.0, 1.0, 0.5, 0.5] },
                { type: "translation", translation: [0.0, 0.0, 0.0, 0.0, 0.0] },
              ],
            },
          ],
        },
      ],
      omero: {
        channels: [
          {
            active: true,
            color: "FFFFFF",
            label: "channel-0",
            window: {
              start: 10,
              end: 900,
              // Note: no min/max keys - this is a hypothetical newer format
            },
          },
        ],
        version: "0.4",
      },
    },
  });

  // This should also work
  const multiscales2 = await fromNgffZarr(store2);

  // Verify the result
  assertExists(multiscales2);
  assertExists(multiscales2.metadata.omero);
  const channel2 = multiscales2.metadata.omero.channels[0];
  // For forward compatibility, start/end should be used as min/max
  assertEquals(channel2.window.start, 10);
  assertEquals(channel2.window.end, 900);
  assertEquals(channel2.window.min, 10);
  assertEquals(channel2.window.max, 900);

  // Test with both formats present (most complete)
  const store3: MemoryStore = new Map<string, Uint8Array>();
  const root3 = zarr.root(store3);

  const array3 = await zarr.create(root3.resolve("0"), {
    shape,
    chunk_shape: chunks,
    data_type: dtype as zarr.DataType,
    fill_value: 0,
  });
  await zarr.set(
    array3,
    [
      zarr.slice(0, 1),
      zarr.slice(0, 1),
      zarr.slice(0, 10),
      zarr.slice(0, 50),
      zarr.slice(0, 50),
    ],
    {
      data: sampleData,
      shape: chunks,
      stride: [
        chunks[1] * chunks[2] * chunks[3] * chunks[4],
        chunks[2] * chunks[3] * chunks[4],
        chunks[3] * chunks[4],
        chunks[4],
        1,
      ],
    },
  );

  const _group3 = await zarr.create(root3, {
    attributes: {
      multiscales: [
        {
          version: "0.4",
          axes: [
            { name: "t", type: "time", unit: "second" },
            { name: "c", type: "channel" },
            { name: "z", type: "space", unit: "micrometer" },
            { name: "y", type: "space", unit: "micrometer" },
            { name: "x", type: "space", unit: "micrometer" },
          ],
          datasets: [
            {
              path: "0",
              coordinateTransformations: [
                { type: "scale", scale: [1.0, 1.0, 1.0, 0.5, 0.5] },
                { type: "translation", translation: [0.0, 0.0, 0.0, 0.0, 0.0] },
              ],
            },
          ],
        },
      ],
      omero: {
        channels: [
          {
            active: true,
            color: "FFFFFF",
            label: "channel-0",
            window: {
              min: 5,
              max: 995,
              start: 15,
              end: 985,
            },
          },
        ],
        version: "0.4",
      },
    },
  });

  // This should preserve both formats
  const multiscales3 = await fromNgffZarr(store3);

  // Verify the result
  assertExists(multiscales3);
  assertExists(multiscales3.metadata.omero);
  const channel3 = multiscales3.metadata.omero.channels[0];
  assertEquals(channel3.window.min, 5);
  assertEquals(channel3.window.max, 995);
  assertEquals(channel3.window.start, 15);
  assertEquals(channel3.window.end, 985);

  console.log("✓ OMERO metadata backward compatibility test passed");
});
