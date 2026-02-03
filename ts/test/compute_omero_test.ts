// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { assertEquals, assertExists, assertRejects } from "@std/assert";
import * as zarr from "zarrita";
import { NgffImage } from "../src/types/ngff_image.ts";
import {
  computeOmeroFromMultiscales,
  computeOmeroFromNgffImage,
  getDefaultColors,
  GLASBEY_COLORS,
} from "../src/utils/compute_omero.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";
import { Methods } from "../src/types/methods.ts";

// Helper function to compute stride from shape
function computeStride(shape: number[]): number[] {
  const stride: number[] = new Array(shape.length);
  let current = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    stride[i] = current;
    current *= shape[i];
  }
  return stride;
}

// Helper function to create a test NgffImage with data
async function createTestImage(
  data: ArrayLike<number>,
  shape: number[],
  dims: string[],
  dtype: zarr.DataType = "float32",
): Promise<NgffImage> {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  const zarrArray = await zarr.create(root.resolve("test"), {
    shape,
    chunk_shape: shape, // Single chunk for simplicity
    data_type: dtype,
  });

  // Write data to the array with proper format
  const typedData = dtype === "uint8"
    ? new Uint8Array(data as number[])
    : new Float32Array(data as number[]);

  await zarr.set(zarrArray, null, {
    data: typedData,
    shape: shape,
    stride: computeStride(shape),
  });

  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  for (const dim of dims) {
    scale[dim] = 1.0;
    translation[dim] = 0.0;
  }

  return new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "test",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

// ============================================================================
// Tests for single-channel images
// ============================================================================

Deno.test("compute basic statistics for single channel", async () => {
  // Create array with values 0-99
  const data = Array.from({ length: 100 }, (_, i) => i);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels.length, 1);
  const channel = omero.channels[0];

  // Check min/max
  assertEquals(channel.window.min, 0);
  assertEquals(channel.window.max, 99);

  // Check quantiles are approximately correct
  assertExists(channel.window.start);
  assertExists(channel.window.end);
  // 2% of 100 values is ~2, 98% is ~97
  assertEquals(channel.window.start! >= 0 && channel.window.start! <= 5, true);
  assertEquals(channel.window.end! >= 94 && channel.window.end! <= 99, true);
});

Deno.test("single channel uses white color", async () => {
  const data = Array(100).fill(1);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels[0].color, "FFFFFF");
});

Deno.test("custom quantiles are respected", async () => {
  const data = Array.from({ length: 100 }, (_, i) => i);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  // Use 10% and 90% quantiles
  const omero = await computeOmeroFromNgffImage(image, {
    quantiles: [0.1, 0.9],
  });

  const channel = omero.channels[0];
  // 10% of 100 values is ~10, 90% is ~90
  assertEquals(
    channel.window.start! >= 5 && channel.window.start! <= 15,
    true,
  );
  assertEquals(channel.window.end! >= 85 && channel.window.end! <= 95, true);
});

Deno.test("custom color is applied", async () => {
  const data = Array(100).fill(1);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image, {
    colors: ["FF0000"],
  });

  assertEquals(omero.channels[0].color, "FF0000");
});

Deno.test("custom label is applied", async () => {
  const data = Array(100).fill(1);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image, {
    labels: ["DAPI"],
  });

  assertEquals(omero.channels[0].label, "DAPI");
});

Deno.test("default label is empty string", async () => {
  const data = Array(100).fill(1);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels[0].label, "");
});

// ============================================================================
// Tests for multi-channel images
// ============================================================================

Deno.test("per-channel statistics are computed", async () => {
  // 3 channels with different values
  // Channel 0: all 0s, Channel 1: all 100s, Channel 2: all 255s
  const ch0 = Array(100).fill(0);
  const ch1 = Array(100).fill(100);
  const ch2 = Array(100).fill(255);
  const data = [...ch0, ...ch1, ...ch2];

  const image = await createTestImage(data, [3, 10, 10], ["c", "y", "x"]);
  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels.length, 3);

  // Channel 0: all zeros
  assertEquals(omero.channels[0].window.min, 0);
  assertEquals(omero.channels[0].window.max, 0);

  // Channel 1: all 100s
  assertEquals(omero.channels[1].window.min, 100);
  assertEquals(omero.channels[1].window.max, 100);

  // Channel 2: all 255s
  assertEquals(omero.channels[2].window.min, 255);
  assertEquals(omero.channels[2].window.max, 255);
});

Deno.test("multi-channel uses glasbey colors", async () => {
  const data = Array(300).fill(1); // 3 channels * 10*10
  const image = await createTestImage(data, [3, 10, 10], ["c", "y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels[0].color, GLASBEY_COLORS[0]); // Red
  assertEquals(omero.channels[1].color, GLASBEY_COLORS[1]); // Green
  assertEquals(omero.channels[2].color, GLASBEY_COLORS[2]); // Blue
});

Deno.test("many channels cycle colors", async () => {
  const nChannels = GLASBEY_COLORS.length + 5;
  const data = Array(nChannels * 100).fill(1);
  const image = await createTestImage(
    data,
    [nChannels, 10, 10],
    ["c", "y", "x"],
  );

  const omero = await computeOmeroFromNgffImage(image);

  // Check that colors cycle
  assertEquals(
    omero.channels[GLASBEY_COLORS.length].color,
    GLASBEY_COLORS[0],
  );
  assertEquals(
    omero.channels[GLASBEY_COLORS.length + 1].color,
    GLASBEY_COLORS[1],
  );
});

Deno.test("custom colors for channels", async () => {
  const data = Array(300).fill(1);
  const image = await createTestImage(data, [3, 10, 10], ["c", "y", "x"]);

  const omero = await computeOmeroFromNgffImage(image, {
    colors: ["AABBCC", "DDEEFF", "112233"],
  });

  assertEquals(omero.channels[0].color, "AABBCC");
  assertEquals(omero.channels[1].color, "DDEEFF");
  assertEquals(omero.channels[2].color, "112233");
});

Deno.test("custom labels for channels", async () => {
  const data = Array(300).fill(1);
  const image = await createTestImage(data, [3, 10, 10], ["c", "y", "x"]);

  const omero = await computeOmeroFromNgffImage(image, {
    labels: ["DAPI", "GFP", "RFP"],
  });

  assertEquals(omero.channels[0].label, "DAPI");
  assertEquals(omero.channels[1].label, "GFP");
  assertEquals(omero.channels[2].label, "RFP");
});

Deno.test("insufficient colors throws error", async () => {
  const data = Array(300).fill(1);
  const image = await createTestImage(data, [3, 10, 10], ["c", "y", "x"]);

  await assertRejects(
    async () => {
      await computeOmeroFromNgffImage(image, {
        colors: ["FF0000", "00FF00"],
      });
    },
    Error,
    "Not enough colors",
  );
});

Deno.test("insufficient labels throws error", async () => {
  const data = Array(300).fill(1);
  const image = await createTestImage(data, [3, 10, 10], ["c", "y", "x"]);

  await assertRejects(
    async () => {
      await computeOmeroFromNgffImage(image, {
        labels: ["DAPI"],
      });
    },
    Error,
    "Not enough labels",
  );
});

// ============================================================================
// Tests for higher-dimensional images
// ============================================================================

Deno.test("3D image without channel dimension", async () => {
  const data = Array.from({ length: 1000 }, (_, i) => i);
  const image = await createTestImage(data, [10, 10, 10], ["z", "y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels.length, 1);
  assertEquals(omero.channels[0].window.min, 0);
  assertEquals(omero.channels[0].window.max, 999);
});

Deno.test("4D image with channel dimension", async () => {
  // Shape: [c, z, y, x] = [2, 5, 10, 10] = 1000 values
  // Channel 0: all 0s, Channel 1: all 100s
  const ch0 = Array(500).fill(0);
  const ch1 = Array(500).fill(100);
  const data = [...ch0, ...ch1];

  const image = await createTestImage(data, [2, 5, 10, 10], [
    "c",
    "z",
    "y",
    "x",
  ]);
  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels.length, 2);
  assertEquals(omero.channels[0].window.min, 0);
  assertEquals(omero.channels[0].window.max, 0);
  assertEquals(omero.channels[1].window.min, 100);
  assertEquals(omero.channels[1].window.max, 100);
});

// ============================================================================
// Tests for special values
// ============================================================================

Deno.test("handles NaN values", async () => {
  const data = [1, 2, NaN, 4, NaN, 6, 7, 8, 9];
  const image = await createTestImage(data, [3, 3], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  // Should ignore NaN values
  assertEquals(omero.channels[0].window.min, 1);
  assertEquals(omero.channels[0].window.max, 9);
});

Deno.test("constant value array", async () => {
  const data = Array(100).fill(42);
  const image = await createTestImage(data, [10, 10], ["y", "x"]);

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels[0].window.min, 42);
  assertEquals(omero.channels[0].window.max, 42);
  assertEquals(omero.channels[0].window.start, 42);
  assertEquals(omero.channels[0].window.end, 42);
});

Deno.test("integer dtype works correctly", async () => {
  const data = Array.from({ length: 256 }, (_, i) => i);
  const image = await createTestImage(data, [16, 16], ["y", "x"], "uint8");

  const omero = await computeOmeroFromNgffImage(image);

  assertEquals(omero.channels[0].window.min, 0);
  assertEquals(omero.channels[0].window.max, 255);
});

// ============================================================================
// Tests for computeOmeroFromMultiscales
// ============================================================================

Deno.test("computeOmeroFromMultiscales uses lowest resolution by default", async () => {
  // Create a simple image
  const data = Array.from({ length: 64 }, (_, i) => i);
  const image = await createTestImage(data, [8, 8], ["y", "x"]);

  const axes = [
    createAxis("y", "space"),
    createAxis("x", "space"),
  ];
  const datasets = [
    createDataset("0", [1.0, 1.0], [0.0, 0.0]),
    createDataset("1", [2.0, 2.0], [0.0, 0.0]),
  ];
  const metadata = createMetadata(axes, datasets, "test");
  const multiscales = createMultiscales(
    [image, image], // Two scales (same image for simplicity)
    metadata,
    [2],
    Methods.ITKWASM_GAUSSIAN,
  );

  const omero = await computeOmeroFromMultiscales(multiscales);

  assertEquals(omero.channels.length, 1);
});

Deno.test("computeOmeroFromMultiscales can use highest resolution", async () => {
  const data = Array.from({ length: 64 }, (_, i) => i);
  const image = await createTestImage(data, [8, 8], ["y", "x"]);

  const axes = [
    createAxis("y", "space"),
    createAxis("x", "space"),
  ];
  const datasets = [
    createDataset("0", [1.0, 1.0], [0.0, 0.0]),
  ];
  const metadata = createMetadata(axes, datasets, "test");
  const multiscales = createMultiscales(
    [image],
    metadata,
    undefined,
    Methods.ITKWASM_GAUSSIAN,
  );

  const omero = await computeOmeroFromMultiscales(multiscales, {
    useLowestResolution: false,
  });

  assertEquals(omero.channels.length, 1);
  assertEquals(omero.channels[0].window.min, 0);
  assertEquals(omero.channels[0].window.max, 63);
});

Deno.test("computeOmeroFromMultiscales passes through options", async () => {
  const data = Array(200).fill(1); // 2 channels * 10*10
  const image = await createTestImage(data, [2, 10, 10], ["c", "y", "x"]);

  const axes = [
    createAxis("c", "channel"),
    createAxis("y", "space"),
    createAxis("x", "space"),
  ];
  const datasets = [
    createDataset("0", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
  ];
  const metadata = createMetadata(axes, datasets, "test");
  const multiscales = createMultiscales(
    [image],
    metadata,
    undefined,
    Methods.ITKWASM_GAUSSIAN,
  );

  const omero = await computeOmeroFromMultiscales(multiscales, {
    quantiles: [0.1, 0.9],
    colors: ["FF0000", "00FF00"],
    labels: ["Red", "Green"],
  });

  assertEquals(omero.channels[0].color, "FF0000");
  assertEquals(omero.channels[0].label, "Red");
  assertEquals(omero.channels[1].color, "00FF00");
  assertEquals(omero.channels[1].label, "Green");
});

Deno.test("computeOmeroFromMultiscales throws error for empty multiscales", async () => {
  const axes = [
    createAxis("y", "space"),
    createAxis("x", "space"),
  ];
  const datasets = [
    createDataset("0", [1.0, 1.0], [0.0, 0.0]),
  ];
  const metadata = createMetadata(axes, datasets, "test");
  const multiscales = createMultiscales(
    [], // Empty images array
    metadata,
    undefined,
    Methods.ITKWASM_GAUSSIAN,
  );

  await assertRejects(
    async () => {
      await computeOmeroFromMultiscales(multiscales);
    },
    Error,
    "no images",
  );
});

// ============================================================================
// Tests for GLASBEY_COLORS constant
// ============================================================================

Deno.test("GLASBEY_COLORS has at least 20 colors", () => {
  assertEquals(GLASBEY_COLORS.length >= 20, true);
});

Deno.test("GLASBEY_COLORS contains valid hex strings", () => {
  const hexPattern = /^[0-9A-Fa-f]{6}$/;
  for (const color of GLASBEY_COLORS) {
    assertEquals(hexPattern.test(color), true, `Invalid color: ${color}`);
  }
});

Deno.test("first GLASBEY_COLORS values match glasbey_hv", () => {
  assertEquals(GLASBEY_COLORS[0], "30A2DA");
  assertEquals(GLASBEY_COLORS[1], "FC4F30");
  assertEquals(GLASBEY_COLORS[2], "E5AE38");
});

// ============================================================================
// Tests for getDefaultColors helper
// ============================================================================

Deno.test("getDefaultColors returns white for single channel", () => {
  const colors = getDefaultColors(1);
  assertEquals(colors.length, 1);
  assertEquals(colors[0], "FFFFFF");
});

Deno.test("getDefaultColors returns glasbey for multiple channels", () => {
  const colors = getDefaultColors(3);
  assertEquals(colors.length, 3);
  assertEquals(colors[0], GLASBEY_COLORS[0]);
  assertEquals(colors[1], GLASBEY_COLORS[1]);
  assertEquals(colors[2], GLASBEY_COLORS[2]);
});

Deno.test("getDefaultColors cycles for many channels", () => {
  const nChannels = GLASBEY_COLORS.length + 3;
  const colors = getDefaultColors(nChannels);
  assertEquals(colors.length, nChannels);
  assertEquals(colors[GLASBEY_COLORS.length], GLASBEY_COLORS[0]);
});
