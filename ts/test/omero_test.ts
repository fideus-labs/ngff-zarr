// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertExists, assertThrows } from "@std/assert";
import { fromNgffZarr } from "../src/io/from_ngff_zarr.ts";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import * as zarr from "zarrita";
import { NgffImage } from "../src/types/ngff_image.ts";
import type {
  Omero,
  OmeroChannel,
  OmeroWindow,
} from "../src/types/zarr_metadata.ts";
import { validateColor } from "../src/types/zarr_metadata.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";
import { Methods } from "../src/types/methods.ts";

Deno.test("read omero metadata from test dataset", async () => {
  const storePath = new URL(
    "../../py/test/data/input/13457537.zarr",
    import.meta.url,
  );
  const resolvedPath = storePath.pathname.replace(/^\/([A-Za-z]:)/, "$1"); // Fix Windows paths

  const multiscales = await fromNgffZarr(resolvedPath, { validate: true });

  const omero = multiscales.metadata.omero;
  assertExists(omero);
  assertEquals(omero.channels.length, 6);

  // Channel 0
  assertEquals(omero.channels[0].color, "FFFFFF");
  assertEquals(omero.channels[0].window.min, 0.0);
  assertEquals(omero.channels[0].window.max, 65535.0);
  assertEquals(omero.channels[0].window.start, 0.0);
  assertEquals(omero.channels[0].window.end, 1200.0);
  assertEquals(omero.channels[0].label, "cy 1");

  // Channel 1
  assertEquals(omero.channels[1].color, "FFFFFF");
  assertEquals(omero.channels[1].window.min, 0.0);
  assertEquals(omero.channels[1].window.max, 65535.0);
  assertEquals(omero.channels[1].window.start, 0.0);
  assertEquals(omero.channels[1].window.end, 1200.0);
  assertEquals(omero.channels[1].label, "cy 2");

  // Channel 2
  assertEquals(omero.channels[2].color, "FFFFFF");
  assertEquals(omero.channels[2].window.min, 0.0);
  assertEquals(omero.channels[2].window.max, 65535.0);
  assertEquals(omero.channels[2].window.start, 0.0);
  assertEquals(omero.channels[2].window.end, 1200.0);
  assertEquals(omero.channels[2].label, "cy 3");

  // Channel 3
  assertEquals(omero.channels[3].color, "FFFFFF");
  assertEquals(omero.channels[3].window.min, 0.0);
  assertEquals(omero.channels[3].window.max, 65535.0);
  assertEquals(omero.channels[3].window.start, 0.0);
  assertEquals(omero.channels[3].window.end, 1200.0);
  assertEquals(omero.channels[3].label, "cy 4");

  // Channel 4
  assertEquals(omero.channels[4].color, "0000FF");
  assertEquals(omero.channels[4].window.min, 0.0);
  assertEquals(omero.channels[4].window.max, 65535.0);
  assertEquals(omero.channels[4].window.start, 0.0);
  assertEquals(omero.channels[4].window.end, 5000.0);
  assertEquals(omero.channels[4].label, "DAPI");

  // Channel 5
  assertEquals(omero.channels[5].color, "FF0000");
  assertEquals(omero.channels[5].window.min, 0.0);
  assertEquals(omero.channels[5].window.max, 65535.0);
  assertEquals(omero.channels[5].window.start, 0.0);
  assertEquals(omero.channels[5].window.end, 100.0);
  assertEquals(omero.channels[5].label, "Hyb probe");
});

Deno.test("write omero metadata", async () => {
  // Create test data (equivalent to Python's np.random.randint)
  const data = new Uint8Array(2 * 32 * 64 * 64);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.floor(Math.random() * 256);
  }

  // Create NgffImage with test data
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  const zarrArray = await zarr.create(root.resolve("test_image"), {
    shape: [2, 32, 64, 64],
    chunk_shape: [2, 32, 64, 64],
    data_type: "uint8",
  });

  const image = new NgffImage({
    data: zarrArray,
    dims: ["c", "z", "y", "x"],
    scale: { c: 1.0, z: 1.0, y: 1.0, x: 1.0 },
    translation: { c: 0.0, z: 0.0, y: 0.0, x: 0.0 },
    name: "test_image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  // Create metadata and multiscales
  const axes = [
    createAxis("c", "channel"),
    createAxis("z", "space"),
    createAxis("y", "space"),
    createAxis("x", "space"),
  ];

  const datasets = [
    createDataset("0", [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
    createDataset("1", [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]),
    createDataset("2", [1.0, 1.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]),
  ];

  const metadata = createMetadata(axes, datasets, "test_image");

  // Create Omero metadata
  const omero: Omero = {
    channels: [
      {
        color: "008000",
        window: {
          min: 0.0,
          max: 255.0,
          start: 10.0,
          end: 150.0,
        },
        label: "Phalloidin",
      },
      {
        color: "0000FF",
        window: {
          min: 0.0,
          max: 255.0,
          start: 30.0,
          end: 200.0,
        },
        label: "",
      },
    ],
  };

  metadata.omero = omero;

  const multiscales = createMultiscales(
    [image],
    metadata,
    [2, 4],
    Methods.ITKWASM_GAUSSIAN,
  );

  const memoryStore = new Map<string, Uint8Array>();
  await toNgffZarr(memoryStore, multiscales, { version: "0.4" });

  // Test the Omero metadata creation and validation directly
  const readOmero = multiscales.metadata.omero;

  assertExists(readOmero);
  assertEquals(readOmero.channels.length, 2);
  assertEquals(readOmero.channels[0].color, "008000");
  assertEquals(readOmero.channels[0].window.start, 10.0);
  assertEquals(readOmero.channels[0].window.end, 150.0);
  assertEquals(readOmero.channels[0].label, "Phalloidin");
  assertEquals(readOmero.channels[1].color, "0000FF");
  assertEquals(readOmero.channels[1].window.start, 30.0);
  assertEquals(readOmero.channels[1].window.end, 200.0);
  assertEquals(readOmero.channels[1].label, "");
});

Deno.test(
  "write omero metadata with baseline verification example",
  async () => {
    // Create test data with fixed values for reproducible baseline
    const data = new Uint8Array(2 * 32 * 64 * 64);
    for (let i = 0; i < data.length; i++) {
      data[i] = i % 256; // Fixed pattern for reproducible results
    }

    // Create NgffImage with test data
    const store = new Map<string, Uint8Array>();
    const root = zarr.root(store);

    const zarrArray = await zarr.create(root.resolve("test_image"), {
      shape: [2, 32, 64, 64],
      chunk_shape: [2, 32, 64, 64],
      data_type: "uint8",
    });

    const image = new NgffImage({
      data: zarrArray,
      dims: ["c", "z", "y", "x"],
      scale: { c: 1.0, z: 1.0, y: 1.0, x: 1.0 },
      translation: { c: 0.0, z: 0.0, y: 0.0, x: 0.0 },
      name: "test_image",
      axesUnits: undefined,
      computedCallbacks: undefined,
    });

    // Create metadata and multiscales
    const axes = [
      createAxis("c", "channel"),
      createAxis("z", "space"),
      createAxis("y", "space"),
      createAxis("x", "space"),
    ];

    const datasets = [
      createDataset("0", [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
    ];

    const metadata = createMetadata(axes, datasets, "test_image");

    // Create Omero metadata
    const omero: Omero = {
      channels: [
        {
          color: "008000",
          window: {
            min: 0.0,
            max: 255.0,
            start: 10.0,
            end: 150.0,
          },
          label: "Phalloidin",
        },
        {
          color: "0000FF",
          window: {
            min: 0.0,
            max: 255.0,
            start: 30.0,
            end: 200.0,
          },
          label: "DAPI",
        },
      ],
    };

    metadata.omero = omero;

    const multiscales = createMultiscales(
      [image],
      metadata,
      [2], // Simpler for baseline
      Methods.ITKWASM_GAUSSIAN,
    );

    // Verify that we can write and read the data correctly
    const memoryStore = new Map<string, Uint8Array>();
    await toNgffZarr(memoryStore, multiscales, { version: "0.4" });

    const readMultiscales = await fromNgffZarr(memoryStore, { validate: true });
    const readOmero = readMultiscales.metadata.omero;

    assertExists(readOmero);
    assertEquals(readOmero.channels.length, 2);
    assertEquals(readOmero.channels[0].color, "008000");
    assertEquals(readOmero.channels[0].label, "Phalloidin");
    assertEquals(readOmero.channels[1].color, "0000FF");
    assertEquals(readOmero.channels[1].label, "DAPI");
  },
);

Deno.test("validate color function", () => {
  // Test valid colors
  validateColor("1A2B3C");
  validateColor("FFFFFF");
  validateColor("000000");
  validateColor("FF00FF");
  validateColor("123ABC");

  // Test invalid colors
  assertThrows(() => validateColor("ZZZZZZ"), Error, "Invalid color 'ZZZZZZ'");

  assertThrows(() => validateColor("FF00"), Error, "Invalid color 'FF00'");

  assertThrows(
    () => validateColor("FF00000"),
    Error,
    "Invalid color 'FF00000'",
  );

  assertThrows(
    () => validateColor("#FF0000"),
    Error,
    "Invalid color '#FF0000'",
  );
});

Deno.test("create omero channel and window", () => {
  const window: OmeroWindow = {
    min: 0.0,
    max: 255.0,
    start: 10.0,
    end: 200.0,
  };

  const channel: OmeroChannel = {
    color: "FF0000",
    window: window,
    label: "Red Channel",
  };

  assertEquals(channel.color, "FF0000");
  assertEquals(channel.window.min, 0.0);
  assertEquals(channel.window.max, 255.0);
  assertEquals(channel.window.start, 10.0);
  assertEquals(channel.window.end, 200.0);
  assertEquals(channel.label, "Red Channel");

  // Test validation
  validateColor(channel.color);
});

Deno.test("create omero metadata with multiple channels", () => {
  const omero: Omero = {
    channels: [
      {
        color: "FF0000",
        window: { min: 0.0, max: 255.0, start: 0.0, end: 100.0 },
        label: "Red",
      },
      {
        color: "00FF00",
        window: { min: 0.0, max: 255.0, start: 0.0, end: 150.0 },
        label: "Green",
      },
      {
        color: "0000FF",
        window: { min: 0.0, max: 255.0, start: 0.0, end: 200.0 },
        label: "Blue",
      },
    ],
  };

  assertEquals(omero.channels.length, 3);
  assertEquals(omero.channels[0].color, "FF0000");
  assertEquals(omero.channels[0].label, "Red");
  assertEquals(omero.channels[1].color, "00FF00");
  assertEquals(omero.channels[1].label, "Green");
  assertEquals(omero.channels[2].color, "0000FF");
  assertEquals(omero.channels[2].label, "Blue");

  // Validate all colors
  omero.channels.forEach((channel) => validateColor(channel.color));
});
