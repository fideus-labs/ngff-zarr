// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Integration tests for RFC 4 anatomical orientation serialization.
 * Ported from py/test/test_rfc4_integration.py and py/test/test_rfc4_integration_validation.py
 */

import { assertEquals } from "@std/assert";
import * as zarr from "zarrita";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import { fromNgffZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { Multiscales } from "../src/types/multiscales.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import {
  AnatomicalOrientationValues,
  createAnatomicalOrientation,
  LPS,
  RAS,
} from "../src/types/rfc4.ts";
import type { AnatomicalOrientation } from "../src/types/rfc4.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
} from "../src/utils/factory.ts";

// Helper function to create a test zarr array with data
async function createTestZarrArray(
  store: MemoryStore,
  path: string,
  shape: number[],
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const root = zarr.root(store);
  const arrayLocation = root.resolve(path);

  const arr = await zarr.create(arrayLocation, {
    shape,
    data_type: "uint8",
    chunk_shape: shape,
    fill_value: 0,
  });

  // Fill with test data
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = new Uint8Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    data[i] = i % 256;
  }
  await zarr.set(arr, null, {
    data,
    shape,
    stride: calculateStride(shape),
  });

  return arr;
}

function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

// Helper function to create a Multiscales with orientation
async function createMultiscalesWithOrientation(
  orientations: Record<string, AnatomicalOrientation>,
  dims: string[] = ["z", "y", "x"],
  shape: number[] = [10, 20, 30],
): Promise<{ multiscales: Multiscales; store: MemoryStore }> {
  const store: MemoryStore = new Map();

  // Create test array
  const zarrArray = await createTestZarrArray(store, "test_array", shape);

  // Create scale and translation
  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  dims.forEach((dim) => {
    scale[dim] = 1.0;
    translation[dim] = 0.0;
  });

  // Create NgffImage with orientations
  const ngffImage = new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "test",
    axesUnits: undefined,
    axesOrientations: orientations,
    computedCallbacks: undefined,
  });

  // Create axes with orientations
  const axes = dims.map((dim) => {
    if (dim === "x" || dim === "y" || dim === "z") {
      return createAxis(
        dim as "x" | "y" | "z",
        "space",
        "micrometer",
        orientations[dim],
      );
    } else if (dim === "c") {
      return createAxis(dim as "c", "channel");
    } else if (dim === "t") {
      return createAxis(dim as "t", "time", "second");
    } else {
      throw new Error(`Unsupported dimension: ${dim}`);
    }
  });

  // Create dataset
  const dataset = createDataset(
    "scale0",
    dims.map((dim) => scale[dim]),
    dims.map((dim) => translation[dim]),
  );

  // Create metadata
  const metadata = createMetadata(axes, [dataset], "test");

  // Create Multiscales
  const multiscales = new Multiscales({
    images: [ngffImage],
    metadata,
    scaleFactors: undefined,
    method: undefined,
    chunks: undefined,
  });

  return { multiscales, store };
}

// Helper to read zarr attributes
async function readZarrAttrs(
  store: MemoryStore,
): Promise<Record<string, unknown>> {
  const root = await zarr.open(store as zarr.Readable, { kind: "group" });
  return root.attrs as unknown as Record<string, unknown>;
}

// Helper to get multiscales array from attrs (handles v0.5 ome namespace and v0.4 root)
function getMultiscalesArray(
  attrs: Record<string, unknown>,
): unknown[] {
  if ("ome" in attrs) {
    const ome = attrs.ome as Record<string, unknown>;
    return ome.multiscales as unknown[];
  }
  return attrs.multiscales as unknown[];
}

Deno.test("toNgffZarr writes orientation when enabledRfcs includes 4", async () => {
  const orientations: Record<string, AnatomicalOrientation> = {
    x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.PosteriorToAnterior,
    ),
    z: createAnatomicalOrientation(
      AnatomicalOrientationValues.SuperiorToInferior,
    ),
  };

  const { multiscales } = await createMultiscalesWithOrientation(orientations);

  // Write to a new store with RFC 4 enabled
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [4] });

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Find spatial axes and check for orientation
  const spatialAxesWithOrientation = axesMetadata.filter(
    (ax) => ax.orientation,
  );
  assertEquals(spatialAxesWithOrientation.length, 3); // x, y, z should have orientation

  // Check specific orientations
  for (const axis of axesMetadata) {
    if (axis.name === "x") {
      assertEquals("orientation" in axis, true);
      assertEquals(
        (axis.orientation as { type: string; value: string }).type,
        "anatomical",
      );
      assertEquals(
        (axis.orientation as { type: string; value: string }).value,
        "left-to-right",
      );
    } else if (axis.name === "y") {
      assertEquals("orientation" in axis, true);
      assertEquals(
        (axis.orientation as { type: string; value: string }).type,
        "anatomical",
      );
      assertEquals(
        (axis.orientation as { type: string; value: string }).value,
        "posterior-to-anterior",
      );
    } else if (axis.name === "z") {
      assertEquals("orientation" in axis, true);
      assertEquals(
        (axis.orientation as { type: string; value: string }).type,
        "anatomical",
      );
      assertEquals(
        (axis.orientation as { type: string; value: string }).value,
        "superior-to-inferior",
      );
    }
  }
});

Deno.test("toNgffZarr omits orientation when enabledRfcs is undefined", async () => {
  const orientations: Record<string, AnatomicalOrientation> = {
    x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.PosteriorToAnterior,
    ),
    z: createAnatomicalOrientation(
      AnatomicalOrientationValues.SuperiorToInferior,
    ),
  };

  const { multiscales } = await createMultiscalesWithOrientation(orientations);

  // Write to a new store WITHOUT RFC 4 enabled (default)
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {
    // enabledRfcs not specified
  });

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Check that no spatial axes have orientation
  for (const axis of axesMetadata) {
    assertEquals(
      "orientation" in axis,
      false,
      `Axis ${axis.name} should not have orientation when RFC 4 is disabled`,
    );
  }
});

Deno.test("toNgffZarr omits orientation when enabledRfcs is empty", async () => {
  const orientations: Record<string, AnatomicalOrientation> = {
    x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.PosteriorToAnterior,
    ),
    z: createAnatomicalOrientation(
      AnatomicalOrientationValues.SuperiorToInferior,
    ),
  };

  const { multiscales } = await createMultiscalesWithOrientation(orientations);

  // Write to a new store with empty enabledRfcs
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [] });

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Check that no spatial axes have orientation
  for (const axis of axesMetadata) {
    assertEquals(
      "orientation" in axis,
      false,
      `Axis ${axis.name} should not have orientation when RFC 4 is disabled`,
    );
  }
});

Deno.test("toNgffZarr writes orientation when enabledRfcs=[1,2,4,5]", async () => {
  const orientations: Record<string, AnatomicalOrientation> = {
    x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.PosteriorToAnterior,
    ),
    z: createAnatomicalOrientation(
      AnatomicalOrientationValues.SuperiorToInferior,
    ),
  };

  const { multiscales } = await createMultiscalesWithOrientation(orientations);

  // Write to a new store with RFC 4 enabled alongside other RFCs
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [1, 2, 4, 5] });

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Find spatial axes and check for orientation
  const spatialAxesWithOrientation = axesMetadata.filter(
    (ax) => ax.orientation,
  );
  assertEquals(spatialAxesWithOrientation.length, 3); // x, y, z should have orientation
});

// Round-trip integration tests (from py/test/test_rfc4_integration_validation.py)

Deno.test("Round-trip with LPS orientation - write and read back", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(LPS);

  // Write to store with RFC 4 enabled
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [4] });

  // Read back with validation enabled - should pass
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

Deno.test("Round-trip with RAS orientation - write and read back", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(RAS);

  // Write to store with RFC 4 enabled
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [4] });

  // Read back with validation enabled - should pass
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

Deno.test("Round-trip with 5D mixed axes (t,c,z,y,x) and LPS orientation", async () => {
  // Create with 5D dimensions
  const { multiscales } = await createMultiscalesWithOrientation(
    LPS,
    ["t", "c", "z", "y", "x"],
    [3, 2, 5, 10, 15],
  );

  // Write to store with RFC 4 enabled
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [4] });

  // Read back with validation enabled - should pass
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

Deno.test("Round-trip without orientation passes validation", async () => {
  // Create multiscales without orientation
  const store: MemoryStore = new Map();
  const zarrArray = await createTestZarrArray(store, "test_array", [
    10,
    20,
    30,
  ]);

  const dims = ["z", "y", "x"];
  const scale: Record<string, number> = { z: 2.5, y: 1.0, x: 1.0 };
  const translation: Record<string, number> = { z: 0.0, y: 0.0, x: 0.0 };

  const ngffImage = new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "test_no_orientation",
    axesUnits: undefined,
    axesOrientations: undefined, // No orientations
    computedCallbacks: undefined,
  });

  const axes = dims.map((dim) =>
    createAxis(dim as "x" | "y" | "z", "space", "micrometer")
  );
  const dataset = createDataset(
    "scale0",
    dims.map((dim) => scale[dim]),
    dims.map((dim) => translation[dim]),
  );
  const metadata = createMetadata(axes, [dataset], "test_no_orientation");

  const multiscales = new Multiscales({
    images: [ngffImage],
    metadata,
    scaleFactors: undefined,
    method: undefined,
    chunks: undefined,
  });

  // Write to store (no RFC 4 enabled)
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back with validation enabled - should pass (no orientation to validate)
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

// Tests that verify orientations are extracted when reading back

Deno.test("fromNgffZarr extracts LPS orientation into NgffImage.axesOrientations", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(LPS);

  // Write to store with RFC 4 enabled
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [4] });

  // Read back
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack.images.length, 1);

  const image = multiscalesBack.images[0];
  assertEquals(image.axesOrientations !== undefined, true);

  // Check LPS orientations
  assertEquals(
    image.axesOrientations?.x?.value,
    AnatomicalOrientationValues.RightToLeft,
  );
  assertEquals(
    image.axesOrientations?.y?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
  assertEquals(
    image.axesOrientations?.z?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("fromNgffZarr extracts RAS orientation into NgffImage.axesOrientations", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(RAS);

  // Write to store with RFC 4 enabled
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, { enabledRfcs: [4] });

  // Read back
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack.images.length, 1);

  const image = multiscalesBack.images[0];
  assertEquals(image.axesOrientations !== undefined, true);

  // Check RAS orientations
  assertEquals(
    image.axesOrientations?.x?.value,
    AnatomicalOrientationValues.LeftToRight,
  );
  assertEquals(
    image.axesOrientations?.y?.value,
    AnatomicalOrientationValues.PosteriorToAnterior,
  );
  assertEquals(
    image.axesOrientations?.z?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("fromNgffZarr returns undefined axesOrientations when no orientation in metadata", async () => {
  // Create multiscales without orientation
  const store: MemoryStore = new Map();
  const zarrArray = await createTestZarrArray(store, "test_array", [
    10,
    20,
    30,
  ]);

  const dims = ["z", "y", "x"];
  const scale: Record<string, number> = { z: 1.0, y: 1.0, x: 1.0 };
  const translation: Record<string, number> = { z: 0.0, y: 0.0, x: 0.0 };

  const ngffImage = new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "test",
    axesUnits: undefined,
    axesOrientations: undefined,
    computedCallbacks: undefined,
  });

  const axes = dims.map((dim) =>
    createAxis(dim as "x" | "y" | "z", "space", "micrometer")
  );
  const dataset = createDataset(
    "scale0",
    dims.map((dim) => scale[dim]),
    dims.map((dim) => translation[dim]),
  );
  const metadata = createMetadata(axes, [dataset], "test");

  const multiscales = new Multiscales({
    images: [ngffImage],
    metadata,
    scaleFactors: undefined,
    method: undefined,
    chunks: undefined,
  });

  // Write to store without orientation
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: false });
  assertEquals(multiscalesBack.images.length, 1);

  const image = multiscalesBack.images[0];
  assertEquals(image.axesOrientations, undefined);
});
