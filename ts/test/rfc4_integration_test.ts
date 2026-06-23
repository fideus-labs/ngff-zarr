// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Integration tests for RFC 4 anatomical orientation serialization.
 * Ported from py/test/test_rfc4_integration.py and py/test/test_rfc4_integration_validation.py
 */

import { assertEquals } from "@std/assert";
import * as zarr from "zarrita";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import { toOmeZarr as toOmeZarrBrowser } from "../src/io/to_ngff_zarr-browser.ts";
import { fromNgffZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { NgffMultiscales } from "../src/types/multiscales.ts";
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

// Helper function to create a NgffMultiscales with orientation
async function createMultiscalesWithOrientation(
  orientations: Record<string, AnatomicalOrientation>,
  dims: string[] = ["z", "y", "x"],
  shape: number[] = [10, 20, 30],
): Promise<{ multiscales: NgffMultiscales; store: MemoryStore }> {
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

  // Create NgffMultiscales
  const multiscales = new NgffMultiscales({
    images: [ngffImage],
    metadata,
    scaleFactors: undefined,
    method: undefined,
    chunks: undefined,
  });

  return { multiscales, store };
}

// Helper function to create a NgffMultiscales without any orientation
async function createMultiscalesWithoutOrientation(
  dims: string[] = ["z", "y", "x"],
  shape: number[] = [10, 20, 30],
): Promise<{ multiscales: NgffMultiscales; store: MemoryStore }> {
  const store: MemoryStore = new Map();
  const zarrArray = await createTestZarrArray(store, "test_array", shape);

  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  dims.forEach((dim) => {
    scale[dim] = 1.0;
    translation[dim] = 0.0;
  });

  const ngffImage = new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "test_no_orientation",
    axesUnits: undefined,
    axesOrientations: undefined,
    computedCallbacks: undefined,
  });

  const axes = dims.map((dim) => {
    if (dim === "x" || dim === "y" || dim === "z") {
      return createAxis(dim as "x" | "y" | "z", "space", "micrometer");
    } else if (dim === "c") {
      return createAxis(dim as "c", "channel");
    } else if (dim === "t") {
      return createAxis(dim as "t", "time", "second");
    } else {
      throw new Error(`Unsupported dimension: ${dim}`);
    }
  });

  const dataset = createDataset(
    "scale0",
    dims.map((dim) => scale[dim]),
    dims.map((dim) => translation[dim]),
  );
  const metadata = createMetadata(axes, [dataset], "test_no_orientation");

  const multiscales = new NgffMultiscales({
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
function getNgffMultiscalesArray(
  attrs: Record<string, unknown>,
): unknown[] {
  if ("ome" in attrs) {
    const ome = attrs.ome as Record<string, unknown>;
    return ome.multiscales as unknown[];
  }
  return attrs.multiscales as unknown[];
}

Deno.test("toNgffZarr writes orientation automatically when present", async () => {
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

  // No opt-in flag: orientation is written because it is present
  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getNgffMultiscalesArray(attrs);
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

Deno.test("toNgffZarr omits orientation when none is present", async () => {
  const { multiscales } = await createMultiscalesWithoutOrientation();

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getNgffMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Check that no spatial axes have orientation
  for (const axis of axesMetadata) {
    assertEquals(
      "orientation" in axis,
      false,
      `Axis ${axis.name} should not have orientation when none was specified`,
    );
  }
});

// Round-trip integration tests (from py/test/test_rfc4_integration_validation.py)

Deno.test("Round-trip with LPS orientation - write and read back", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(LPS);

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back with validation enabled - should pass
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

Deno.test("Round-trip with RAS orientation - write and read back", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(RAS);

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back with validation enabled - should pass
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

Deno.test("fromNgffZarr preserves RFC 4 orientation on parsed metadata.axes", async () => {
  // Regression for the read path dropping axisObj.orientation when building
  // metadata.axes: that silently turned the strict structural pass's
  // validateAxisOrientation rule into a no-op and lost orientation that the
  // Python port retains via _filter_axis_dict. Reading back must surface
  // orientation on the spatial axes as the raw { type, value } record.
  const { multiscales } = await createMultiscalesWithOrientation(LPS);

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  const axisByName = new Map(
    multiscalesBack.metadata.axes.map((ax) => [ax.name, ax]),
  );

  for (const [name, expected] of Object.entries(LPS)) {
    const orientation = axisByName.get(name as "x" | "y" | "z")
      ?.orientation as { type: string; value: string } | undefined;
    assertEquals(orientation?.type, "anatomical");
    assertEquals(orientation?.value, expected.value);
  }
});

Deno.test("Round-trip with 5D mixed axes (t,c,z,y,x) and LPS orientation", async () => {
  // Create with 5D dimensions
  const { multiscales } = await createMultiscalesWithOrientation(
    LPS,
    ["t", "c", "z", "y", "x"],
    [3, 2, 5, 10, 15],
  );

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back with validation enabled - should pass
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: true });
  assertEquals(multiscalesBack !== undefined, true);
  assertEquals(multiscalesBack.images.length, 1);
});

Deno.test("Round-trip without orientation passes validation", async () => {
  const { multiscales } = await createMultiscalesWithoutOrientation();

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

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

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

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

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
  const { multiscales } = await createMultiscalesWithoutOrientation();

  const outputStore: MemoryStore = new Map();
  await toNgffZarr(outputStore, multiscales, {});

  // Read back
  const multiscalesBack = await fromNgffZarr(outputStore, { validate: false });
  assertEquals(multiscalesBack.images.length, 1);

  const image = multiscalesBack.images[0];
  assertEquals(image.axesOrientations, undefined);
});

// Browser parity tests (issue #481): the browser toOmeZarr must serialize
// orientation identically to the Node implementation -- written automatically
// whenever it is present.

Deno.test("toOmeZarr (browser) writes orientation automatically when present", async () => {
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

  const outputStore: MemoryStore = new Map();
  await toOmeZarrBrowser(outputStore, multiscales, {});

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getNgffMultiscalesArray(attrs);
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
      assertEquals(
        (axis.orientation as { type: string; value: string }).value,
        "left-to-right",
      );
    } else if (axis.name === "y") {
      assertEquals(
        (axis.orientation as { type: string; value: string }).value,
        "posterior-to-anterior",
      );
    } else if (axis.name === "z") {
      assertEquals(
        (axis.orientation as { type: string; value: string }).value,
        "superior-to-inferior",
      );
    }
  }
});

Deno.test("toOmeZarr (browser) omits orientation when none is present", async () => {
  const { multiscales } = await createMultiscalesWithoutOrientation();

  const outputStore: MemoryStore = new Map();
  await toOmeZarrBrowser(outputStore, multiscales, {});

  // Read back and check metadata
  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getNgffMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Check that no spatial axes have orientation
  for (const axis of axesMetadata) {
    assertEquals(
      "orientation" in axis,
      false,
      `Axis ${axis.name} should not have orientation when none was specified`,
    );
  }
});

Deno.test("toOmeZarr (browser) preserves axis name, type, and unit", async () => {
  // Guards the switch from raw metadata.axes to processAxes output:
  // the standard OME-Zarr axis fields must survive, and orientation present on
  // the source axes is written through.
  const { multiscales } = await createMultiscalesWithOrientation(LPS);

  const outputStore: MemoryStore = new Map();
  await toOmeZarrBrowser(outputStore, multiscales, {});

  const attrs = await readZarrAttrs(outputStore);
  const multiscalesArray = getNgffMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // Default dims are z, y, x — all spatial axes carrying the micrometer unit
  assertEquals(
    axesMetadata.map((ax) => ax.name),
    ["z", "y", "x"],
  );
  for (const axis of axesMetadata) {
    assertEquals(axis.type, "space");
    assertEquals(axis.unit, "micrometer");
    // Orientation is present because the source axes carry it
    assertEquals("orientation" in axis, true);
  }
});

Deno.test("toOmeZarr (browser) writes orientation under ome namespace for version 0.5", async () => {
  const { multiscales } = await createMultiscalesWithOrientation(LPS);

  const outputStore: MemoryStore = new Map();
  await toOmeZarrBrowser(outputStore, multiscales, {
    version: "0.5",
  });

  // Version 0.5 nests multiscales metadata under the "ome" attribute
  const attrs = await readZarrAttrs(outputStore);
  assertEquals("ome" in attrs, true);

  const multiscalesArray = getNgffMultiscalesArray(attrs);
  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
  const axesMetadata = multiscalesMetadata.axes as Array<
    Record<string, unknown>
  >;

  // All three spatial axes should carry orientation through the v0.5 path
  const spatialAxesWithOrientation = axesMetadata.filter(
    (ax) => ax.orientation,
  );
  assertEquals(spatialAxesWithOrientation.length, 3);

  // LPS x axis increases right-to-left
  const xAxis = axesMetadata.find((ax) => ax.name === "x");
  assertEquals(
    (xAxis?.orientation as { type: string; value: string }).value,
    "right-to-left",
  );
});
