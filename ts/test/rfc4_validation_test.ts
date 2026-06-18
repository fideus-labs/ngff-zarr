// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Tests for RFC 4 validation in from_ngff_zarr.
 * Ported from py/test/test_rfc4_validation.py
 */

import { assertEquals, assertThrows } from "@std/assert";
import * as zarr from "zarrita";
import {
  hasRfc4OrientationMetadata,
  validateRfc4Orientation,
} from "../src/utils/rfc4_validation.ts";
import { fromNgffZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";

Deno.test("hasRfc4OrientationMetadata - returns false for axes without orientation", () => {
  const axesNoOrientation = [
    { name: "x", type: "space", unit: "micrometer" },
    { name: "y", type: "space", unit: "micrometer" },
    { name: "z", type: "space", unit: "micrometer" },
  ];
  assertEquals(hasRfc4OrientationMetadata(axesNoOrientation), false);
});

Deno.test("hasRfc4OrientationMetadata - returns true when spatial axes have orientation", () => {
  const axesWithOrientation = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "anterior-to-posterior" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
  ];
  assertEquals(hasRfc4OrientationMetadata(axesWithOrientation), true);
});

Deno.test("validateRfc4Orientation - valid LPS orientation passes", () => {
  const axesLps = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "left-to-right" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "posterior-to-anterior" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "superior-to-inferior" },
    },
  ];

  // Should not throw any exception
  validateRfc4Orientation(axesLps);
});

Deno.test("validateRfc4Orientation - subject-local tissue values pass", () => {
  // Layered/polarized-tissue values (ome/ngff#528) must be accepted.
  const axesSubjectLocal = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "superficial-to-deep" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "apical-to-basal" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "apex-to-base" },
    },
  ];

  // Should not throw any exception
  validateRfc4Orientation(axesSubjectLocal);
});

Deno.test("validateRfc4Orientation - mixed axis types (time/channel) pass", () => {
  const axesMixed = [
    { name: "t", type: "time", unit: "second" },
    { name: "c", type: "channel" },
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "anterior-to-posterior" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
  ];

  // Should not throw any exception
  validateRfc4Orientation(axesMixed);
});

Deno.test("validateRfc4Orientation - throws on incomplete orientation", () => {
  const axesIncomplete = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "anterior-to-posterior" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      // Missing orientation
    },
  ];

  assertThrows(
    () => validateRfc4Orientation(axesIncomplete),
    Error,
    "RFC 4 requires that if orientation is defined",
  );
});

Deno.test("validateRfc4Orientation - throws on inconsistent orientation types", () => {
  const axesInconsistent = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "different_type", value: "anterior-to-posterior" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
  ];

  assertThrows(
    () => validateRfc4Orientation(axesInconsistent),
    Error,
    "All spatial axis orientations must have the same type",
  );
});

Deno.test("validateRfc4Orientation - throws on invalid orientation value", () => {
  const axesInvalid = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "invalid-orientation" },
    },
    {
      name: "y",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "anterior-to-posterior" },
    },
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "inferior-to-superior" },
    },
  ];

  assertThrows(
    () => validateRfc4Orientation(axesInvalid),
    Error,
    "Invalid orientation value",
  );
});

// Helper function to create a test zarr store with multiscales metadata
async function createTestStore(
  multiscalesMetadata: Record<string, unknown>,
): Promise<MemoryStore> {
  const store: MemoryStore = new Map();

  // Create root group
  const root = zarr.root(store);
  await zarr.create(root, {
    attributes: {
      multiscales: [multiscalesMetadata],
    },
  });

  // Create a simple zarr array at path "0"
  const arrayLocation = root.resolve("0");
  await zarr.create(arrayLocation, {
    shape: [10, 10, 10],
    data_type: "uint8",
    chunk_shape: [10, 10, 10],
    fill_value: 0,
  });

  return store;
}

Deno.test("fromNgffZarr with valid RFC-4 orientation - validation passes", async () => {
  // Spatial axes must be ordered (z, y, x) per the OME-Zarr v0.4 axis-order
  // rule; each axis keeps its own anatomical orientation.
  const multiscalesMetadata = {
    version: "0.4",
    name: "test",
    axes: [
      {
        name: "z",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "inferior-to-superior" },
      },
      {
        name: "y",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "anterior-to-posterior" },
      },
      {
        name: "x",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "right-to-left" },
      },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          { type: "scale", scale: [1.0, 1.0, 1.0] },
        ],
      },
    ],
  };

  const store = await createTestStore(multiscalesMetadata);

  // Should succeed with valid orientation
  const multiscales = await fromNgffZarr(store, { validate: true });
  assertEquals(multiscales !== undefined, true);
});

Deno.test("fromNgffZarr with invalid RFC-4 orientation - throws error", async () => {
  const multiscalesMetadata = {
    version: "0.4",
    name: "test",
    axes: [
      {
        name: "x",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "right-to-left" },
      },
      {
        name: "y",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "anterior-to-posterior" },
      },
      {
        name: "z",
        type: "space",
        unit: "micrometer",
        // Missing orientation - this should cause validation to fail
      },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          { type: "scale", scale: [1.0, 1.0, 1.0] },
        ],
      },
    ],
  };

  const store = await createTestStore(multiscalesMetadata);

  // Should fail with incomplete orientation
  let errorThrown = false;
  try {
    await fromNgffZarr(store, { validate: true });
  } catch (error) {
    errorThrown = true;
    assertEquals(
      (error as Error).message.includes(
        "RFC 4 requires that if orientation is defined",
      ),
      true,
    );
  }
  assertEquals(errorThrown, true);
});

Deno.test("fromNgffZarr without validation - loads invalid data", async () => {
  const multiscalesMetadata = {
    version: "0.4",
    name: "test",
    axes: [
      {
        name: "x",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "right-to-left" },
      },
      {
        name: "y",
        type: "space",
        unit: "micrometer",
        orientation: { type: "anatomical", value: "anterior-to-posterior" },
      },
      {
        name: "z",
        type: "space",
        unit: "micrometer",
        // Missing orientation - should be fine without validation
      },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          { type: "scale", scale: [1.0, 1.0, 1.0] },
        ],
      },
    ],
  };

  const store = await createTestStore(multiscalesMetadata);

  // Should succeed without validation
  const multiscales = await fromNgffZarr(store, { validate: false });
  assertEquals(multiscales !== undefined, true);
});
