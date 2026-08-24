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

Deno.test("hasRfc4OrientationMetadata - empty or null orientation does not count", () => {
  // Mirrors the Python `if axis['orientation']:` guard: a spatial axis whose
  // orientation is present but falsy ({}, null, "") carries no real
  // orientation metadata and must not trigger RFC-4 validation.
  const emptyObject = [
    { name: "x", type: "space", unit: "micrometer", orientation: {} },
  ];
  assertEquals(hasRfc4OrientationMetadata(emptyObject), false);

  const nullOrientation = [
    { name: "y", type: "space", unit: "micrometer", orientation: null },
  ];
  assertEquals(hasRfc4OrientationMetadata(nullOrientation), false);

  const emptyString = [
    { name: "z", type: "space", unit: "micrometer", orientation: "" },
  ];
  assertEquals(hasRfc4OrientationMetadata(emptyString), false);

  // A non-empty orientation on the same axis shape still counts.
  const populated = [
    {
      name: "x",
      type: "space",
      unit: "micrometer",
      orientation: { type: "anatomical", value: "right-to-left" },
    },
  ];
  assertEquals(hasRfc4OrientationMetadata(populated), true);
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

Deno.test("validateRfc4Orientation - accepts partial orientation", () => {
  // RFC 4 makes orientation optional per spatial axis: no all-or-none rule.
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

  validateRfc4Orientation(axesIncomplete);
});

Deno.test("validateRfc4Orientation - rejects a non-object orientation", () => {
  // Only an absent field, null and an empty object are undefined. Any other
  // non-null value is a malformed use of orientation and must fail: on a spatial
  // axis via the type check, on a non-spatial axis via the spatial-only rule.
  // Mirrors the Python test_validate_rfc4_orientation_non_object_is_rejected.
  assertThrows(
    () =>
      validateRfc4Orientation([
        { name: "x", type: "space", orientation: "nope" },
      ]),
    Error,
    "must be anatomical",
  );
  assertThrows(
    () =>
      validateRfc4Orientation([
        { name: "t", type: "time", orientation: "nope" },
        {
          name: "x",
          type: "space",
          orientation: { type: "anatomical", value: "right-to-left" },
        },
      ]),
    Error,
    "non-space axes",
  );
});

Deno.test("validateRfc4Orientation - throws on a non-anatomical type", () => {
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
    "must be anatomical",
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

Deno.test("fromNgffZarr - an invalid orientation reads below the RFC-4 versions", async () => {
  // RFC 4 orientation is normative from OME-Zarr 0.9.dev1 only
  // (fideus-labs/ngff-zarr#667); this store declares 0.4, so its
  // out-of-vocabulary orientation value is read back without complaint. The
  // version gate lives in the read path alone: the module-level check rejects
  // the value at every version. Mirrors the Python
  // test_from_ngff_zarr_invalid_orientation_reads_below_rfc4.
  const axes = [
    {
      name: "z",
      type: "space",
      unit: "micrometer",
      // Out-of-vocabulary value: rejected by validateRfc4Orientation,
      // tolerated by every pre-RFC-4 read.
      orientation: { type: "anatomical", value: "not-a-direction" },
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
  ];
  const multiscalesMetadata = {
    version: "0.4",
    name: "test",
    axes,
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

  // The module-level check rejects the out-of-vocabulary value...
  const error = assertThrows(() => validateRfc4Orientation(axes), Error);
  assertEquals(error.message.includes("Invalid orientation value"), true);

  // ...and the 0.4 read path does not apply it: RFC 4 gates on 0.9.dev1.
  const multiscales = await fromNgffZarr(store, { validate: true });
  assertEquals(multiscales !== undefined, true);
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
