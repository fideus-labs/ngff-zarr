// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for the RFC 4 anatomical-orientation rule in
 * `../src/utils/structural_validation.ts`.
 *
 * Pass/fail tests for {@link validateAxisOrientation}, the wrapper that surfaces
 * the package's RFC 4 orientation checks through the unified
 * {@link ValidationError} channel:
 *
 * - no spatial axis carries orientation -> no-op;
 * - a fully specified, consistent orientation passes for both the serialized
 *   `{ type, value }` form (the image read path) and the
 *   {@link AnatomicalOrientationValues} enum form (the write path);
 * - mixed orientation `type` -> {@link SpecRule.AxisOrientationConsistentType};
 * - orientation on some but not all spatial axes ->
 *   {@link SpecRule.AxisOrientationCompleteness}.
 *
 * Each failing case asserts the expected {@link SpecRule} and the
 * `multiscales[0].axes` location. The identifiers and location strings are kept
 * byte-for-byte identical to the package's Python implementation; this suite
 * mirrors `py/test/test_structural_validation_orientation.py`.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  SpecRule,
  validateAxisOrientation,
  ValidationError,
} from "../src/utils/structural_validation.ts";
import { validateRfc4Orientation } from "../src/utils/rfc4_validation.ts";
import {
  type Axis,
  createScale,
  type Metadata,
} from "../src/types/zarr_metadata.ts";
import { AnatomicalOrientationValues } from "../src/types/rfc4.ts";

/** Canonical (z, y, x) LPS orientation values, in axis order. */
const LPS_VALUES = [
  "inferior-to-superior",
  "anterior-to-posterior",
  "right-to-left",
] as const;

/**
 * Wrap three spatial axes carrying the given orientations in an otherwise-valid
 * single-level metadata.
 *
 * {@link validateAxisOrientation} inspects only the axes, so the dataset's
 * `scale` is a placeholder sized to the axis count. An `undefined` entry leaves
 * that axis without an orientation.
 */
function metadataWithOrientations(
  orientations: ReadonlyArray<Axis["orientation"]>,
): Metadata {
  const names = ["z", "y", "x"] as const;
  return {
    axes: names.map((name, i): Axis => {
      const orientation = orientations[i];
      return orientation === undefined
        ? { name, type: "space", unit: undefined }
        : { name, type: "space", unit: undefined, orientation };
    }),
    datasets: [
      { path: "0", coordinateTransformations: [createScale([1.0, 1.0, 1.0])] },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

/**
 * Assert that `fn` throws a {@link ValidationError} carrying `expectedRule` and
 * exactly `expectedLocation` (a non-empty, populated location).
 */
function assertRuleViolation(
  fn: () => void,
  expectedRule: SpecRule,
  expectedLocation: string,
): void {
  const error = assertThrows(fn, ValidationError);
  assertEquals(error.rule, expectedRule);
  assertEquals(error.location, expectedLocation);
}

Deno.test("validateAxisOrientation - no orientation is a no-op", () => {
  // No spatial axis declares orientation: the rule returns without raising.
  validateAxisOrientation(
    metadataWithOrientations([undefined, undefined, undefined]),
  );
});

Deno.test("validateAxisOrientation - accepts a consistent orientation", () => {
  // The serialized `{ type, value }` form, as produced by the image read path.
  validateAxisOrientation(
    metadataWithOrientations(
      LPS_VALUES.map((value) => ({ type: "anatomical", value })),
    ),
  );
  // The AnatomicalOrientation enum form, as produced by the write path.
  const enumValues = [
    AnatomicalOrientationValues.InferiorToSuperior,
    AnatomicalOrientationValues.AnteriorToPosterior,
    AnatomicalOrientationValues.RightToLeft,
  ];
  validateAxisOrientation(
    metadataWithOrientations(
      enumValues.map((value) => ({ type: "anatomical", value })),
    ),
  );
});

Deno.test("validateAxisOrientation - rejects inconsistent orientation types", () => {
  // The "y" axis declares a different orientation type than its siblings.
  const metadata = metadataWithOrientations([
    { type: "anatomical", value: LPS_VALUES[0] },
    { type: "other", value: LPS_VALUES[1] },
    { type: "anatomical", value: LPS_VALUES[2] },
  ]);
  assertRuleViolation(
    () => validateAxisOrientation(metadata),
    SpecRule.AxisOrientationConsistentType,
    "multiscales[0].axes",
  );
});

Deno.test("validateAxisOrientation - rejects incomplete orientation", () => {
  // Orientation is defined for "z" and "y" but missing on "x".
  const metadata = metadataWithOrientations([
    { type: "anatomical", value: LPS_VALUES[0] },
    { type: "anatomical", value: LPS_VALUES[1] },
    undefined,
  ]);
  const error = assertThrows(
    () => validateAxisOrientation(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.AxisOrientationCompleteness);
  assertEquals(error.location, "multiscales[0].axes");
  // The axis-name lists render Python-style (`['z', 'y']`) so the surfaced
  // RFC 4 message is byte-for-byte identical to the Python port (locks parity).
  assertEquals(
    error.message,
    "Spec rule [axis-orientation-completeness] violated: RFC 4 requires " +
      "that if orientation is defined for one spatial axis, it must be " +
      "defined for all spatial axes. Axes with orientation: ['z', 'y'], " +
      "axes without orientation: ['x']",
  );
});

Deno.test(
  "validateRfc4Orientation - messages carry the SpecRule mapping markers",
  () => {
    // validateAxisOrientation distinguishes the two RFC 4 orientation failures
    // by searching this function's message text for "same type" and "all
    // spatial axes". Those markers are a load-bearing contract: editing the
    // wording away makes the wrapper silently rethrow a plain Error instead of
    // the mapped ValidationError. Asserting them here fails loudly at the
    // source. Mirrors the Python
    // test_rfc4_orientation_messages_carry_mapping_markers.

    // Mixed orientation type -> the marker the consistent-type rule maps.
    assertThrows(
      () =>
        validateRfc4Orientation([
          {
            name: "y",
            type: "space",
            orientation: { type: "anatomical", value: "anterior-to-posterior" },
          },
          {
            name: "x",
            type: "space",
            orientation: { type: "other", value: "right-to-left" },
          },
        ]),
      Error,
      "same type",
    );

    // Orientation on some but not all spatial axes -> the completeness marker.
    assertThrows(
      () =>
        validateRfc4Orientation([
          {
            name: "y",
            type: "space",
            orientation: { type: "anatomical", value: "anterior-to-posterior" },
          },
          { name: "x", type: "space" },
        ]),
      Error,
      "all spatial axes",
    );

    // Orientation on a non-spatial axis -> the on-non-space marker.
    assertThrows(
      () =>
        validateRfc4Orientation([
          {
            name: "t",
            type: "time",
            orientation: { type: "anatomical", value: "inferior-to-superior" },
          },
        ]),
      Error,
      "non-space axes",
    );

    // Two spatial axes on one anatomical axis -> the unique-axis marker.
    assertThrows(
      () =>
        validateRfc4Orientation([
          {
            name: "y",
            type: "space",
            orientation: { type: "anatomical", value: "left-to-right" },
          },
          {
            name: "x",
            type: "space",
            orientation: { type: "anatomical", value: "right-to-left" },
          },
        ]),
      Error,
      "same anatomical axis",
    );
  },
);

Deno.test(
  "validateRfc4Orientation - same-name spatial entries still collide",
  () => {
    // Malformed metadata can repeat an axis name. A name-keyed map would
    // overwrite the first entry and let the opposing pair slip through, so the
    // check keeps every [name, value] entry. Mirrors the Python
    // test_validate_rfc4_orientation_duplicate_anatomical_axis_same_name.
    assertThrows(
      () =>
        validateRfc4Orientation([
          {
            name: "x",
            type: "space",
            orientation: { type: "anatomical", value: "left-to-right" },
          },
          {
            name: "x",
            type: "space",
            orientation: { type: "anatomical", value: "right-to-left" },
          },
        ]),
      Error,
      "same anatomical axis",
    );
  },
);
