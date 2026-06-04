// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * End-to-end demo of OME-Zarr v0.4 structural validation.
 *
 * Proves {@link validateStructural} works on hand-built {@link Metadata}
 * objects with no network access and no test fixtures.
 *
 * It first builds a fully valid `[c, y, x]` two-level multiscales metadata and
 * confirms it passes. It then builds a deliberately broken copy for each
 * image-metadata rule `validateStructural` enforces over plain multiscales
 * metadata -- axis count, axis type, axis order, scale-vector length,
 * per-level coordinate-transform ordering, dataset ordering, and OMERO channel
 * color -- each mutating a single field, and confirms `validateStructural`
 * raises the expected {@link SpecRule}, printing the caught rule identifier,
 * location, and message for each.
 *
 * Out of this demo's scope: the two RFC 4 anatomical-orientation rules and the
 * two HCS plate/well rules (enforced through the separate `validatePlate` /
 * `validateWell` orchestrators).
 *
 * Run from the `ts/` directory:
 *
 *     deno run examples/validate_structural_demo.ts
 *
 * The process exits non-zero if the valid metadata is rejected, if any broken
 * copy fails to raise, or if a raised rule does not match the expected one, so
 * the script doubles as a self-checking smoke test.
 */

import {
  SpecRule,
  validateStructural,
  ValidationError,
} from "../src/utils/structural_validation.ts";
import {
  createScale,
  createTranslation,
  type Metadata,
} from "../src/types/zarr_metadata.ts";

/**
 * Return a fully valid v0.4 `[c, y, x]` two-level multiscales metadata.
 *
 * Three axes (one channel, two space); two datasets, each with a single
 * length-3 `scale`; the second level is coarser (larger spatial scale) than
 * the first, as the spec requires. A fresh object is returned on each call so
 * callers may mutate one field to construct an invalid case in isolation.
 */
function buildValidMetadata(): Metadata {
  return {
    axes: [
      { name: "c", type: "channel", unit: undefined },
      { name: "y", type: "space", unit: undefined },
      { name: "x", type: "space", unit: undefined },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [createScale([1.0, 1.0, 1.0])],
      },
      {
        path: "1",
        coordinateTransformations: [createScale([1.0, 2.0, 2.0])],
      },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

/** One deliberately broken metadata copy and the rule it should violate. */
interface InvalidCase {
  /** Human-readable description of the mutation. */
  label: string;
  /** The {@link SpecRule} the orchestrator is expected to raise first. */
  expectedRule: SpecRule;
  /** The broken metadata to validate. */
  metadata: Metadata;
}

/**
 * Build one broken copy for each image-metadata rule the demo covers.
 *
 * Each case mutates exactly one field of the valid baseline so that the named
 * rule is the first (and only) one violated, given the orchestrator's
 * fail-fast spec-MUST ordering. The RFC 4 orientation rules and the HCS
 * plate/well rules are out of scope; see the module-level comment.
 */
function buildInvalidCases(): InvalidCase[] {
  const cases: InvalidCase[] = [];

  // Six axes: the count is outside the permitted 2..=5 range.
  let metadata = buildValidMetadata();
  metadata.axes = [
    { name: "t", type: "time", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
  ];
  cases.push({
    label: "six axes (count out of 2..=5)",
    expectedRule: SpecRule.AxisCount,
    metadata,
  });

  // Two channel axes: at most one is permitted.
  metadata = buildValidMetadata();
  metadata.axes = [
    { name: "c", type: "channel", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "y", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
  ];
  cases.push({
    label: "two channel axes",
    expectedRule: SpecRule.AxisType,
    metadata,
  });

  // Spatial axes in (x, y) order: names must be a suffix of (z, y, x).
  metadata = buildValidMetadata();
  metadata.axes = [
    { name: "c", type: "channel", unit: undefined },
    { name: "x", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
  ];
  cases.push({
    label: "spatial axes in x, y order",
    expectedRule: SpecRule.AxisOrder,
    metadata,
  });

  // A dataset scale vector of length 2 against 3 axes.
  metadata = buildValidMetadata();
  metadata.datasets[0].coordinateTransformations = [createScale([1.0, 1.0])];
  cases.push({
    label: "length-2 scale vector",
    expectedRule: SpecRule.ScaleLengthMismatch,
    metadata,
  });

  // A translation that precedes its scale within a dataset.
  metadata = buildValidMetadata();
  metadata.datasets[1].coordinateTransformations = [
    createTranslation([0.0, 0.0, 0.0]),
    createScale([1.0, 2.0, 2.0]),
  ];
  cases.push({
    label: "translation before scale",
    expectedRule: SpecRule.GlobalCoordTransformAfterPerLevel,
    metadata,
  });

  // A coarser level whose spatial scale is finer than the level above it.
  metadata = buildValidMetadata();
  metadata.datasets[0].coordinateTransformations = [
    createScale([1.0, 2.0, 2.0]),
  ];
  metadata.datasets[1].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0]),
  ];
  cases.push({
    label: "coarser level with finer scale",
    expectedRule: SpecRule.DatasetOrderHighestToLowest,
    metadata,
  });

  // An OMERO channel color that is not exactly six hex digits.
  metadata = buildValidMetadata();
  metadata.omero = {
    channels: [
      {
        color: "xyz",
        window: { min: 0.0, max: 255.0, start: 0.0, end: 255.0 },
      },
    ],
  };
  cases.push({
    label: "omero channel color 'xyz'",
    expectedRule: SpecRule.OmeroChannelColorFormat,
    metadata,
  });

  return cases;
}

/** Run the demo; return a process exit code (0 on full success). */
function main(): number {
  console.log("OME-Zarr v0.4 structural validation demo");
  console.log("=".repeat(64));

  const valid = buildValidMetadata();
  try {
    validateStructural(valid);
  } catch (error) {
    if (!(error instanceof ValidationError)) {
      throw error;
    }
    console.log(`UNEXPECTED: valid metadata was rejected: ${error.message}`);
    return 1;
  }
  console.log("valid [c, y, x] two-level metadata: passed validateStructural");
  console.log("-".repeat(64));

  const cases = buildInvalidCases();
  let caught = 0;
  for (const { label, expectedRule, metadata } of cases) {
    let threw = false;
    try {
      validateStructural(metadata);
    } catch (error) {
      threw = true;
      if (!(error instanceof ValidationError)) {
        throw error;
      }
      if (error.rule !== expectedRule) {
        console.log(
          `UNEXPECTED rule for ${JSON.stringify(label)}: ` +
            `got [${error.rule}], expected [${expectedRule}]`,
        );
        return 1;
      }
      caught += 1;
      console.log(`caught [${error.rule}] at ${error.location}`);
      console.log(`    ${label}: ${error.message}`);
    }
    if (!threw) {
      console.log(
        `UNEXPECTED: ${JSON.stringify(label)} did not raise ValidationError`,
      );
      return 1;
    }
  }

  console.log("-".repeat(64));
  const total = cases.length;
  console.log(`caught ${caught}/${total} expected violations`);
  return caught === total ? 0 : 1;
}

if (import.meta.main) {
  Deno.exit(main());
}
