// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for the OME-Zarr v0.4 structural validation engine in
 * `../src/utils/structural_validation.ts`.
 *
 * One paired pass/fail test per structural rule: a shared, fully valid
 * `[c, y, x]` two-level baseline ({@link buildValidMetadata}) is mutated in
 * exactly one field to trip exactly one rule, asserting the raised
 * {@link ValidationError} carries the expected {@link SpecRule} and a populated
 * `location`. Each rule function is exercised directly so the violation is
 * isolated to that rule. Further tests cover the public surface (rule-identifier
 * strings, level values, error formatting) and the orchestrator's fail-fast
 * ordering and `schema_only` short-circuit.
 *
 * The rule identifiers and location strings are kept byte-for-byte identical to
 * the package's Python implementation; this suite mirrors
 * `py/test/test_structural_validation.py` so the two stay in lockstep.
 */

import { assertEquals, assertInstanceOf, assertThrows } from "@std/assert";
import {
  SpecRule,
  validateAxisCount,
  validateAxisOrder,
  validateAxisType,
  validateDatasetOrder,
  validateOmeroColorHex,
  validatePerDatasetScaleCount,
  validateScaleLength,
  validateSpatialAxisOrder,
  validateStructural,
  validateTransformOrder,
  ValidationError,
  ValidationLevel,
} from "../src/utils/structural_validation.ts";
import {
  type Axis,
  createScale,
  createTranslation,
  type Metadata,
  type Omero,
} from "../src/types/zarr_metadata.ts";

/**
 * Return a fully valid v0.4 `[c, y, x]` two-level multiscales metadata.
 *
 * Three axes (one channel, two space); two datasets, each with a single
 * length-3 `scale`; the second level is coarser than the first. This baseline
 * passes every structural rule, so each test mutates exactly one field of a
 * fresh copy to trip exactly one rule.
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

/** Build a single-channel OMERO block with the given channel `color`. */
function makeOmero(color: string): Omero {
  return {
    channels: [
      {
        color,
        window: { min: 0.0, max: 255.0, start: 0.0, end: 255.0 },
      },
    ],
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

// ---------------------------------------------------------------------------
// Per-rule paired pass/fail tests
// ---------------------------------------------------------------------------

Deno.test("validateAxisCount - accepts the 3-axis baseline", () => {
  validateAxisCount(buildValidMetadata());
});

Deno.test("validateAxisCount - rejects counts outside 2..5", () => {
  for (const count of [1, 6]) {
    const metadata = buildValidMetadata();
    metadata.axes = Array.from(
      { length: count },
      (): Axis => ({ name: "x", type: "space", unit: undefined }),
    );
    assertRuleViolation(
      () => validateAxisCount(metadata),
      SpecRule.AxisCount,
      "multiscales[0].axes",
    );
  }
});

Deno.test("validateAxisType - accepts a single channel axis", () => {
  validateAxisType(buildValidMetadata());
});

Deno.test("validateAxisType - rejects two time or two channel axes", () => {
  for (const [axisType, name] of [["time", "t"], ["channel", "c"]] as const) {
    const metadata = buildValidMetadata();
    metadata.axes = [
      { name, type: axisType, unit: undefined },
      { name, type: axisType, unit: undefined },
      { name: "y", type: "space", unit: undefined },
      { name: "x", type: "space", unit: undefined },
    ];
    assertRuleViolation(
      () => validateAxisType(metadata),
      SpecRule.AxisType,
      "multiscales[0].axes",
    );
  }
});

Deno.test("validateAxisOrder - accepts channel-before-space order", () => {
  validateAxisOrder(buildValidMetadata());
});

Deno.test("validateAxisOrder - rejects time after channel", () => {
  // A 'time' axis after a 'channel' axis inverts the time < channel ranking.
  const metadata = buildValidMetadata();
  metadata.axes = [
    { name: "c", type: "channel", unit: undefined },
    { name: "t", type: "time", unit: undefined },
    { name: "y", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
  ];
  assertRuleViolation(
    () => validateAxisOrder(metadata),
    SpecRule.AxisOrder,
    "multiscales[0].axes[1]",
  );
});

Deno.test("validateSpatialAxisOrder - accepts the (y, x) suffix", () => {
  validateSpatialAxisOrder(buildValidMetadata());
});

Deno.test("validateSpatialAxisOrder - rejects bad suffix and too many", () => {
  // (x, y) is not the (y, x) suffix of (z, y, x).
  const suffixMismatch = buildValidMetadata();
  suffixMismatch.axes = [
    { name: "c", type: "channel", unit: undefined },
    { name: "x", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
  ];
  assertRuleViolation(
    () => validateSpatialAxisOrder(suffixMismatch),
    SpecRule.AxisOrder,
    "multiscales[0].axes[1]",
  );

  // Four space axes exceed the v0.4 maximum of three.
  const tooMany = buildValidMetadata();
  tooMany.axes = [
    { name: "z", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
  ];
  assertRuleViolation(
    () => validateSpatialAxisOrder(tooMany),
    SpecRule.AxisOrder,
    "multiscales[0].axes",
  );
});

Deno.test("validatePerDatasetScaleCount - accepts one scale per level", () => {
  validatePerDatasetScaleCount(buildValidMetadata());
});

Deno.test("validatePerDatasetScaleCount - rejects zero or two scales", () => {
  const zeroScales = buildValidMetadata();
  zeroScales.datasets[0].coordinateTransformations = [
    createTranslation([0.0, 0.0, 0.0]),
  ];
  assertRuleViolation(
    () => validatePerDatasetScaleCount(zeroScales),
    SpecRule.GlobalCoordTransformAfterPerLevel,
    "multiscales[0].datasets[0].coordinateTransformations",
  );

  const twoScales = buildValidMetadata();
  twoScales.datasets[0].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0]),
    createScale([1.0, 1.0, 1.0]),
  ];
  assertRuleViolation(
    () => validatePerDatasetScaleCount(twoScales),
    SpecRule.GlobalCoordTransformAfterPerLevel,
    "multiscales[0].datasets[0].coordinateTransformations",
  );
});

Deno.test("validateScaleLength - accepts matching global and per-level", () => {
  const metadata = buildValidMetadata();
  validateScaleLength(metadata);
  // A correctly sized global scale (length == axis count) is also accepted.
  metadata.coordinateTransformations = [createScale([1.0, 1.0, 1.0])];
  validateScaleLength(metadata);
});

Deno.test("validateScaleLength - rejects a short per-dataset scale", () => {
  // A length-2 dataset scale against 3 axes.
  const metadata = buildValidMetadata();
  metadata.datasets[0].coordinateTransformations = [createScale([1.0, 1.0])];
  assertRuleViolation(
    () => validateScaleLength(metadata),
    SpecRule.ScaleLengthMismatch,
    "multiscales[0].datasets[0].coordinateTransformations[0]",
  );
});

Deno.test("validateScaleLength - rejects a short global scale", () => {
  // A length-2 global scale against 3 axes.
  const metadata = buildValidMetadata();
  metadata.coordinateTransformations = [createScale([1.0, 1.0])];
  assertRuleViolation(
    () => validateScaleLength(metadata),
    SpecRule.ScaleLengthMismatch,
    "multiscales[0].coordinateTransformations[0]",
  );
});

Deno.test("validateTransformOrder - accepts a scale then translation", () => {
  const metadata = buildValidMetadata();
  validateTransformOrder(metadata);
  // A translation following its scale is valid.
  metadata.datasets[0].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0]),
    createTranslation([0.0, 0.0, 0.0]),
  ];
  validateTransformOrder(metadata);
});

Deno.test("validateTransformOrder - rejects a scale after a translation", () => {
  // A scale after a translation: a translation must follow its scale.
  const metadata = buildValidMetadata();
  metadata.datasets[0].coordinateTransformations = [
    createTranslation([0.0, 0.0, 0.0]),
    createScale([1.0, 1.0, 1.0]),
  ];
  assertRuleViolation(
    () => validateTransformOrder(metadata),
    SpecRule.GlobalCoordTransformAfterPerLevel,
    "multiscales[0].datasets[0].coordinateTransformations[1]",
  );
});

Deno.test("validateDatasetOrder - accepts finest-to-coarsest order", () => {
  validateDatasetOrder(buildValidMetadata());
});

Deno.test("validateDatasetOrder - rejects a finer scale at a coarser level", () => {
  // The second (coarser) level has a finer spatial scale than the first.
  const metadata = buildValidMetadata();
  metadata.datasets[0].coordinateTransformations = [
    createScale([1.0, 2.0, 2.0]),
  ];
  metadata.datasets[1].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0]),
  ];
  const error = assertThrows(
    () => validateDatasetOrder(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.DatasetOrderHighestToLowest);
  assertEquals(error.location, "multiscales[0].datasets[1]");
  // Whole-number float scales render Python-style (`1.0`, not `1`) so the
  // message is byte-for-byte identical to the Python port (locks parity).
  assertEquals(
    error.message,
    "Spec rule [dataset-order-highest-to-lowest] violated: Datasets must " +
      "be ordered finest to coarsest; dataset 1 has spatial scale 1.0 on " +
      "axis 'y', smaller than dataset 0's 2.0.",
  );
});

Deno.test("validateOmeroColorHex - accepts absent omero and valid hex", () => {
  validateOmeroColorHex(buildValidMetadata()); // omero is undefined
  const metadata = buildValidMetadata();
  metadata.omero = makeOmero("00FF88");
  validateOmeroColorHex(metadata);
});

Deno.test("validateOmeroColorHex - rejects a non-hex channel color", () => {
  const metadata = buildValidMetadata();
  metadata.omero = makeOmero("xyz");
  assertRuleViolation(
    () => validateOmeroColorHex(metadata),
    SpecRule.OmeroChannelColorFormat,
    "multiscales[0].omero.channels[0].color",
  );
});

// ---------------------------------------------------------------------------
// Public-surface tests
// ---------------------------------------------------------------------------

Deno.test("SpecRule values are the exact kebab-case identifiers", () => {
  assertEquals(SpecRule.AxisCount, "axis-count");
  assertEquals(SpecRule.AxisType, "axis-type");
  assertEquals(SpecRule.AxisOrder, "axis-order");
  assertEquals(SpecRule.ScaleLengthMismatch, "scale-length-mismatch");
  assertEquals(
    SpecRule.GlobalCoordTransformAfterPerLevel,
    "global-coord-transform-after-per-level",
  );
  assertEquals(
    SpecRule.DatasetOrderHighestToLowest,
    "dataset-order-highest-to-lowest",
  );
  assertEquals(SpecRule.OmeroChannelColorFormat, "omero-channel-color-format");
});

Deno.test("ValidationLevel values match the spec strings", () => {
  assertEquals(ValidationLevel.Strict, "strict");
  assertEquals(ValidationLevel.SchemaOnly, "schema_only");
});

Deno.test("ValidationError formats its message and carries rule/location", () => {
  const error = new ValidationError(
    SpecRule.AxisCount,
    "found 6 axes",
    "multiscales[0].axes",
  );
  assertInstanceOf(error, ValidationError);
  assertInstanceOf(error, Error);
  assertEquals(error.name, "ValidationError");
  assertEquals(error.message, "Spec rule [axis-count] violated: found 6 axes");
  assertEquals(error.rule, SpecRule.AxisCount);
  assertEquals(error.location, "multiscales[0].axes");
});

// ---------------------------------------------------------------------------
// Orchestrator tests
// ---------------------------------------------------------------------------

Deno.test("validateStructural - accepts the valid baseline", () => {
  // Default options (strict) and an explicit strict both accept the baseline.
  validateStructural(buildValidMetadata());
  validateStructural(buildValidMetadata(), { level: ValidationLevel.Strict });
});

Deno.test("validateStructural - is fail-fast in canonical rule order", () => {
  // Two independent violations: dataset-order (rule 8) and omero-color (9).
  const metadata = buildValidMetadata();
  metadata.datasets[0].coordinateTransformations = [
    createScale([1.0, 2.0, 2.0]),
  ];
  metadata.datasets[1].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0]),
  ];
  metadata.omero = makeOmero("xyz");
  const error = assertThrows(
    () => validateStructural(metadata),
    ValidationError,
  );
  // Fail-fast reports the earlier-ordered rule, not the OMERO color.
  assertEquals(error.rule, SpecRule.DatasetOrderHighestToLowest);
});

Deno.test("validateStructural - default is strict; schema_only skips", () => {
  // A single axis is a clear axis-count violation under strict...
  const metadata = buildValidMetadata();
  metadata.axes = [{ name: "x", type: "space", unit: undefined }];
  // ...with no options the default strict level runs the rules and rejects it,
  // confirming the documented default; an explicit strict rejects it too.
  assertThrows(() => validateStructural(metadata), ValidationError);
  assertThrows(
    () => validateStructural(metadata, { level: ValidationLevel.Strict }),
    ValidationError,
  );
  // ...but schema_only runs no structural rule, so the same metadata passes.
  validateStructural(metadata, { level: ValidationLevel.SchemaOnly });
});

Deno.test("validateStructural - option defaults: strict, allowUnknownFields true", () => {
  // ValidateOptions is a bare interface, so its documented defaults --
  // { level: "strict", allowUnknownFields: true } -- are resolved inside
  // validateStructural rather than on a constructable options object (unlike
  // Python's ValidateOptions dataclass). Pin the defaults' observable contract:
  // omitting options, an empty object, and either allowUnknownFields value all
  // accept the valid baseline (the default level is strict and the baseline
  // passes the strict rules).
  validateStructural(buildValidMetadata());
  validateStructural(buildValidMetadata(), {});
  validateStructural(buildValidMetadata(), { allowUnknownFields: true });
  validateStructural(buildValidMetadata(), { allowUnknownFields: false });

  // allowUnknownFields is inert with respect to the structural rules (matching
  // Python): a structural violation is still reported under its default-true
  // value and when explicitly disabled.
  const invalid = buildValidMetadata();
  invalid.datasets[0].coordinateTransformations = [createScale([1.0, 1.0])];
  assertRuleViolation(
    () => validateStructural(invalid, { allowUnknownFields: true }),
    SpecRule.ScaleLengthMismatch,
    "multiscales[0].datasets[0].coordinateTransformations[0]",
  );
  assertRuleViolation(
    () => validateStructural(invalid, { allowUnknownFields: false }),
    SpecRule.ScaleLengthMismatch,
    "multiscales[0].datasets[0].coordinateTransformations[0]",
  );
});
