// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Parity/manifest test locking the observable structural-validation surface.
 *
 * This is the Deno mirror of `py/test/test_structural_validation_parity.py`. It
 * pins the *cross-language contract* shared by the TypeScript and Python ports:
 * the exact set and ordering of {@link SpecRule} identifiers, their
 * lower-kebab-case format, the fail-fast evaluation order of the
 * image/multiscales orchestrator ({@link validateStructural}), and the
 * {@link ValidationLevel} values and default. The two suites assert the same
 * facts against the identical literal identifier list
 * ({@link CANONICAL_SPEC_RULE_IDS}), so they are line-for-line comparable and
 * any future drift between the ports fails CI.
 *
 * Adding, removing, renaming, or reordering a rule -- in either language -- must
 * therefore be a deliberate edit to {@link CANONICAL_SPEC_RULE_IDS} (and its
 * Python twin), not an accident.
 *
 * Two adaptations to TypeScript are unavoidable and are the only intentional
 * departures from the Python twin:
 *
 * 1. Axis names. TypeScript constrains `Axis.name` to the closed `SupportedDims`
 *    union (`"c" | "x" | "y" | "z" | "t"`), so the Python ordering test's
 *    free-form helper names (`"c2"`, `"w"`) are replaced with valid members
 *    (a repeated `"c"`, a `"y"`). The per-stage violations -- and therefore the
 *    observed evaluation order -- are identical; only the labels differ.
 * 2. The default level. Python pins it on a constructable options object
 *    (`ValidateOptions().level == STRICT`); TypeScript's `ValidateOptions` is a
 *    bare interface whose default is resolved inside {@link validateStructural}.
 *    The same default is asserted observably instead (see the level test).
 *
 * The validation surface is imported from the package root (`../src/mod.ts`),
 * mirroring the Python test's import from `ngff_zarr`; the metadata constructors
 * come from the types module, mirroring its import from
 * `ngff_zarr.v04.zarr_metadata`.
 */

import { assertEquals, assertMatch, assertThrows } from "@std/assert";
import {
  SpecRule,
  validateStructural,
  ValidationError,
  ValidationLevel,
} from "../src/mod.ts";
import {
  type Axis,
  type AxisOrientation,
  createScale,
  createTranslation,
  type Metadata,
  type Omero,
} from "../src/types/zarr_metadata.ts";

// The locked rule manifest: every active OME-Zarr structural rule, in canonical
// declaration order. This identical literal list appears in the Python twin so
// the two are directly comparable. The first eleven entries are the
// image/multiscales rules dispatched by validateStructural -- the first nine are
// the v0.4 rules and the next two (zarr-format, ome-namespace) are the v0.5
// namespacing rules, inert for v0.4; the final two are the HCS plate/well rules
// dispatched by validatePlate / validateWell.
const CANONICAL_SPEC_RULE_IDS: string[] = [
  "axis-count",
  "axis-type",
  "axis-order",
  "scale-length-mismatch",
  "global-coord-transform-after-per-level",
  "dataset-order-highest-to-lowest",
  "omero-channel-color-format",
  "axis-orientation-consistent-type",
  "axis-orientation-completeness",
  "axis-orientation-on-non-space",
  "axis-orientation-unique-axis",
  "zarr-format",
  "ome-namespace",
  "plate-row-index-consistency",
  "well-acquisition-missing",
];

// The canonical fail-fast evaluation order of the image/multiscales
// orchestrator (validateStructural). Each entry is the SpecRule the
// orchestrator must raise when that rule -- and every rule after it -- is
// violated at once. Two SpecRule values appear twice because two rule functions
// share each: validateAxisOrder and validateSpatialAxisOrder both surface
// AxisOrder (positions 3-4), and validatePerDatasetScaleCount and
// validateTransformOrder both surface GlobalCoordTransformAfterPerLevel
// (positions 5 and 7). The two HCS rules are absent: they are not part of the
// image/multiscales orchestrator. The two v0.5 namespacing rules (zarr-format,
// ome-namespace) run last in the orchestrator but are inert for the v0.4
// metadata exercised here, so they never appear in the observed order.
const EXPECTED_EVALUATION_ORDER: SpecRule[] = [
  SpecRule.AxisCount,
  SpecRule.AxisType,
  SpecRule.AxisOrder, // validateAxisOrder (class ordering)
  SpecRule.AxisOrder, // validateSpatialAxisOrder (spatial suffix)
  SpecRule.GlobalCoordTransformAfterPerLevel, // per-dataset scale count
  SpecRule.ScaleLengthMismatch,
  SpecRule.GlobalCoordTransformAfterPerLevel, // transform order
  SpecRule.DatasetOrderHighestToLowest,
  SpecRule.OmeroChannelColorFormat,
  SpecRule.AxisOrientationConsistentType,
];

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

/** Render an anatomical orientation as the record an image read produces. */
function orientation(type: string, value: string): AxisOrientation {
  return { type, value };
}

/**
 * Return valid `[c, z, y, x]` axes whose orientations mix RFC 4 types.
 *
 * Every axis rule passes (channel-then-space class order, `(z, y, x)` spatial
 * suffix, count 4), but the `y` axis declares orientation type `"other"` while
 * `z` and `x` declare `"anatomical"` -- tripping
 * {@link SpecRule.AxisOrientationConsistentType}, the last orchestrator rule,
 * once every earlier rule is satisfied.
 */
function validAxesWithInconsistentOrientation(): Axis[] {
  return [
    { name: "c", type: "channel", unit: undefined },
    {
      name: "z",
      type: "space",
      unit: undefined,
      orientation: orientation("anatomical", "inferior-to-superior"),
    },
    {
      name: "y",
      type: "space",
      unit: undefined,
      orientation: orientation("other", "anterior-to-posterior"),
    },
    {
      name: "x",
      type: "space",
      unit: undefined,
      orientation: orientation("anatomical", "right-to-left"),
    },
  ];
}

/** Run the orchestrator and return the single {@link SpecRule} it raises. */
function firstViolatedRule(metadata: Metadata): SpecRule {
  const error = assertThrows(
    () => validateStructural(metadata),
    ValidationError,
  );
  return error.rule;
}

// ---------------------------------------------------------------------------
// Manifest: the locked SpecRule set, ordering, and format
// ---------------------------------------------------------------------------

Deno.test("SpecRule manifest is locked to the canonical identifier set", () => {
  const ruleValues: string[] = Object.values(SpecRule);
  // The exact set must equal the canonical manifest: adding, removing, or
  // renaming a rule must be a deliberate edit to CANONICAL_SPEC_RULE_IDS.
  assertEquals(new Set(ruleValues), new Set(CANONICAL_SPEC_RULE_IDS));
  // Declaration order is part of the cross-language contract, so lock it too;
  // this is the order the Python SpecRule enum must also declare.
  assertEquals(ruleValues, CANONICAL_SPEC_RULE_IDS);
  // No two members may share a value (which would create a silent alias).
  assertEquals(new Set(ruleValues).size, CANONICAL_SPEC_RULE_IDS.length);
});

Deno.test("SpecRule identifiers are lower-kebab-case", () => {
  // Each identifier is one or more lowercase ASCII words joined by hyphens.
  // The anchored RegExp.test is the equivalent of Python's re.fullmatch with
  // the identical pattern body `[a-z]+(-[a-z]+)*`.
  const kebab = /^[a-z]+(-[a-z]+)*$/;
  for (const value of Object.values(SpecRule)) {
    assertMatch(value, kebab);
  }
});

Deno.test("ValidationLevel manifest and default", () => {
  const levelValues: string[] = Object.values(ValidationLevel);
  // Exactly two levels, with the documented string values...
  assertEquals(new Set(levelValues), new Set(["schema_only", "strict"]));
  assertEquals(ValidationLevel.SchemaOnly, "schema_only");
  assertEquals(ValidationLevel.Strict, "strict");
  // ...and strict is the default applied when no options are passed. Python
  // pins this on a constructable options object (ValidateOptions().level ==
  // STRICT); TypeScript's ValidateOptions is a bare interface whose default is
  // resolved inside validateStructural. Assert the identical default
  // observably: omitting options (and an empty options object) runs the strict
  // rules and rejects an invalid metadata, while schema_only skips them.
  const invalid: Metadata = {
    axes: [{ name: "x", type: "space", unit: undefined }],
    datasets: [
      { path: "0", coordinateTransformations: [createScale([1.0])] },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
  assertThrows(() => validateStructural(invalid), ValidationError);
  assertThrows(() => validateStructural(invalid, {}), ValidationError);
  validateStructural(invalid, { level: ValidationLevel.SchemaOnly });
});

// ---------------------------------------------------------------------------
// Orchestrator evaluation order (fail-fast, canonical spec-MUST order)
// ---------------------------------------------------------------------------

Deno.test("validateStructural evaluates rules in canonical order", () => {
  // Starts from metadata that violates the earliest rule *and* every later
  // rule, then repairs exactly one rule at a time. Because each repair leaves
  // all later violations intact, the rule raised at each step proves that rule
  // is evaluated strictly before every rule after it. The observed sequence
  // must equal EXPECTED_EVALUATION_ORDER.
  const observed: SpecRule[] = [];
  const badOmero = makeOmero("xyz"); // OMERO violation (rule 9), until repaired

  // Stage 1 -> AxisCount. Six axes simultaneously trip axis-type (two
  // channels), axis-order (a space precedes a channel), and spatial-order
  // (names are not the (z, y, x) suffix); axis-count is evaluated first.
  const metadata: Metadata = {
    axes: [
      { name: "x", type: "space", unit: undefined },
      { name: "c", type: "channel", unit: undefined },
      { name: "c", type: "channel", unit: undefined },
      { name: "t", type: "time", unit: undefined },
      { name: "z", type: "space", unit: undefined },
      { name: "y", type: "space", unit: undefined },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          createScale([1.0, 1.0, 1.0]),
          createScale([1.0, 1.0, 1.0]),
        ],
      },
      {
        path: "1",
        coordinateTransformations: [createScale([1.0, 2.0, 2.0, 2.0])],
      },
    ],
    coordinateTransformations: undefined,
    omero: badOmero,
    name: "image",
    version: "0.4",
  };
  observed.push(firstViolatedRule(metadata));

  // Stage 2 -> AxisType. Count fixed (5 axes); two channels remain, and a
  // space still precedes a channel and the spatial names are still wrong.
  metadata.axes = [
    { name: "x", type: "space", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 3 -> AxisOrder (class ordering). One channel now, but a space axis
  // still precedes it; the spatial names are still not the (z, y, x) suffix.
  metadata.axes = [
    { name: "x", type: "space", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 4 -> AxisOrder (spatial suffix). Class order fixed (channel first),
  // but spatial names (x, z, y) are not the length-3 suffix of (z, y, x).
  metadata.axes = [
    { name: "c", type: "channel", unit: undefined },
    { name: "x", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 5 -> per-dataset scale count. Axes are now fully valid [c, z, y, x]
  // (with an inconsistent-orientation violation lurking as rule 10). Dataset 0
  // still has two scales, so the per-dataset-scale-count rule fires.
  metadata.axes = validAxesWithInconsistentOrientation();
  observed.push(firstViolatedRule(metadata));

  // Stage 6 -> ScaleLengthMismatch. Dataset 0 now has exactly one scale, but a
  // length-3 vector against 4 axes; a scale-after-translation (rule 7) lurks.
  metadata.datasets[0].coordinateTransformations = [
    createTranslation([0.0, 0.0, 0.0]),
    createScale([1.0, 1.0, 1.0]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 7 -> transform order. Lengths fixed to 4; dataset 0's scale still
  // follows a translation. Dataset 1 is made coarser-but-smaller so the
  // dataset-order rule (8) lurks behind the transform-order violation.
  metadata.datasets[0].coordinateTransformations = [
    createTranslation([0.0, 0.0, 0.0, 0.0]),
    createScale([1.0, 1.0, 1.0, 1.0]),
  ];
  metadata.datasets[1].coordinateTransformations = [
    createScale([1.0, 0.5, 0.5, 0.5]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 8 -> dataset order. Transform order fixed (scale before
  // translation); dataset 1 is still coarser-but-smaller than dataset 0.
  metadata.datasets[0].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0, 1.0]),
    createTranslation([0.0, 0.0, 0.0, 0.0]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 9 -> OMERO color. Dataset order fixed (level 1 coarser-larger); only
  // the bad OMERO color and the orientation violation remain.
  metadata.datasets[1].coordinateTransformations = [
    createScale([1.0, 2.0, 2.0, 2.0]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 10 -> orientation. OMERO color fixed; the y axis still declares a
  // different orientation type than its spatial siblings.
  metadata.omero = makeOmero("00FF88");
  observed.push(firstViolatedRule(metadata));

  assertEquals(observed, EXPECTED_EVALUATION_ORDER);

  // Fully repaired: give the y axis a consistent orientation type and the
  // orchestrator accepts the metadata with no violation.
  metadata.axes[2] = {
    name: "y",
    type: "space",
    unit: undefined,
    orientation: orientation("anatomical", "anterior-to-posterior"),
  };
  validateStructural(metadata);
});
