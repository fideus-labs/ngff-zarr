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
 * One adaptation to TypeScript is unavoidable and is the only intentional
 * departure from the Python twin: the default level. Python pins it on a
 * constructable options object (`ValidateOptions().level == STRICT`);
 * TypeScript's `ValidateOptions` is a bare interface whose default is resolved
 * inside {@link validateStructural}. The same default is asserted observably
 * instead (see the level test). `Axis.name` is `AxisName`, so the ordering
 * fixtures use the same free-form names as the Python twin.
 *
 * The validation surface is imported from the package root (`../src/mod.ts`),
 * mirroring the Python test's import from `ngff_zarr`; the metadata constructors
 * come from the types module, mirroring its import from
 * `ngff_zarr.v04.zarr_metadata`.
 */

import { assertEquals, assertMatch, assertThrows } from "@std/assert";
import { validateCollection } from "../src/mod.ts";
import {
  SpecRule,
  SUPPORTED_VERSIONS,
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
// the two are directly comparable. The first thirteen entries are the
// image/multiscales rules dispatched by validateStructural -- the first eleven
// are the v0.4 rules and the next two (zarr-format, ome-namespace) are the v0.5
// namespacing rules, inert for v0.4; the next two are the HCS plate/well rules
// dispatched by validatePlate / validateWell; the final twelve are the
// RFC-8 node/collection rules dispatched by validateCollection.
const CANONICAL_SPEC_RULE_IDS: string[] = [
  "axis-count",
  "axis-type",
  "axis-order",
  "axis-names-unique",
  "scale-length-mismatch",
  "global-coord-transform-after-per-level",
  "dataset-order-highest-to-lowest",
  "omero-channel-color-format",
  "axis-orientation-anatomical-type",
  "axis-orientation-on-non-space",
  "axis-orientation-unique-axis",
  "zarr-format",
  "ome-namespace",
  "plate-row-index-consistency",
  "well-acquisition-missing",
  "node-type-required",
  "node-name-required",
  "node-name-unique",
  "node-id-format",
  "node-id-unique",
  "node-nodes-xor-path",
  "path-type-known",
  "coordinate-system-id-required",
  "reference-id-required",
  "reference-path-required",
  "label-value-required",
  "label-color-format",
];

// The locked RFC-3 version manifest: the versions whose axis model is
// unrestricted, so that validateAxisCount / validateAxisType /
// validateAxisOrder / validateSpatialAxisOrder are inert for them. This
// identical literal list appears in the Python twin. axis-names-unique is
// deliberately absent: RFC-3 adds that rule rather than lifting it, so it is
// never inert (see docs/validation/rule-reference.md).
const CANONICAL_RFC3_VERSIONS: string[] = [
  "0.9.dev1",
  "0.9.dev3",
];

// Every other supported version, plus the no-version default, must enforce the
// axis rules. Read off SUPPORTED_VERSIONS so a newly supported version has to
// be classified here rather than silently defaulting to "restricted".
const NON_RFC3_VERSIONS: (string | undefined)[] = [
  ...SUPPORTED_VERSIONS
    .map((version) => version as string)
    .filter((version) => !CANONICAL_RFC3_VERSIONS.includes(version)),
  undefined,
];

// The locked RFC-4 orientation version manifest: the versions at which the
// three axis-orientation rules are normative (ome/ngff-spec#190 folds RFC-4
// into 0.9.dev1). This identical literal list appears in the Python twin.
// The gate points the opposite way from CANONICAL_RFC3_VERSIONS: RFC-3 *lifts*
// the axis restrictions at 0.9.dev1 while RFC-4 *adds* the orientation
// requirements, so the rules are inert below these versions. No version at all
// keeps them on, as a strictness choice (like axis-names-unique).
const CANONICAL_RFC4_VERSIONS: string[] = [
  "0.9.dev1",
  "0.9.dev3",
];

// Every other supported version must leave the orientation rules inert. Read
// off SUPPORTED_VERSIONS so a newly supported version has to be classified
// here rather than silently defaulting to either side.
const PRE_RFC4_VERSIONS: string[] = SUPPORTED_VERSIONS
  .map((version) => version as string)
  .filter((version) => !CANONICAL_RFC4_VERSIONS.includes(version));

// The locked RFC-8 version manifest: the versions that store the node model
// under `ome`, at which the seven node/collection rules are normative. This
// identical literal list appears in the Python twin. Like the RFC-4 rules,
// the collection rules stay on when no version is declared.
const CANONICAL_RFC8_VERSIONS: string[] = [
  "0.9.dev3",
];

// Every other supported version must leave the collection rules inert. Read
// off SUPPORTED_VERSIONS so a newly supported version has to be classified
// here rather than silently defaulting to either side.
const PRE_RFC8_VERSIONS: string[] = SUPPORTED_VERSIONS
  .map((version) => version as string)
  .filter((version) => !CANONICAL_RFC8_VERSIONS.includes(version));

// The canonical fail-fast evaluation order of the image/multiscales
// orchestrator (validateStructural). Each entry is the SpecRule the
// orchestrator must raise when that rule -- and every rule after it -- is
// violated at once. Two SpecRule values appear twice because two rule functions
// share each: validateAxisOrder and validateSpatialAxisOrder both surface
// AxisOrder (positions 3-4), and validatePerDatasetScaleCount and
// validateTransformOrder both surface GlobalCoordTransformAfterPerLevel
// (positions 6 and 8). The two HCS rules are absent: they are not part of the
// image/multiscales orchestrator. The two v0.5 namespacing rules (zarr-format,
// ome-namespace) run last in the orchestrator but are inert for the v0.4
// metadata exercised here, so they never appear in the observed order.
const EXPECTED_EVALUATION_ORDER: SpecRule[] = [
  SpecRule.AxisCount,
  SpecRule.AxisType,
  SpecRule.AxisOrder, // validateAxisOrder (class ordering)
  SpecRule.AxisOrder, // validateSpatialAxisOrder (spatial suffix)
  SpecRule.AxisNamesUnique,
  SpecRule.GlobalCoordTransformAfterPerLevel, // per-dataset scale count
  SpecRule.ScaleLengthMismatch,
  SpecRule.GlobalCoordTransformAfterPerLevel, // transform order
  SpecRule.DatasetOrderHighestToLowest,
  SpecRule.OmeroChannelColorFormat,
  SpecRule.AxisOrientationAnatomicalType,
];

/**
 * Valid `[c, z, y, x]` axes with the channel axis renamed `z`.
 *
 * Every rule before AxisNamesUnique still passes: the class order is
 * channel-then-space, the spatial names remain the `(z, y, x)` suffix, and the
 * count is 4. Only the repeated `z` is left for the orchestrator to catch,
 * which pins the rule's position in the cascade.
 */
function axesWithRepeatedName(): Axis[] {
  const axes = validAxesWithInconsistentOrientation();
  axes[0] = { ...axes[0], name: "z" };
  return axes;
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
 * {@link SpecRule.AxisOrientationAnatomicalType}, the last orchestrator rule,
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
  // channels), axis-order (a space precedes a channel), spatial-order (names
  // are not the (z, y, x) suffix) and axis-names-unique (a repeated "z");
  // axis-count is evaluated first.
  const metadata: Metadata = {
    axes: [
      { name: "x", type: "space", unit: undefined },
      { name: "c", type: "channel", unit: undefined },
      { name: "c2", type: "channel", unit: undefined },
      { name: "t", type: "time", unit: undefined },
      { name: "z", type: "space", unit: undefined },
      { name: "z", type: "space", unit: undefined },
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

  // Stage 2 -> AxisType. Count fixed (5 axes); two channels remain, a space
  // still precedes a channel, the spatial names are still wrong and "z" is
  // still repeated.
  metadata.axes = [
    { name: "x", type: "space", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "c2", type: "channel", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 3 -> AxisOrder (class ordering). One channel now, but a space axis
  // still precedes it; the spatial names are still not the (z, y, x) suffix
  // and "z" is still repeated.
  metadata.axes = [
    { name: "x", type: "space", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 4 -> AxisOrder (spatial suffix). Class order fixed (channel first),
  // but spatial names (x, z, z) are not the length-3 suffix of (z, y, x).
  metadata.axes = [
    { name: "c", type: "channel", unit: undefined },
    { name: "x", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 5 -> AxisNamesUnique. Every axis rule before it now passes, but the
  // channel axis is named "z" like the first space axis.
  metadata.axes = axesWithRepeatedName();
  observed.push(firstViolatedRule(metadata));

  // Stage 6 -> per-dataset scale count. Axes are now fully valid [c, z, y, x]
  // (with an inconsistent-orientation violation lurking as rule 10). Dataset 0
  // still has two scales, so the per-dataset-scale-count rule fires.
  metadata.axes = validAxesWithInconsistentOrientation();
  observed.push(firstViolatedRule(metadata));

  // Stage 7 -> ScaleLengthMismatch. Dataset 0 now has exactly one scale, but a
  // length-3 vector against 4 axes; a scale-after-translation (rule 7) lurks.
  metadata.datasets[0].coordinateTransformations = [
    createTranslation([0.0, 0.0, 0.0]),
    createScale([1.0, 1.0, 1.0]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 8 -> transform order. Lengths fixed to 4; dataset 0's scale still
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

  // Stage 9 -> dataset order. Transform order fixed (scale before
  // translation); dataset 1 is still coarser-but-smaller than dataset 0.
  metadata.datasets[0].coordinateTransformations = [
    createScale([1.0, 1.0, 1.0, 1.0]),
    createTranslation([0.0, 0.0, 0.0, 0.0]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 10 -> OMERO color. Dataset order fixed (level 1 coarser-larger); only
  // the bad OMERO color and the orientation violation remain.
  metadata.datasets[1].coordinateTransformations = [
    createScale([1.0, 2.0, 2.0, 2.0]),
  ];
  observed.push(firstViolatedRule(metadata));

  // Stage 11 -> orientation. OMERO color fixed; the y axis still declares a
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

// ---------------------------------------------------------------------------
// Manifest: the locked RFC-3 version set
// ---------------------------------------------------------------------------

/**
 * Six same-type axes: legal under RFC-3, illegal at every other version.
 *
 * Violates axis-count (6 > 5) and the spatial-axis rules (6 > 3 `space` axes)
 * at once, and nothing else, so the orchestrator accepts it exactly when the
 * axis rules are inert.
 */
function rfc3AxisMetadata(): Metadata {
  const names = ["a", "b", "c", "d", "e", "f"];
  return {
    axes: names.map((name) => ({
      name,
      type: "space",
      unit: undefined,
    } as Axis)),
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          createScale(names.map(() => 1.0)),
          createTranslation(names.map(() => 0.0)),
        ],
      },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

Deno.test("RFC-3 version manifest is locked", () => {
  // Inert at exactly the manifest versions...
  for (const version of CANONICAL_RFC3_VERSIONS) {
    validateStructural(rfc3AxisMetadata(), undefined, version);
  }

  // ...and enforced at every other supported version, and by default.
  for (const version of NON_RFC3_VERSIONS) {
    const error = assertThrows(
      () => validateStructural(rfc3AxisMetadata(), undefined, version),
      ValidationError,
    );
    assertEquals(error.rule, SpecRule.AxisCount, String(version));
  }
});

/**
 * A repeated axis name that no *other* axis rule can catch.
 *
 * `(time "x", space "y", space "x")` satisfies all four restricted axis rules:
 * 3 axes, one `time` and two `space`, ordered time then space, and the spatial
 * names are the `(y, x)` suffix. `axis-names-unique` is therefore the only rule
 * that can fire, at every version.
 */
function repeatedNameMetadata(): Metadata {
  return {
    axes: [
      { name: "x", type: "time", unit: undefined } as Axis,
      { name: "y", type: "space", unit: undefined } as Axis,
      { name: "x", type: "space", unit: undefined } as Axis,
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          createScale([1.0, 1.0, 1.0]),
          createTranslation([0.0, 0.0, 0.0]),
        ],
      },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

Deno.test("axis-names-unique is never inert", () => {
  // RFC-3 *adds* this rule rather than lifting one, so unlike the other four
  // axis rules it fires at the RFC-3 versions too -- and at every other.
  for (const version of [...CANONICAL_RFC3_VERSIONS, ...NON_RFC3_VERSIONS]) {
    const error = assertThrows(
      () => validateStructural(repeatedNameMetadata(), undefined, version),
      ValidationError,
    );
    assertEquals(error.rule, SpecRule.AxisNamesUnique, String(version));
  }
});

// ---------------------------------------------------------------------------
// Manifest: the locked RFC-4 orientation version set
// ---------------------------------------------------------------------------

/**
 * Wrap `axes` in metadata that satisfies every non-orientation rule.
 *
 * The axis lists below are legal under both the restricted axis model and
 * RFC-3, so at every version the orientation rules alone decide the verdict.
 */
function orientationMetadata(axes: Axis[]): Metadata {
  return {
    axes,
    datasets: [
      {
        path: "0",
        coordinateTransformations: [
          createScale(axes.map(() => 1.0)),
          createTranslation(axes.map(() => 0.0)),
        ],
      },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

/**
 * Orientation on the non-spatial time axis, and nowhere else.
 *
 * Only a stray orientation violates RFC 4 here, and it sits on the one axis
 * that may not carry it -- exercising the non-space arm the orchestrator
 * reaches even when no spatial axis is oriented.
 */
function orientationOnNonSpaceAxes(): Axis[] {
  return [
    {
      name: "t",
      type: "time",
      unit: undefined,
      orientation: orientation("anatomical", "inferior-to-superior"),
    },
    { name: "y", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
  ];
}

/** Two spatial axes on the one left-right anatomical axis. */
function duplicateAnatomicalAxisAxes(): Axis[] {
  return [
    {
      name: "y",
      type: "space",
      unit: undefined,
      orientation: orientation("anatomical", "left-to-right"),
    },
    {
      name: "x",
      type: "space",
      unit: undefined,
      orientation: orientation("anatomical", "right-to-left"),
    },
  ];
}

// One violating document per orientation rule. Each satisfies every other rule
// at every version, so the orientation rule alone decides accept or reject.
const RFC4_ORIENTATION_CASES: Array<[() => Axis[], SpecRule]> = [
  [
    validAxesWithInconsistentOrientation,
    SpecRule.AxisOrientationAnatomicalType,
  ],
  [orientationOnNonSpaceAxes, SpecRule.AxisOrientationOnNonSpace],
  [duplicateAnatomicalAxisAxes, SpecRule.AxisOrientationUniqueAxis],
];

Deno.test("RFC-4 orientation version manifest is locked", () => {
  for (const [buildAxes, rule] of RFC4_ORIENTATION_CASES) {
    // Enforced at the manifest versions, and when no version is given...
    for (const version of [...CANONICAL_RFC4_VERSIONS, undefined]) {
      const error = assertThrows(
        () =>
          validateStructural(
            orientationMetadata(buildAxes()),
            undefined,
            version,
          ),
        ValidationError,
      );
      assertEquals(error.rule, rule, String(version));
    }
    // ...and inert at every earlier supported version.
    for (const version of PRE_RFC4_VERSIONS) {
      validateStructural(orientationMetadata(buildAxes()), undefined, version);
    }
  }
});

/**
 * A collection whose sibling nodes share a name, and nothing else wrong.
 *
 * Trips `node-name-unique` exactly when the RFC-8 rules run, so the
 * orchestrator accepts it exactly when they are inert.
 */
function invalidCollectionDocument(): Record<string, unknown> {
  return {
    type: "collection",
    name: "c",
    nodes: [
      { type: "multiscale", name: "img" },
      { type: "multiscale", name: "img" },
    ],
  };
}

Deno.test("RFC-8 collection version manifest is locked", () => {
  // Enforced at the manifest versions, and when no version is given...
  for (const version of [...CANONICAL_RFC8_VERSIONS, undefined]) {
    const error = assertThrows(
      () => validateCollection(invalidCollectionDocument(), version),
      ValidationError,
    );
    assertEquals(error.rule, SpecRule.NodeNameUnique, String(version));
  }
  // ...and inert at every other supported version, 0.9.dev1 included:
  // RFC-8 lands at 0.9.dev3.
  for (const version of PRE_RFC8_VERSIONS) {
    validateCollection(invalidCollectionDocument(), version);
  }
});
