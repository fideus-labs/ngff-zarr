// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Structural validation for OME-Zarr v0.4 image/multiscales metadata.
 *
 * Where the Zod schema validation in {@link validateMetadata} answers "is this
 * shaped like OME-Zarr?", the rules in this module answer "does this obey the
 * specification's structural invariants?" -- axis counts, axis ordering,
 * coordinate-transformation arity, finest-to-coarsest dataset ordering, and
 * OMERO channel color format. These are spec MUSTs that a JSON Schema cannot
 * express.
 *
 * Each rule is identified by a stable, kebab-case {@link SpecRule} value that is
 * part of the observable surface: the identifiers, their evaluation order, and
 * the dotted-segment location strings are kept byte-for-byte identical to the
 * package's Python implementation so the two stay in lockstep (a property the
 * cross-language parity tests assert). Rules fail fast -- the orchestrator
 * throws a {@link ValidationError} on the first violation, in canonical
 * specification order.
 *
 * This module provides the image/multiscales rule surface -- including the
 * RFC 4 anatomical-orientation checks -- plus the HCS plate/well rules, which
 * operate on separate metadata objects and are dispatched by the companion
 * {@link validatePlate} and {@link validateWell} entry points.
 */

import {
  type Axis,
  type Dataset,
  type Metadata,
  type Transform,
  type V06Transform,
  validateColor,
} from "../types/zarr_metadata.ts";
import type { PlateMetadata, WellMetadata } from "../types/hcs.ts";
import {
  hasRfc4OrientationMetadata,
  validateRfc4Orientation,
} from "./rfc4_validation.ts";
import { formatNameList, pyRepr } from "./py_format.ts";
import {
  isRfc3AxisModelAllowed,
  isRfc4OrientationEnforced,
  isV06Version,
} from "../types/supported_versions.ts";

/**
 * Stable, kebab-case identifiers for the structural specification rules.
 *
 * The string values are the observable surface -- they appear in
 * {@link ValidationError.rule}, logs, and tests -- and must match the package's
 * Python implementation exactly. Members are listed in canonical
 * specification-MUST evaluation order.
 */
export const SpecRule = {
  /** Axis count within the v0.4 range of 2..5 inclusive. */
  AxisCount: "axis-count",
  /** At most one `time` axis and at most one `channel` axis. */
  AxisType: "axis-type",
  /** time before channel before space; spatial names suffix (z, y, x). */
  AxisOrder: "axis-order",
  /** Axis names are unique within a dataset. */
  AxisNamesUnique: "axis-names-unique",
  /** Every scale/translation vector length equals the axis count. */
  ScaleLengthMismatch: "scale-length-mismatch",
  /** Exactly one scale per dataset; a translation must follow it. */
  GlobalCoordTransformAfterPerLevel: "global-coord-transform-after-per-level",
  /** Datasets ordered finest to coarsest; spatial scale must not shrink. */
  DatasetOrderHighestToLowest: "dataset-order-highest-to-lowest",
  /** Each OMERO channel color is exactly six hexadecimal digits. */
  OmeroChannelColorFormat: "omero-channel-color-format",
  /** Each RFC 4 orientation `type` is "anatomical" (the only value defined). */
  AxisOrientationAnatomicalType: "axis-orientation-anatomical-type",
  /** Orientation is declared only on spatial axes (never time/channel). */
  AxisOrientationOnNonSpace: "axis-orientation-on-non-space",
  /** At most one direction per anatomical axis (antonyms are exclusive). */
  AxisOrientationUniqueAxis: "axis-orientation-unique-axis",
  /** A v0.5 entry implies a Zarr v3 store: a leaked `zarr_format` must be 3. */
  ZarrFormat: "zarr-format",
  /** A v0.5 entry must not retain a group-level `ome`/`multiscales` wrapper. */
  OmeNamespace: "ome-namespace",
  /** Each well's rowIndex/columnIndex agrees with the row/column in its path. */
  PlateRowIndexConsistency: "plate-row-index-consistency",
  /** Each well image references an acquisition when the plate declares many. */
  WellAcquisitionMissing: "well-acquisition-missing",
  /** Every RFC-8 node declares a non-empty string type; from 0.9.dev3. */
  NodeTypeRequired: "node-type-required",
  /** Every RFC-8 node declares a non-empty string name; from 0.9.dev3. */
  NodeNameRequired: "node-name-required",
  /** Sibling RFC-8 nodes carry distinct names; from 0.9.dev3. */
  NodeNameUnique: "node-name-unique",
  /** RFC-8 node and reference ids match [a-zA-Z0-9-_.]+; from 0.9.dev3. */
  NodeIdFormat: "node-id-format",
  /** RFC-8 ids are unique within the JSON document; from 0.9.dev3. */
  NodeIdUnique: "node-id-unique",
  /** A collection/multiscale carries exactly one of nodes or path. */
  NodeNodesXorPath: "node-nodes-xor-path",
  /** An RFC-8 path type is zarr, json, or prefixed; from 0.9.dev3. */
  PathTypeKnown: "path-type-known",
  /** Every RFC-8 coordinate system declares an id; from 0.9.dev3. */
  CoordinateSystemIdRequired: "coordinate-system-id-required",
  /** Every RFC-8 transformation input/output is a reference with an id. */
  ReferenceIdRequired: "reference-id-required",
  /** A reference that resolves to nothing in-document carries a path. */
  ReferencePathRequired: "reference-path-required",
  /** Every RFC-8 label attributes entry declares a numeric labelValue. */
  LabelValueRequired: "label-value-required",
  /** An RFC-8 label color is four integers between 0 and 255. */
  LabelColorFormat: "label-color-format",
  /** A scene declares a non-empty coordinateTransformations array. */
  SceneTransformationsRequired: "scene-transformations-required",
  /** An RFC-8 plate declares non-empty columns and rows arrays. */
  PlateColumnsRowsRequired: "plate-columns-rows-required",
  /** A well's column/row reference an entry of the enclosing plate. */
  WellReferenceResolves: "well-reference-resolves",
  /** An acquisition reference names one the enclosing plate declares. */
  AcquisitionReferenceResolves: "acquisition-reference-resolves",
} as const;

/** Union of the {@link SpecRule} string values. */
export type SpecRule = (typeof SpecRule)[keyof typeof SpecRule];

/**
 * How much validation {@link validateStructural} performs.
 *
 * `"strict"` is the default and runs every structural image/multiscales rule.
 * `"schema_only"` skips them -- schema validation is the separate concern of
 * {@link validateMetadata}, which checks the raw attributes via Zod.
 */
export const ValidationLevel = {
  /** Skip structural rules; rely on Zod schema validation only. */
  SchemaOnly: "schema_only",
  /** Run every structural image/multiscales rule (the default). */
  Strict: "strict",
} as const;

/** Union of the {@link ValidationLevel} string values. */
export type ValidationLevel =
  (typeof ValidationLevel)[keyof typeof ValidationLevel];

/**
 * Options controlling structural validation.
 */
export interface ValidateOptions {
  /**
   * Validation level to run. Defaults to {@link ValidationLevel.Strict}
   * (`"strict"`).
   */
  level?: ValidationLevel;
  /**
   * Whether unknown metadata fields are tolerated. Defaults to `true`,
   * mirroring the parser, which warns on rather than rejects unknown fields.
   */
  allowUnknownFields?: boolean;
}

/**
 * Raised when a structural specification rule is violated.
 *
 * Carries the offending {@link SpecRule} and an optional `location` identifying
 * the offending metadata node in dotted-segment (JSON-Pointer-style) form, e.g.
 * `multiscales[0].datasets[2].coordinateTransformations[0]`. The `message` is
 * formatted as `Spec rule [<rule>] violated: <message>`.
 */
export class ValidationError extends Error {
  /** The structural rule that was violated. */
  readonly rule: SpecRule;
  /** Dotted-segment location of the offending metadata node, if known. */
  readonly location?: string;
  /**
   * The rule text without the `Spec rule [...] violated: ` prefix.
   *
   * `Error.message` is the prefixed form, matching Python's `str(exc)`; this is
   * Python's `exc.message`, for callers that supply their own prefix.
   */
  readonly detail: string;

  constructor(rule: SpecRule, message: string, location?: string) {
    super(`Spec rule [${rule}] violated: ${message}`);
    this.detail = message;
    this.name = "ValidationError";
    this.rule = rule;
    if (location !== undefined) {
      this.location = location;
    }
  }
}

/**
 * Check one coordinate transform's vector length against the axis count.
 *
 * Centralizes the length invariant shared by the global and per-dataset
 * coordinate-transformation checks (see {@link SpecRule.ScaleLengthMismatch}),
 * so the rule lives in one place. Transforms that carry no length-bearing
 * vector are ignored.
 *
 * @param transform - The coordinate transform to inspect.
 * @param axesLen - The expected vector length (the axis count).
 * @returns `["scale", actual]` or `["translation", actual]` when the
 * transform's vector length disagrees with `axesLen`; otherwise `null`.
 */
function transformLenMismatch(
  transform: Transform | V06Transform,
  axesLen: number,
): [kind: string, actual: number] | null {
  if (transform.type === "scale") {
    const actual = transform.scale.length;
    if (actual !== axesLen) {
      return ["scale", actual];
    }
  } else if (transform.type === "translation") {
    const actual = transform.translation.length;
    if (actual !== axesLen) {
      return ["translation", actual];
    }
  }
  return null;
}

/**
 * Return the first `scale` vector in a dataset, or `null` if none is present.
 *
 * Lets {@link validateDatasetOrder} compare adjacent multiscale levels without
 * re-deriving the per-dataset `scale` lookup.
 *
 * @param dataset - The dataset whose coordinate transforms are scanned.
 * @returns The first `scale` transform's vector, or `null` when the dataset
 * declares no `scale` transform.
 */
function firstScaleVector(dataset: Dataset): number[] | null {
  for (const transform of dataset.coordinateTransformations) {
    if (transform.type === "scale") {
      return transform.scale;
    }
  }
  return null;
}

/**
 * Class-ordering rank for axis types: a `time` axis must precede a `channel`
 * axis, which must precede any `space` axis. Lower rank must never follow
 * higher rank (see {@link validateAxisOrder}). Axis types absent from this map
 * (e.g. `array`) carry no rank, so a pair involving one is skipped -- mirroring
 * the Python implementation's `dict.get(...) is None` handling.
 */
const AXIS_TYPE_RANK: Record<string, number | undefined> = {
  time: 0,
  channel: 1,
  space: 2,
};

/**
 * Spatial axis names in canonical order. A valid set of `space` axis names is
 * the matching-length suffix of this tuple: 1 -> `["x"]`, 2 -> `["y", "x"]`,
 * 3 -> `["z", "y", "x"]` (see {@link validateSpatialAxisOrder}).
 */
const SPATIAL_AXIS_NAMES = ["z", "y", "x"] as const;

/**
 * The v0.6 `axes` schema is a `oneOf`: either 2 or 3 `space` axes, or two or
 * more `array` axes. An RFC-5 array coordinate system takes the second branch
 * and declares no `space` axis at all.
 */
const MIN_ARRAY_AXES = 2;

/**
 * Whether `axes` satisfies the `array` arm of the v0.6 axes schema.
 *
 * Only v0.6 has that arm; the v0.4 and v0.5 schemas require 2 or 3 `space`
 * axes unconditionally, so the space-axis floor is not relaxed for them.
 */
function takesArraySchemaBranch(
  axes: Pick<Axis, "type">[],
  version?: string,
): boolean {
  if (version === undefined || !isV06Version(version)) {
    return false;
  }
  return axes.filter((ax) => ax.type === "array").length >= MIN_ARRAY_AXES;
}

/**
 * Render a number the way Python's `str()`/f-string renders a `float`: a
 * whole-number value keeps a trailing `.0` (e.g. `2` -> `"2.0"`), matching the
 * `list[float]` scale vectors the Python port interpolates into the
 * dataset-order message. Non-integer scale ratios (e.g. `1.5`) already render
 * identically across both languages, so they pass through unchanged. This keeps
 * the message byte-for-byte identical to Python for the whole-number
 * downsampling ratios OME-Zarr writers (including this package) emit to disk.
 */
function pyFloat(value: number): string {
  return Number.isInteger(value) ? `${value}.0` : `${value}`;
}

/**
 * Validate that the axis count is within the v0.4-permitted range.
 *
 * OME-Zarr v0.4, v0.5 and v0.6 require between 2 and 5 axes, inclusive.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.AxisCount} when
 * `metadata.axes.length` lies outside `2..5`; location `multiscales[0].axes`.
 */
export function validateAxisCount(
  metadata: Pick<Metadata, "axes">,
  version?: string,
): void {
  if (isRfc3AxisModelAllowed(version)) {
    return;
  }
  const count = metadata.axes.length;
  if (count < 2 || count > 5) {
    throw new ValidationError(
      SpecRule.AxisCount,
      `OME-Zarr v0.4, v0.5 and v0.6 require between 2 and 5 axes, ` +
        `inclusive; found ${count}.`,
      "multiscales[0].axes",
    );
  }
}

/**
 * Validate axis-type multiplicity.
 *
 * At most one `time` axis and at most one `channel` axis may be present.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.AxisType} when more than one
 * `time` axis or more than one `channel` axis is present; location
 * `multiscales[0].axes`.
 */
export function validateAxisType(
  metadata: Pick<Metadata, "axes">,
  version?: string,
): void {
  if (isRfc3AxisModelAllowed(version)) {
    return;
  }
  const timeCount = metadata.axes.filter((ax) => ax.type === "time").length;
  if (timeCount > 1) {
    throw new ValidationError(
      SpecRule.AxisType,
      `At most one 'time' axis is permitted; found ${timeCount}.`,
      "multiscales[0].axes",
    );
  }
  const channelCount =
    metadata.axes.filter((ax) => ax.type === "channel").length;
  if (channelCount > 1) {
    throw new ValidationError(
      SpecRule.AxisType,
      `At most one 'channel' axis is permitted; found ${channelCount}.`,
      "multiscales[0].axes",
    );
  }
}

/**
 * Validate the class ordering of axes.
 *
 * Axes are ranked by type (`time` < `channel` < `space`) and must be listed in
 * non-decreasing rank order: every `time` axis precedes every `channel` axis,
 * which precedes every `space` axis. The first adjacent pair that inverts this
 * ranking is reported. Pairs involving an axis type with no rank (e.g. `array`)
 * are skipped.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.AxisOrder} for the first
 * adjacent pair where a lower-ranked axis type follows a higher-ranked one;
 * location `multiscales[0].axes[i+1]`.
 */
export function validateAxisOrder(
  metadata: Pick<Metadata, "axes">,
  version?: string,
): void {
  if (isRfc3AxisModelAllowed(version)) {
    return;
  }
  const axes = metadata.axes;
  for (let i = 0; i < axes.length - 1; i++) {
    const current = axes[i];
    const following = axes[i + 1];
    // An axis with no `type` has no rank, mirroring the Python port's dict
    // `.get()`; the pair is then skipped by the guard below.
    const currentRank = current.type === undefined
      ? undefined
      : AXIS_TYPE_RANK[current.type];
    const followingRank = following.type === undefined
      ? undefined
      : AXIS_TYPE_RANK[following.type];
    if (currentRank === undefined || followingRank === undefined) {
      continue;
    }
    if (followingRank < currentRank) {
      throw new ValidationError(
        SpecRule.AxisOrder,
        `Axis '${following.name}' of type '${following.type}' must not ` +
          `follow axis '${current.name}' of type '${current.type}'; axes ` +
          `must be ordered time, then channel, then space.`,
        `multiscales[0].axes[${i + 1}]`,
      );
    }
  }
}

/**
 * Validate the count and names of spatial axes.
 *
 * The `space` axes, taken in order, must be the matching-length suffix of
 * `(z, y, x)`: one spatial axis must be `(x)`, two must be `(y, x)`, and three
 * must be `(z, y, x)`. At most three `space` axes are permitted.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.AxisOrder} when there are more
 * than three `space` axes, or when their names are not the expected suffix of
 * `(z, y, x)`; location `multiscales[0].axes` or the first offending axis.
 */
export function validateSpatialAxisOrder(
  metadata: Pick<Metadata, "axes">,
  version?: string,
): void {
  if (isRfc3AxisModelAllowed(version)) {
    return;
  }
  const spaceIndices: number[] = [];
  metadata.axes.forEach((ax, i) => {
    if (ax.type === "space") {
      spaceIndices.push(i);
    }
  });
  const count = spaceIndices.length;
  if (count > 3) {
    throw new ValidationError(
      SpecRule.AxisOrder,
      `OME-Zarr v0.4, v0.5 and v0.6 permit at most 3 'space' axes; ` +
        `found ${count}.`,
      "multiscales[0].axes",
    );
  }
  if (count < 2 && !takesArraySchemaBranch(metadata.axes, version)) {
    throw new ValidationError(
      SpecRule.AxisOrder,
      `OME-Zarr v0.4, v0.5 and v0.6 require 2 or 3 'space' axes; ` +
        `found ${count}.`,
      "multiscales[0].axes",
    );
  }
  const expected = SPATIAL_AXIS_NAMES.slice(SPATIAL_AXIS_NAMES.length - count);
  const actual = spaceIndices.map((i) => metadata.axes[i].name);
  const mismatch = actual.findIndex((name, j) => name !== expected[j]);
  if (mismatch !== -1) {
    throw new ValidationError(
      SpecRule.AxisOrder,
      `Spatial axis names ${formatNameList(actual)} must be the ` +
        `length-${count} suffix of (z, y, x): ${formatNameList(expected)}.`,
      `multiscales[0].axes[${spaceIndices[mismatch]}]`,
    );
  }
}

/**
 * Validate that axis names are unique within the dataset.
 *
 * Never inert. It states RFC-3 rule 5, "axis names MUST NOT be repeated within
 * a dataset". No released schema carries it, so below `0.9.dev1` it is a
 * strictness choice rather than a spec MUST of those versions. Rule 5 also says
 * names SHOULD NOT differ only by case. That is a SHOULD and is not enforced.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.AxisNamesUnique} for the first
 * axis whose `name` duplicates an earlier one; location
 * `multiscales[0].axes[i]`.
 */
export function validateAxisNamesUnique(
  metadata: Pick<Metadata, "axes">,
  _version?: string,
): void {
  const seen = new Set<string>();
  for (let i = 0; i < metadata.axes.length; i++) {
    const name = metadata.axes[i].name;
    if (seen.has(name)) {
      throw new ValidationError(
        SpecRule.AxisNamesUnique,
        `Axis name '${name}' is repeated; axis names must be unique ` +
          `within a dataset.`,
        `multiscales[0].axes[${i}]`,
      );
    }
    seen.add(name);
  }
}

/**
 * Validate that each dataset defines exactly one `scale` transform.
 *
 * Every dataset's `coordinateTransformations` must contain exactly one `scale`
 * (the per-level voxel size). This shares the per-dataset
 * coordinate-transform-shape rule identifier
 * ({@link SpecRule.GlobalCoordTransformAfterPerLevel}); there is no dedicated
 * identifier for the scale count.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With
 * {@link SpecRule.GlobalCoordTransformAfterPerLevel} when a dataset has zero or
 * more than one `scale` transform; location
 * `multiscales[0].datasets[i].coordinateTransformations`.
 */
export function validatePerDatasetScaleCount(metadata: Metadata): void {
  for (let i = 0; i < metadata.datasets.length; i++) {
    const dataset = metadata.datasets[i];
    const scaleCount = dataset.coordinateTransformations.filter(
      (t) => t.type === "scale",
    ).length;
    if (scaleCount !== 1) {
      throw new ValidationError(
        SpecRule.GlobalCoordTransformAfterPerLevel,
        `Each dataset must define exactly one 'scale' coordinate ` +
          `transformation; dataset ${i} has ${scaleCount}.`,
        `multiscales[0].datasets[${i}].coordinateTransformations`,
      );
    }
  }
}

/**
 * Validate coordinate-transform vector lengths against the axis count.
 *
 * Every `scale` and `translation` vector -- in the global
 * `coordinateTransformations` (if present) and in each dataset -- must have
 * exactly one entry per axis. The first mismatch, scanned
 * global-then-per-dataset, is reported. The length invariant itself lives in
 * {@link transformLenMismatch}.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.ScaleLengthMismatch} for the
 * first transform whose vector length disagrees with `metadata.axes.length`;
 * location identifies the offending transform.
 */
export function validateScaleLength(metadata: Metadata): void {
  const axesLen = metadata.axes.length;
  const globalTransforms = metadata.coordinateTransformations;
  if (globalTransforms) {
    for (let j = 0; j < globalTransforms.length; j++) {
      const mismatch = transformLenMismatch(globalTransforms[j], axesLen);
      if (mismatch !== null) {
        const [kind, actualLen] = mismatch;
        throw new ValidationError(
          SpecRule.ScaleLengthMismatch,
          `Global '${kind}' transform has length ${actualLen}, but ` +
            `there are ${axesLen} axes; lengths must match.`,
          `multiscales[0].coordinateTransformations[${j}]`,
        );
      }
    }
  }
  for (let i = 0; i < metadata.datasets.length; i++) {
    const dataset = metadata.datasets[i];
    for (let j = 0; j < dataset.coordinateTransformations.length; j++) {
      const mismatch = transformLenMismatch(
        dataset.coordinateTransformations[j],
        axesLen,
      );
      if (mismatch !== null) {
        const [kind, actualLen] = mismatch;
        throw new ValidationError(
          SpecRule.ScaleLengthMismatch,
          `Dataset ${i} '${kind}' transform has length ${actualLen}, ` +
            `but there are ${axesLen} axes; lengths must match.`,
          `multiscales[0].datasets[${i}].coordinateTransformations[${j}]`,
        );
      }
    }
  }
}

/**
 * Validate coordinate-transform ordering within each dataset.
 *
 * Within a dataset's `coordinateTransformations`, a `translation` (if present)
 * must follow its `scale` -- a `scale` must never appear after a `translation`.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With
 * {@link SpecRule.GlobalCoordTransformAfterPerLevel} for the first dataset
 * where a `scale` follows a `translation`; location identifies the offending
 * `scale`.
 */
export function validateTransformOrder(metadata: Metadata): void {
  for (let i = 0; i < metadata.datasets.length; i++) {
    const dataset = metadata.datasets[i];
    let seenTranslation = false;
    for (let j = 0; j < dataset.coordinateTransformations.length; j++) {
      const transform = dataset.coordinateTransformations[j];
      if (transform.type === "translation") {
        seenTranslation = true;
      } else if (transform.type === "scale" && seenTranslation) {
        throw new ValidationError(
          SpecRule.GlobalCoordTransformAfterPerLevel,
          `In dataset ${i}, a 'scale' transform must not follow a ` +
            `'translation'; a translation must follow its scale.`,
          `multiscales[0].datasets[${i}].coordinateTransformations[${j}]`,
        );
      }
    }
  }
}

/**
 * Validate that datasets are ordered finest to coarsest.
 *
 * Multiscale datasets must be listed from highest resolution (finest, i.e.
 * smallest spatial `scale`) to lowest (coarsest). For each adjacent pair, the
 * later level's `scale` must not be smaller than the earlier level's on any
 * `space` axis. Pairs whose `scale` vectors are missing or too short to index a
 * spatial axis are skipped -- those shapes are the concern of
 * {@link validatePerDatasetScaleCount} and {@link validateScaleLength} -- so
 * this rule never indexes out of bounds.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.DatasetOrderHighestToLowest}
 * for the first adjacent pair whose later (coarser) level has a smaller spatial
 * scale; location `multiscales[0].datasets[i+1]`.
 */
export function validateDatasetOrder(metadata: Metadata): void {
  const spaceIndices: number[] = [];
  metadata.axes.forEach((ax, i) => {
    if (ax.type === "space") {
      spaceIndices.push(i);
    }
  });
  const datasets = metadata.datasets;
  for (let i = 0; i < datasets.length - 1; i++) {
    const finer = firstScaleVector(datasets[i]);
    const coarser = firstScaleVector(datasets[i + 1]);
    if (finer === null || coarser === null) {
      continue;
    }
    for (const axisIndex of spaceIndices) {
      if (axisIndex >= finer.length || axisIndex >= coarser.length) {
        // Scale too short to compare this axis: another rule's concern.
        // Skip so we never index out of bounds here.
        continue;
      }
      if (coarser[axisIndex] < finer[axisIndex]) {
        const axisName = metadata.axes[axisIndex].name;
        throw new ValidationError(
          SpecRule.DatasetOrderHighestToLowest,
          `Datasets must be ordered finest to coarsest; dataset ` +
            `${i + 1} has spatial scale ${pyFloat(coarser[axisIndex])} ` +
            `on axis '${axisName}', smaller than dataset ${i}'s ` +
            `${pyFloat(finer[axisIndex])}.`,
          `multiscales[0].datasets[${i + 1}]`,
        );
      }
    }
  }
}

/**
 * Validate the color format of each OMERO channel.
 *
 * When OMERO rendering metadata is present, every channel `color` must be
 * exactly six hexadecimal digits (RGB). This reuses the package's existing
 * {@link validateColor} predicate rather than re-implementing the format check,
 * surfacing its failure message through the unified {@link ValidationError}
 * channel.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.OmeroChannelColorFormat} for
 * the first channel whose `color` is not six hex digits; location
 * `multiscales[0].omero.channels[i].color`.
 */
export function validateOmeroColorHex(metadata: Metadata): void {
  const omero = metadata.omero;
  if (omero === undefined) {
    return;
  }
  for (let i = 0; i < omero.channels.length; i++) {
    try {
      validateColor(omero.channels[i].color);
    } catch (error) {
      throw new ValidationError(
        SpecRule.OmeroChannelColorFormat,
        error instanceof Error ? error.message : String(error),
        `multiscales[0].omero.channels[${i}].color`,
      );
    }
  }
}

/**
 * Render a parsed {@link Axis} back to the record form the RFC 4 helpers
 * consume.
 *
 * Emits only the fields the RFC 4 helpers inspect -- `name`, `type`, and (for a
 * spatial axis that declares it) `orientation` as a `{ type, value }` record. A
 * stored `orientation` may be the serialized {@link AxisOrientation} form or
 * an {@link AnatomicalOrientation}; both already expose a string `type` and
 * `value`, so the record is built directly without further coercion.
 *
 * @param axis - The parsed axis to render.
 * @returns A plain record carrying `name`, `type`, and, when present,
 * `orientation`.
 */
function axisToValidationRecord(axis: Axis): Record<string, unknown> {
  const record: Record<string, unknown> = {
    name: axis.name,
    type: axis.type,
  };
  const orientation = axis.orientation;
  // An empty orientation object is undefined under RFC 4, exactly like an absent
  // one, so it must not be rendered as a populated { type, value } object -- that
  // would read back as a real orientation.
  if (
    orientation !== undefined &&
    orientation !== null &&
    Object.keys(orientation).length > 0
  ) {
    record.orientation = {
      type: orientation.type,
      value: orientation.value,
    };
  }
  return record;
}

/**
 * Validate RFC 4 anatomical-orientation metadata on the spatial axes.
 *
 * This rule does not reimplement RFC 4; it wraps the package's existing
 * logic -- {@link hasRfc4OrientationMetadata} and
 * {@link validateRfc4Orientation} -- and surfaces its failures through the
 * unified {@link ValidationError} channel. The parsed {@link Axis} objects are
 * first rendered back to the record form those helpers expect (see
 * {@link axisToValidationRecord}).
 *
 * Orientation is optional in RFC 4, so when no spatial axis carries it the rule
 * is a no-op.
 *
 * Inert when `version` declares a release below 0.9.dev1, where RFC-4 has no
 * normative status; `undefined` keeps the checks on (see
 * {@link isRfc4OrientationEnforced}).
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @param version - The OME-Zarr version the metadata declares.
 * @throws {ValidationError} With {@link SpecRule.AxisOrientationAnatomicalType}
 * when an orientation `type` is not `"anatomical"`,
 * {@link SpecRule.AxisOrientationOnNonSpace} when
 * orientation is declared on a non-spatial axis, or
 * {@link SpecRule.AxisOrientationUniqueAxis} when two spatial axes describe the
 * same anatomical axis; location `multiscales[0].axes`. The original
 * RFC 4 message text is preserved. Any other failure from
 * {@link validateRfc4Orientation} -- e.g. an orientation value outside the
 * RFC 4 vocabulary, a schema-level concern with no dedicated structural rule
 * -- propagates unchanged.
 */
export function validateAxisOrientation(
  metadata: Metadata,
  version?: string,
): void {
  if (!isRfc4OrientationEnforced(version)) {
    return;
  }
  const axesRecords = metadata.axes.map(axisToValidationRecord);
  // A stray orientation on a non-spatial axis carries no spatial-axis
  // orientation, so hasRfc4OrientationMetadata alone would skip it; check for any
  // real (non-empty) orientation so the non-space rule is reachable. A null or
  // empty orientation is undefined under RFC 4 and so is not a violation.
  const nonSpaceOrientation = axesRecords.some(
    (axisRecord) =>
      typeof axisRecord === "object" &&
      axisRecord !== null &&
      axisRecord.type !== "space" &&
      typeof axisRecord.orientation === "object" &&
      axisRecord.orientation !== null &&
      Object.keys(axisRecord.orientation as Record<string, unknown>).length > 0,
  );
  if (!hasRfc4OrientationMetadata(axesRecords) && !nonSpaceOrientation) {
    return;
  }
  try {
    validateRfc4Orientation(axesRecords);
  } catch (error) {
    // validateRfc4Orientation throws three messages this rule maps: an
    // orientation type that is not "anatomical", orientation on "non-space
    // axes", and two axes on the "same anatomical axis". An out-of-vocabulary
    // orientation value -- a schema-level concern with no dedicated structural
    // rule -- carries none of these markers and so propagates unchanged,
    // matching the package's Python implementation.
    //
    // The marker substrings below ("must be anatomical", "non-space axes",
    // "same anatomical axis") are a load-bearing contract with
    // validateRfc4Orientation's message text: an unrecognized error falls
    // through to the final `throw` and surfaces as a plain Error rather than a
    // ValidationError. The message-stability test in
    // structural_validation_orientation_test.ts pins the markers so editing that
    // wording fails CI loudly instead of silently breaking this mapping.
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes("must be anatomical")) {
      throw new ValidationError(
        SpecRule.AxisOrientationAnatomicalType,
        message,
        "multiscales[0].axes",
      );
    }
    if (message.includes("non-space axes")) {
      throw new ValidationError(
        SpecRule.AxisOrientationOnNonSpace,
        message,
        "multiscales[0].axes",
      );
    }
    if (message.includes("same anatomical axis")) {
      throw new ValidationError(
        SpecRule.AxisOrientationUniqueAxis,
        message,
        "multiscales[0].axes",
      );
    }
    throw error;
  }
}

/**
 * Validate that a v0.5 multiscales entry implies a Zarr v3 store.
 *
 * OME-Zarr v0.5 metadata MUST be backed by a Zarr v3 store
 * (`zarr_format === 3`). A v0.4 dataset is always Zarr v2 and carries no
 * `zarr_format`, so this rule is inert below v0.5. The parsed {@link Metadata}
 * does not surface the group's `zarr_format` byte -- its authoritative on-disk
 * verification is a store-backed concern -- but a `zarr_format` mis-embedded
 * *inside* the multiscale entry (captured in {@link Metadata.extra}) is
 * reachable here: it belongs on the enclosing Zarr v3 group, never on the entry,
 * and any value other than `3` is incompatible with v0.5.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.ZarrFormat} when v0.5 metadata
 * carries a `zarr_format` other than `3` in its `extra` passthrough; location
 * `multiscales[0]`.
 */
export function validateZarrFormatForVersion(metadata: Metadata): void {
  if (metadata.version !== "0.5") {
    return;
  }
  const zarrFormat = metadata.extra?.zarr_format;
  if (zarrFormat !== undefined && zarrFormat !== 3) {
    throw new ValidationError(
      SpecRule.ZarrFormat,
      `OME-Zarr v0.5 requires a Zarr v3 store (zarr_format == 3), but ` +
        `the entry declares zarr_format = ${String(zarrFormat)}.`,
      "multiscales[0]",
    );
  }
}

/**
 * Validate that a v0.5 multiscales entry is not double-namespaced.
 *
 * At v0.5 the multiscales metadata lives under the top-level `ome` namespace key
 * of the group's `zarr.json` attributes, with the spec `version` hoisted to
 * `ome.version`. The reader unwraps that namespace into this {@link Metadata}, so
 * the group-level wrapper keys -- `ome` itself and the `multiscales` array --
 * must never reappear *inside* a multiscale entry. A surviving wrapper key
 * (captured in {@link Metadata.extra}) means the document is double-namespaced,
 * was not unwrapped, or is otherwise malformed with respect to the v0.5 `ome`
 * namespacing. A v0.4 store keeps multiscales at the `.zattrs` top level with no
 * `ome` wrapper, so this rule is inert below v0.5.
 *
 * @param metadata - The parsed multiscales metadata to validate.
 * @throws {ValidationError} With {@link SpecRule.OmeNamespace} when v0.5 metadata
 * retains a group-level `ome` or `multiscales` key in its `extra` passthrough;
 * location `multiscales[0]`.
 */
export function validateOmeNamespace(metadata: Metadata): void {
  if (metadata.version !== "0.5") {
    return;
  }
  const extra = metadata.extra ?? {};
  for (const wrapperKey of ["ome", "multiscales"]) {
    if (wrapperKey in extra) {
      throw new ValidationError(
        SpecRule.OmeNamespace,
        `v0.5 metadata entry retains the group-level '${wrapperKey}' ` +
          `key; the 'ome' namespace wraps the group attributes, not each ` +
          `multiscale entry.`,
        "multiscales[0]",
      );
    }
  }
}

/**
 * Validate each plate well's recorded indices against its `path`.
 *
 * Every `PlateWell` records a `path` of the form `"<row>/<column>"` (e.g.
 * `"B/03"`) alongside numeric `rowIndex` and `columnIndex` fields. OME-Zarr v0.4
 * requires these to agree: the row segment must name an entry in `plate.rows`
 * whose position equals `rowIndex`, and the column segment must name an entry in
 * `plate.columns` whose position equals `columnIndex`. The first offending well,
 * scanned in order, is reported.
 *
 * @param plate - The plate metadata whose wells are validated.
 * @throws {ValidationError} With {@link SpecRule.PlateRowIndexConsistency} for
 * the first well whose `path` is not `"<row>/<column>"`, whose row/column
 * segment is absent from `plate.rows`/`plate.columns`, or whose recorded
 * `rowIndex`/`columnIndex` disagrees with that segment's position; location
 * `plate.wells[i]`.
 */
export function validatePlateWellIndexConsistency(plate: PlateMetadata): void {
  const rowIndexByName = new Map<string, number>();
  plate.rows.forEach((row, i) => rowIndexByName.set(row.name, i));
  const columnIndexByName = new Map<string, number>();
  plate.columns.forEach((column, i) => columnIndexByName.set(column.name, i));
  for (let i = 0; i < plate.wells.length; i++) {
    const well = plate.wells[i];
    const location = `plate.wells[${i}]`;
    const parts = well.path.split("/");
    if (parts.length !== 2) {
      throw new ValidationError(
        SpecRule.PlateRowIndexConsistency,
        `Well path ${pyRepr(well.path)} must have the form '<row>/<column>'.`,
        location,
      );
    }
    const [rowName, columnName] = parts;
    const expectedRow = rowIndexByName.get(rowName);
    if (expectedRow === undefined) {
      throw new ValidationError(
        SpecRule.PlateRowIndexConsistency,
        `Well path ${pyRepr(well.path)} names row ${pyRepr(rowName)}, which ` +
          `is not declared in plate.rows.`,
        location,
      );
    }
    const expectedColumn = columnIndexByName.get(columnName);
    if (expectedColumn === undefined) {
      throw new ValidationError(
        SpecRule.PlateRowIndexConsistency,
        `Well path ${pyRepr(well.path)} names column ${pyRepr(columnName)}, ` +
          `which is not declared in plate.columns.`,
        location,
      );
    }
    if (well.rowIndex !== expectedRow) {
      throw new ValidationError(
        SpecRule.PlateRowIndexConsistency,
        `Well ${pyRepr(well.path)} has rowIndex ${well.rowIndex}, but row ` +
          `${pyRepr(rowName)} is at index ${expectedRow} in plate.rows.`,
        location,
      );
    }
    if (well.columnIndex !== expectedColumn) {
      throw new ValidationError(
        SpecRule.PlateRowIndexConsistency,
        `Well ${pyRepr(well.path)} has columnIndex ${well.columnIndex}, but ` +
          `column ${pyRepr(columnName)} is at index ${expectedColumn} in ` +
          `plate.columns.`,
        location,
      );
    }
  }
}

/**
 * Validate that well images reference an acquisition when required.
 *
 * When a plate declares more than one acquisition, every image in each of its
 * wells must carry an `acquisition` reference so the field can be attributed to
 * a specific acquisition. Plates with zero or one acquisition impose no such
 * requirement, so this rule is a no-op for them. The first image missing its
 * reference, scanned in order, is reported.
 *
 * @param plate - The plate metadata declaring the acquisitions.
 * @param well - The well metadata whose images are validated.
 * @throws {ValidationError} With {@link SpecRule.WellAcquisitionMissing} for the
 * first `WellImage` whose `acquisition` is absent while the plate declares
 * multiple acquisitions; location `well.images[i].acquisition`.
 */
export function validateWellAcquisition(
  plate: PlateMetadata,
  well: WellMetadata,
): void {
  const acquisitions = plate.acquisitions;
  if (acquisitions === undefined || acquisitions.length <= 1) {
    return;
  }
  for (let i = 0; i < well.images.length; i++) {
    const image = well.images[i];
    if (image.acquisition === undefined || image.acquisition === null) {
      throw new ValidationError(
        SpecRule.WellAcquisitionMissing,
        `Plate declares ${acquisitions.length} acquisitions, so well ` +
          `image ${i} must reference one via 'acquisition'.`,
        `well.images[${i}].acquisition`,
      );
    }
  }
}

/**
 * Run the structural image/multiscales rules in canonical specification order.
 *
 * Orchestrates the per-rule `validate*` functions in this module. It is
 * fail-fast: the rules run in canonical specification-MUST order and the first
 * {@link ValidationError} thrown propagates to the caller -- later rules do not
 * run. The rules run in this order:
 *
 * 1. {@link validateAxisCount}
 * 2. {@link validateAxisType}
 * 3. {@link validateAxisOrder}
 * 4. {@link validateSpatialAxisOrder}
 * 5. {@link validateAxisNamesUnique}
 * 6. {@link validatePerDatasetScaleCount}
 * 7. {@link validateScaleLength}
 * 8. {@link validateTransformOrder}
 * 9. {@link validateDatasetOrder}
 * 10. {@link validateOmeroColorHex}
 * 11. {@link validateAxisOrientation}
 * 12. {@link validateZarrFormatForVersion}
 * 13. {@link validateOmeNamespace}
 *
 * Rules 12 and 13 are the OME-Zarr v0.5 namespacing checks; they fire only for
 * v0.5 metadata and are inert (a no-op) for v0.4.
 *
 * Under {@link ValidationLevel.SchemaOnly} this function returns immediately
 * without running any structural rule -- shape/schema validation is the
 * separate concern of {@link validateMetadata}, which checks the raw
 * attributes via Zod. The default level is {@link ValidationLevel.Strict}.
 *
 * This orchestrator covers the image/multiscales rules, including the RFC 4
 * anatomical-orientation checks ({@link validateAxisOrientation}, a no-op when
 * no axis declares orientation and inert when `version` declares a release
 * below 0.9.dev1). The HCS plate/well structural rules operate on
 * separate metadata objects and are dispatched by the companion
 * {@link validatePlate} and {@link validateWell} entry points.
 *
 * @param metadata - The parsed OME-Zarr v0.4 multiscales metadata to validate.
 * @param options - Validation options; defaults to
 * `{ level: "strict", allowUnknownFields: true }`.
 * @param version - The OME-Zarr version the metadata declares. Rules 1-3 are
 * inert for the versions that adopt the RFC-3 axis model, so omitting it holds
 * every store to the v0.4 axis caps. The RFC 4 orientation checks
 * ({@link validateAxisOrientation}) gate the opposite way: they are normative
 * from 0.9.dev1, inert when an earlier version is declared, and kept on when
 * the version is omitted.
 * @throws {ValidationError} For the first structural rule violated, carrying
 * the offending {@link SpecRule} and `location`. Never thrown under
 * {@link ValidationLevel.SchemaOnly}, which runs no structural rule.
 */
export function validateStructural(
  metadata: Metadata,
  options?: ValidateOptions,
  version?: string,
): void {
  const resolved: Required<ValidateOptions> = {
    level: options?.level ?? ValidationLevel.Strict,
    allowUnknownFields: options?.allowUnknownFields ?? true,
  };
  if (resolved.level === ValidationLevel.SchemaOnly) {
    return;
  }
  validateAxisCount(metadata, version);
  validateAxisType(metadata, version);
  validateAxisOrder(metadata, version);
  validateSpatialAxisOrder(metadata, version);
  validateAxisNamesUnique(metadata, version);
  validatePerDatasetScaleCount(metadata);
  validateScaleLength(metadata);
  validateTransformOrder(metadata);
  validateDatasetOrder(metadata);
  validateOmeroColorHex(metadata);
  validateAxisOrientation(metadata, version);
  validateZarrFormatForVersion(metadata);
  validateOmeNamespace(metadata);
}

/**
 * Run the structural HCS plate rules in canonical specification order.
 *
 * The plate-level counterpart to {@link validateStructural}: it orchestrates
 * the structural rules that operate on a parsed {@link PlateMetadata}. Like its
 * image/multiscales sibling it is fail-fast -- the first {@link ValidationError}
 * thrown propagates to the caller and later rules do not run. The rules run in
 * this order:
 *
 * 1. {@link validatePlateWellIndexConsistency}
 *
 * Per-well image rules (e.g. acquisition references) are the separate concern
 * of {@link validateWell}, called where each well's own metadata is loaded.
 *
 * Under {@link ValidationLevel.SchemaOnly} this function returns immediately
 * without running any structural rule -- shape/schema validation is the
 * separate concern of {@link validateMetadata}, which checks the raw attributes
 * via Zod. The default level is {@link ValidationLevel.Strict}.
 *
 * @param plate - The parsed OME-Zarr v0.4 plate metadata to validate.
 * @param options - Validation options; defaults to
 * `{ level: "strict", allowUnknownFields: true }`.
 * @throws {ValidationError} For the first structural plate rule violated,
 * carrying the offending {@link SpecRule} and `location`. Never thrown under
 * {@link ValidationLevel.SchemaOnly}, which runs no structural rule.
 */
export function validatePlate(
  plate: PlateMetadata,
  options?: ValidateOptions,
): void {
  const resolved: Required<ValidateOptions> = {
    level: options?.level ?? ValidationLevel.Strict,
    allowUnknownFields: options?.allowUnknownFields ?? true,
  };
  if (resolved.level === ValidationLevel.SchemaOnly) {
    return;
  }
  validatePlateWellIndexConsistency(plate);
}

/**
 * Run the structural HCS well rules in canonical specification order.
 *
 * Validates a single {@link WellMetadata} in the context of its parent
 * {@link PlateMetadata} -- the plate's acquisition declarations govern whether
 * each well image must reference an acquisition. Fail-fast, like
 * {@link validateStructural} and {@link validatePlate}. The rules run in this
 * order:
 *
 * 1. {@link validateWellAcquisition}
 *
 * Under {@link ValidationLevel.SchemaOnly} this function returns immediately
 * without running any structural rule -- shape/schema validation is the
 * separate concern of {@link validateMetadata}, which checks the raw attributes
 * via Zod. The default level is {@link ValidationLevel.Strict}.
 *
 * @param plate - The parsed plate metadata that owns `well`; supplies the
 * acquisition context the well rules consult.
 * @param well - The parsed well metadata to validate.
 * @param options - Validation options; defaults to
 * `{ level: "strict", allowUnknownFields: true }`.
 * @throws {ValidationError} For the first structural well rule violated,
 * carrying the offending {@link SpecRule} and `location`. Never thrown under
 * {@link ValidationLevel.SchemaOnly}, which runs no structural rule.
 */
export function validateWell(
  plate: PlateMetadata,
  well: WellMetadata,
  options?: ValidateOptions,
): void {
  const resolved: Required<ValidateOptions> = {
    level: options?.level ?? ValidationLevel.Strict,
    allowUnknownFields: options?.allowUnknownFields ?? true,
  };
  if (resolved.level === ValidationLevel.SchemaOnly) {
    return;
  }
  validateWellAcquisition(plate, well);
}
