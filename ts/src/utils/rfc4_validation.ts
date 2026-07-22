// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * RFC 4 validation for anatomical orientation in OME-NGFF.
 *
 * This module provides validation for RFC 4 anatomical orientation metadata.
 */

import { AnatomicalOrientationValuesSchema } from "../schemas/rfc4.ts";
import { formatNameList } from "./py_format.ts";

/** Valid anatomical orientation values */
const VALID_ORIENTATION_VALUES = new Set([
  "left-to-right",
  "right-to-left",
  "anterior-to-posterior",
  "posterior-to-anterior",
  "inferior-to-superior",
  "superior-to-inferior",
  "dorsal-to-ventral",
  "ventral-to-dorsal",
  "dorsal-to-palmar",
  "palmar-to-dorsal",
  "dorsal-to-plantar",
  "plantar-to-dorsal",
  "rostral-to-caudal",
  "caudal-to-rostral",
  "cranial-to-caudal",
  "caudal-to-cranial",
  "proximal-to-distal",
  "distal-to-proximal",
  "superficial-to-deep",
  "deep-to-superficial",
  "apical-to-basal",
  "basal-to-apical",
  "apex-to-base",
  "base-to-apex",
]);

/**
 * Each RFC 4 value maps to a stable key for the anatomical axis it lies on, so
 * the two antonyms of a pair (e.g. "left-to-right" / "right-to-left") collapse to
 * one key. Used to enforce "one direction per anatomical axis" (mutual exclusion).
 */
const ANATOMICAL_AXIS_OF: Record<string, string> = {
  "left-to-right": "left-right",
  "right-to-left": "left-right",
  "anterior-to-posterior": "anterior-posterior",
  "posterior-to-anterior": "anterior-posterior",
  "inferior-to-superior": "inferior-superior",
  "superior-to-inferior": "inferior-superior",
  "dorsal-to-ventral": "dorsal-ventral",
  "ventral-to-dorsal": "dorsal-ventral",
  "dorsal-to-palmar": "dorsal-palmar",
  "palmar-to-dorsal": "dorsal-palmar",
  "dorsal-to-plantar": "dorsal-plantar",
  "plantar-to-dorsal": "dorsal-plantar",
  "rostral-to-caudal": "rostral-caudal",
  "caudal-to-rostral": "rostral-caudal",
  "cranial-to-caudal": "cranial-caudal",
  "caudal-to-cranial": "cranial-caudal",
  "proximal-to-distal": "proximal-distal",
  "distal-to-proximal": "proximal-distal",
  "superficial-to-deep": "superficial-deep",
  "deep-to-superficial": "superficial-deep",
  "apical-to-basal": "apical-basal",
  "basal-to-apical": "apical-basal",
  "apex-to-base": "apex-base",
  "base-to-apex": "apex-base",
};

/**
 * Check if the axes contain RFC 4 anatomical orientation metadata.
 *
 * @param axes - List of axis metadata objects
 * @returns True if any spatial axis has orientation metadata
 */
export function hasRfc4OrientationMetadata(
  axes: Array<Record<string, unknown>>,
): boolean {
  for (const axis of axes) {
    if (
      typeof axis === "object" &&
      axis !== null &&
      axis.type === "space" &&
      "orientation" in axis &&
      // Orientation value has to be non-empty for it to count as orientation
      // metadata, mirroring the Python `if axis['orientation']:` guard (null,
      // {}, and "" are all treated as "no orientation").
      isNonEmptyOrientation(axis.orientation)
    ) {
      return true;
    }
  }
  return false;
}

function isNonEmptyOrientation(orientation: unknown): boolean {
  if (orientation === null || orientation === undefined) {
    return false;
  }
  if (typeof orientation === "object") {
    return Object.keys(orientation as Record<string, unknown>).length > 0;
  }
  if (typeof orientation === "string") {
    return orientation.length > 0;
  }
  return Boolean(orientation);
}

/**
 * Validate RFC 4 anatomical orientation metadata.
 *
 * The errors carry stable marker substrings -- `same type` (mixed `type`),
 * `all spatial axes` (all-or-none completeness), `non-space axes` (orientation on
 * a non-spatial axis) and `same anatomical axis` (two axes on one antonym pair) --
 * that `validateAxisOrientation` reads to map each failure onto its `SpecRule`.
 * These markers are a load-bearing contract pinned by a message-stability test
 * (see `structural_validation_orientation_test.ts`) so the mapping cannot silently
 * break if the wording is later edited.
 *
 * @param axes - List of axis metadata dictionaries to validate
 * @throws Error if the orientation metadata is invalid or inconsistent
 */
export function validateRfc4Orientation(
  axes: Array<Record<string, unknown>>,
): void {
  let hasOrientation = false;
  let orientationType: string | null = null;
  const spatialAxesWithOrientation: string[] = [];
  const spatialAxesWithoutOrientation: string[] = [];
  const spatialOrientationValues: Array<[string, string]> = [];

  for (const axis of axes) {
    if (typeof axis === "object" && axis !== null && axis.type === "space") {
      const axisName = String(axis.name ?? "unknown");

      if ("orientation" in axis && axis.orientation !== null) {
        hasOrientation = true;
        spatialAxesWithOrientation.push(axisName);

        const orientation = axis.orientation as Record<string, unknown>;

        // Check that all orientations have the same type
        const currentType = orientation.type as string | undefined;
        if (orientationType === null) {
          orientationType = currentType ?? null;
        } else if (currentType !== orientationType) {
          throw new Error(
            `All spatial axis orientations must have the same type. ` +
              `Found types: ${orientationType} and ${currentType}`,
          );
        }

        // Validate the orientation value using the schema
        const orientationValue = orientation.value as string | undefined;
        if (orientationValue !== undefined) {
          const result = AnatomicalOrientationValuesSchema.safeParse(
            orientationValue,
          );
          if (!result.success) {
            throw new Error(
              `Invalid orientation value '${orientationValue}' for axis '${axisName}'. ` +
                `Valid values are: ${
                  [...VALID_ORIENTATION_VALUES].sort().join(", ")
                }`,
            );
          }
          spatialOrientationValues.push([axisName, orientationValue]);
        }
      } else {
        spatialAxesWithoutOrientation.push(axisName);
      }
    }
  }

  // RFC 4 requirement: if orientation is defined for one spatial axis, it must
  // be defined for all spatial axes. Checked before the rules below to keep the
  // canonical SpecRule precedence (completeness is declared before the non-space
  // and unique-axis rules).
  if (hasOrientation && spatialAxesWithoutOrientation.length > 0) {
    throw new Error(
      `RFC 4 requires that if orientation is defined for one spatial axis, ` +
        `it must be defined for all spatial axes. ` +
        `Axes with orientation: ` +
        `${formatNameList(spatialAxesWithOrientation)}, ` +
        `axes without orientation: ` +
        `${formatNameList(spatialAxesWithoutOrientation)}`,
    );
  }

  // RFC 4 requirement: orientation is only allowed on spatial axes. (Checked
  // over all axes, independently of hasOrientation, so a stray orientation on a
  // time/channel axis is caught even when no spatial axis carries one.)
  const nonSpaceWithOrientation: string[] = [];
  for (const axis of axes) {
    if (
      typeof axis === "object" &&
      axis !== null &&
      axis.type !== "space" &&
      "orientation" in axis
    ) {
      nonSpaceWithOrientation.push(String(axis.name));
    }
  }
  if (nonSpaceWithOrientation.length > 0) {
    throw new Error(
      `RFC 4 orientation is only allowed on spatial axes; found orientation ` +
        `on non-space axes: ${formatNameList(nonSpaceWithOrientation)}`,
    );
  }

  // RFC 4 requirement: at most one direction per anatomical axis -- the two
  // antonyms of a pair are mutually exclusive across the spatial axes. An ordered
  // list of [name, value] entries (not a name-keyed map) keeps every entry, so a
  // repeated axis name cannot mask a collision.
  const anatomicalAxisSeen: Record<string, string> = {};
  for (const [name, value] of spatialOrientationValues) {
    const axisKey = ANATOMICAL_AXIS_OF[value];
    if (axisKey in anatomicalAxisSeen) {
      throw new Error(
        `Axes '${anatomicalAxisSeen[axisKey]}' and '${name}' describe the ` +
          `same anatomical axis; RFC 4 allows only one direction per axis.`,
      );
    }
    anatomicalAxisSeen[axisKey] = name;
  }
}
