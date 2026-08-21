// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * RFC 4 validation for anatomical orientation in OME-NGFF.
 *
 * This module provides validation for RFC 4 anatomical orientation metadata.
 */

import { AnatomicalOrientationValuesSchema } from "../schemas/rfc4.ts";
import { formatNameList } from "./py_format.ts";

/**
 * Each RFC 4 value maps to a stable key for the anatomical axis it lies on, so
 * the two antonyms of a pair (e.g. "left-to-right" / "right-to-left") collapse to
 * one key. Used to enforce "one direction per anatomical axis" (mutual exclusion).
 */
export const ANATOMICAL_AXIS_OF: Record<string, string> = {
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
 * The controlled vocabulary, derived from the pair table rather than typed out
 * again: the message that lists the valid values then cannot disagree with what
 * the mutual-exclusion rule can look up. `rfc4_vocabulary_test.ts` pins it
 * against the Zod schema and the `AnatomicalOrientationValues` enum.
 */
const VALID_ORIENTATION_VALUES = new Set(Object.keys(ANATOMICAL_AXIS_OF));

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
 * The errors carry stable marker substrings -- `must be anatomical` (a `type`
 * other than the single value the vocabulary defines), `non-space axes`
 * (orientation on a non-spatial axis) and `same anatomical axis` (two axes on one
 * antonym pair) -- that `validateAxisOrientation` reads to map each failure onto
 * its `SpecRule`. These markers are a load-bearing contract pinned by a
 * message-stability test (see `structural_validation_orientation_test.ts`) so the
 * mapping cannot silently break if the wording is later edited.
 *
 * RFC 4 makes `orientation` optional per spatial axis: an image may orient only
 * some of its spatial axes, and an absent orientation is equivalent to an explicit
 * `null` (the axis orientation is undefined). No all-or-none completeness rule is
 * enforced.
 *
 * @param axes - List of axis metadata dictionaries to validate
 * @throws Error if the orientation metadata is invalid or inconsistent
 */
export function validateRfc4Orientation(
  axes: Array<Record<string, unknown>>,
): void {
  const spatialOrientationValues: Array<[string, string]> = [];

  for (const axis of axes) {
    if (typeof axis === "object" && axis !== null && axis.type === "space") {
      const axisName = String(axis.name ?? "unknown");

      // RFC 4: an absent orientation and an explicit null (or empty object) are
      // equivalent -- the axis orientation is simply undefined -- so only a real,
      // non-empty orientation object is validated here. Orientation is optional
      // per spatial axis, so a partially oriented image is valid.
      if (isNonEmptyOrientation(axis.orientation)) {
        const orientation = axis.orientation as Record<string, unknown>;

        // RFC 4: the orientation type is restricted to the single value the
        // controlled vocabulary defines, "anatomical".
        const orientationType = orientation.type as string | undefined;
        if (orientationType !== "anatomical") {
          throw new Error(
            `RFC 4 orientation type must be anatomical; axis ` +
              `'${axisName}' has type '${orientationType}'.`,
          );
        }

        // RFC 4 requires a value, drawn from the controlled vocabulary.
        const orientationValue = orientation.value as string | undefined;
        if (
          orientationValue === undefined ||
          !AnatomicalOrientationValuesSchema.safeParse(orientationValue).success
        ) {
          throw new Error(
            `Invalid orientation value '${orientationValue}' for axis '${axisName}'. ` +
              `Valid values are: ${
                [...VALID_ORIENTATION_VALUES].sort().join(", ")
              }`,
          );
        }
        spatialOrientationValues.push([axisName, orientationValue]);
      }
    }
  }

  // RFC 4 requirement: orientation is only allowed on spatial axes. Only a real
  // (non-empty) orientation object counts as "used"; a null or empty orientation
  // is undefined, so it is not a violation on a non-spatial axis. Checked over all
  // axes so a stray orientation is caught even when no spatial axis carries one.
  const nonSpaceWithOrientation: string[] = [];
  for (const axis of axes) {
    if (
      typeof axis === "object" &&
      axis !== null &&
      axis.type !== "space" &&
      isNonEmptyOrientation(axis.orientation)
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
