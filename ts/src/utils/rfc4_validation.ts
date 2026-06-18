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
]);

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
      "orientation" in axis
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Validate RFC 4 anatomical orientation metadata.
 *
 * The two inconsistency errors carry stable marker substrings -- `same type`
 * for a mixed-`type` failure and `all spatial axes` for the all-or-none
 * completeness failure -- that `validateAxisOrientation` reads to map each
 * failure onto its `SpecRule`. These markers are a load-bearing contract
 * pinned by a message-stability test (see
 * `structural_validation_orientation_test.ts`) so the mapping cannot silently
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
        }
      } else {
        spatialAxesWithoutOrientation.push(axisName);
      }
    }
  }

  // RFC 4 requirement: if orientation is defined for one spatial axis,
  // it must be defined for all spatial axes
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
}
