// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Constants for ngff-zarr package.
 */

export enum NgffVersion {
  V01 = "0.1",
  V02 = "0.2",
  V03 = "0.3",
  V04 = "0.4",
  V05 = "0.5",
  V06 = "0.6",
  /**
   * OME-Zarr 0.9 in development: v0.6 plus RFC-3 (any axis count, names,
   * types and ordering). This is the on-disk string. {@link LATEST} stays
   * `0.6`, so 0.9.dev1 is opt-in.
   */
  V09dev1 = "0.9.dev1",
  LATEST = "0.6",
}

/**
 * The on-disk `ome.version` string written for OME-Zarr v0.6. Since the 0.6
 * release of ome/ngff-spec the tag a store carries is `"0.6"` itself, the
 * same string the public `version` option accepts. Earlier releases wrote
 * the pre-release tags `0.6.dev4` and `0.6rc0` here; this constant remains
 * the one place both writers and `upgradeOmeZarr` read the tag from, and
 * mirrors the Python port's `V06_ONDISK_VERSION`.
 */
export const V06_ONDISK_VERSION: NgffVersion = NgffVersion.V06;

/**
 * Supported NGFF specification versions
 */
export const SUPPORTED_VERSIONS: readonly NgffVersion[] = [
  NgffVersion.V01,
  NgffVersion.V02,
  NgffVersion.V03,
  NgffVersion.V04,
  NgffVersion.V05,
  NgffVersion.V06,
  NgffVersion.V09dev1,
] as const;

/**
 * Check if a version string is supported
 */
export function isSupportedVersion(version: string): version is NgffVersion {
  return Object.values(NgffVersion).includes(version as NgffVersion);
}

/**
 * Whether an `ome.version` string identifies an OME-Zarr v0.6 store. Matches
 * the whole `0.6*` family: the released `0.6` tag as well as the historical
 * pre-release tags `0.6.dev4` and `0.6rc0` that earlier releases wrote on
 * disk. Those tags are no longer in {@link SUPPORTED_VERSIONS}, so
 * {@link isSupportedVersion} rejects them while this check keeps such stores
 * readable. Mirrors the Python port's `version.startswith("0.6")` check so a
 * store written by either implementation reads back as v0.6.
 */
export function isV06Version(version: string): boolean {
  return version.startsWith("0.6");
}

/**
 * Whether a version adopts the RFC-3 free-form axis model.
 *
 * Only `0.9.dev1` does: the bundled 0.4, 0.5 and 0.6 axes schemas all cap the
 * axis count at 5, and require 2-3 `space` axes (0.6 excepted for an `array`
 * coordinate system, see `takesArraySchemaBranch`). `undefined` applies the
 * restrictions.
 */
export function isRfc3AxisModelAllowed(version?: string): boolean {
  return version !== undefined && version === NgffVersion.V09dev1;
}

/**
 * Whether a version makes the RFC-4 orientation rules normative.
 *
 * Only `0.9.dev1` does (ome/ngff-spec#190 folds RFC-4 into it): the released
 * 0.4, 0.5 and 0.6 specs give `orientation` no normative status, so when the
 * caller declares one of those versions the orientation rules are inert.
 * `undefined` keeps them on: with no declared version, enforcement is a
 * strictness choice, exactly like `axis-names-unique` below 0.9.dev1 (see
 * docs/validation/rule-reference.md).
 *
 * The gate points the opposite way from {@link isRfc3AxisModelAllowed}: at
 * 0.9.dev1 RFC-3 *lifts* the axis restrictions while RFC-4 *adds* the
 * orientation requirements, so the rules exit early below 0.9.dev1 rather
 * than at it.
 */
export function isRfc4OrientationEnforced(version?: string): boolean {
  return version === undefined || version === NgffVersion.V09dev1;
}
