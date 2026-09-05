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
   * Pre-release tags of the OME-Zarr v0.6 (RFC-5) spec. Both stay readable:
   * stores written while 0.6 was a draft carry the `dev4` tag on disk.
   */
  V06dev4 = "0.6.dev4",
  V06rc0 = "0.6rc0",
  /**
   * OME-Zarr 0.9 in development: v0.6 plus RFC-3 (any axis count, names,
   * types and ordering). Unlike {@link V06dev4} this is the on-disk string.
   * {@link LATEST} stays `0.6rc0`, so 0.9.dev1 is opt-in.
   */
  V09dev1 = "0.9.dev1",
  /**
   * 0.9.dev1 plus RFC-8: the root `ome` value becomes a typed node
   * (collections, references, typed paths) in place of the multiscales
   * wrapper. Also opt-in; {@link LATEST} stays `0.6rc0`.
   */
  V09dev3 = "0.9.dev3",
  LATEST = "0.6rc0",
}

/**
 * The versions an image (multiscales) store can carry, as the readers,
 * writers and upgraders accept them. Widens to `"0.9.dev3"` when the RFC-8
 * multiscale node model lands (issue #714); until then images are written at
 * 0.9.dev1 and referenced from 0.9.dev3 collections.
 */
export type ImageVersion = "0.4" | "0.5" | "0.6" | "0.9.dev1";

/**
 * The OME-Zarr 0.9 development series, oldest first. Each tag extends the
 * one before: dev1 is 0.6 plus RFC-3 and RFC-4, dev3 is dev1 plus RFC-8.
 */
export const V09_DEV_SERIES: readonly string[] = [
  NgffVersion.V09dev1,
  NgffVersion.V09dev3,
] as const;

/**
 * The on-disk `ome.version` string written for OME-Zarr v0.6. The v0.6 spec
 * is a release candidate, so a store is tagged with the pre-release the
 * bundled `spec/0.6` schemas carry rather than with the bare `"0.6"` the
 * public `version` option accepts. Mirrors the Python port's
 * `V06_ONDISK_VERSION`; this constant is the one place the tag lives.
 */
export const V06_ONDISK_VERSION: NgffVersion = NgffVersion.V06rc0;

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
  NgffVersion.V06dev4,
  NgffVersion.V06rc0,
  NgffVersion.V09dev1,
  NgffVersion.V09dev3,
] as const;

/**
 * Check if a version string is supported
 */
export function isSupportedVersion(version: string): version is NgffVersion {
  return Object.values(NgffVersion).includes(version as NgffVersion);
}

/**
 * Whether an `ome.version` string identifies an OME-Zarr v0.6 store, including
 * draft development versions such as `0.6.dev4`. Mirrors the Python port's
 * `version.startswith("0.6")` check so a store written by either implementation
 * reads back as v0.6.
 */
export function isV06Version(version: string): boolean {
  return version.startsWith("0.6");
}

/**
 * Whether a version adopts the RFC-3 free-form axis model.
 *
 * Only the OME-Zarr 0.9 development series does: the bundled 0.4, 0.5 and
 * 0.6 axes schemas all cap the axis count at 5, and require 2-3 `space` axes
 * (0.6 excepted for an `array` coordinate system, see
 * `takesArraySchemaBranch`). `undefined` applies the restrictions.
 */
export function isRfc3AxisModelAllowed(version?: string): boolean {
  return version !== undefined && V09_DEV_SERIES.includes(version);
}

/**
 * Whether a version makes the RFC-4 orientation rules normative.
 *
 * Only the OME-Zarr 0.9 development series does (ome/ngff-spec#190 folds
 * RFC-4 into 0.9.dev1, and 0.9.dev3 extends 0.9.dev1): the released 0.4,
 * 0.5 and 0.6 specs give `orientation` no normative status, so when the
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
  return version === undefined || V09_DEV_SERIES.includes(version);
}

/**
 * Whether a version stores the RFC-8 node model under `ome`.
 *
 * Only `0.9.dev3` does: RFC-8 replaces the multiscales wrapper with a typed
 * node document, so this predicate answers a structural question (which
 * document shape a version stores) rather than gating a strictness rule,
 * and `undefined` is `false`: with no declared version nothing says the
 * document is node-shaped. The RFC-8 collection rules gate themselves the
 * RFC-4 way instead (see `validateCollection`).
 */
export function isRfc8NodeModel(version?: string): boolean {
  return version !== undefined && version === NgffVersion.V09dev3;
}
