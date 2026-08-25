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
  LATEST = "0.6rc0",
}

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
