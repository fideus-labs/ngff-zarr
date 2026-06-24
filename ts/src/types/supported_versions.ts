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
   * Draft development version of the OME-Zarr v0.6 (RFC-5) spec. Stores are
   * tagged on disk with this string while the spec is still in development;
   * see {@link V06_ONDISK_VERSION}. Temporary — tracks the upstream dev release.
   */
  V06dev4 = "0.6.dev4",
  LATEST = "0.6.dev4",
}

/**
 * The on-disk `ome.version` string written for OME-Zarr v0.6. The v0.6 / RFC-5
 * spec is still a draft, so stores are tagged with the development version
 * `0.6.dev4` even though the library's public `version` option is `"0.6"`.
 * This is a temporary shim that tracks the upstream spec's dev releases;
 * mirrors the Python port writing `"0.6.dev4"` for a requested version `"0.6"`.
 */
export const V06_ONDISK_VERSION: NgffVersion = NgffVersion.V06dev4;

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
