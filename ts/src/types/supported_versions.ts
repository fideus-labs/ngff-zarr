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
  LATEST = "0.5",
}

/**
 * Supported NGFF specification versions
 */
export const SUPPORTED_VERSIONS: readonly NgffVersion[] = [
  NgffVersion.V01,
  NgffVersion.V02,
  NgffVersion.V03,
  NgffVersion.V04,
  NgffVersion.V05,
] as const;

/**
 * Check if a version string is supported
 */
export function isSupportedVersion(version: string): version is NgffVersion {
  return Object.values(NgffVersion).includes(version as NgffVersion);
}
