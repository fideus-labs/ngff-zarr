// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Utilities for parsing OME-Zarr metadata from zarr attributes.
 */

import {
  isSupportedVersion,
  isV06Version,
  NgffVersion,
} from "../types/supported_versions.ts";
import type { Methods } from "../types/methods.ts";
import { methodsValues } from "../types/methods.ts";
import type {
  MethodMetadata,
  Omero,
  OmeroChannel,
  OmeroWindow,
} from "../types/zarr_metadata.ts";

/**
 * Result from extracting method metadata
 */
export interface MethodMetadataResult {
  method: Methods | undefined;
  methodType: string | undefined;
  methodMetadata: MethodMetadata | undefined;
}

/**
 * Extract method type and convert to Methods enum.
 */
export function extractMethodMetadata(
  metadataDict: Record<string, unknown>,
): MethodMetadataResult {
  let method: Methods | undefined = undefined;
  let methodType: string | undefined = undefined;
  let methodMetadata: MethodMetadata | undefined = undefined;

  if (metadataDict && typeof metadataDict === "object") {
    if ("type" in metadataDict && metadataDict.type !== null) {
      methodType = metadataDict.type as string;
      // Find the corresponding Methods enum
      if (methodsValues.includes(methodType as Methods)) {
        method = methodType as Methods;
      }
    }

    // Extract method metadata if present
    if ("metadata" in metadataDict && metadataDict.metadata !== null) {
      const metadata = metadataDict.metadata as Record<string, unknown>;
      if (metadata && typeof metadata === "object") {
        methodMetadata = {
          description: String(metadata.description ?? ""),
          method: String(metadata.method ?? ""),
          version: String(metadata.version ?? ""),
        };
      }
    }
  }

  return { method, methodType, methodMetadata };
}

/**
 * Parse OMERO metadata dictionary into Omero interface.
 */
/**
 * Build an `OmeroWindow` from whatever bounds a channel carries.
 *
 * Stores in the wild write `min`/`max`, `start`/`end`, or both. When only one
 * pair is present it stands for the other, which is how this library has
 * always read them. A window with neither pair keeps the bounds it has and
 * leaves the rest unset rather than making the channel unreadable.
 */
function parseOmeroWindow(windowData: unknown): OmeroWindow | undefined {
  if (!windowData || typeof windowData !== "object") {
    return undefined;
  }
  const data = windowData as Record<string, unknown>;
  const number = (key: string): number | undefined => {
    const value = data[key];
    return value === undefined || value === null ? undefined : Number(value);
  };

  let min = number("min");
  let max = number("max");
  let start = number("start");
  let end = number("end");
  if (start === undefined && end === undefined) {
    start = min;
    end = max;
  } else if (min === undefined && max === undefined) {
    min = start;
    max = end;
  }
  return {
    ...(min !== undefined ? { min } : {}),
    ...(max !== undefined ? { max } : {}),
    ...(start !== undefined ? { start } : {}),
    ...(end !== undefined ? { end } : {}),
  };
}

export function parseOmero(
  omeroData: Record<string, unknown> | undefined | null,
): Omero | undefined {
  if (!omeroData || typeof omeroData !== "object") {
    return undefined;
  }

  if (!("channels" in omeroData) || !Array.isArray(omeroData.channels)) {
    return undefined;
  }

  const channels: OmeroChannel[] = [];

  for (const channel of omeroData.channels as Array<Record<string, unknown>>) {
    if (!channel || typeof channel !== "object") {
      channels.push({});
      continue;
    }

    const window = parseOmeroWindow(channel.window);
    const omeroChannel: OmeroChannel = {
      ...(channel.color !== undefined && channel.color !== null
        ? { color: String(channel.color) }
        : {}),
      ...(window !== undefined ? { window } : {}),
      ...(channel.label !== undefined && channel.label !== null
        ? { label: String(channel.label) }
        : {}),
      ...(typeof channel.active === "boolean"
        ? { active: channel.active }
        : {}),
    };

    channels.push(omeroChannel);
  }

  return {
    channels,
    ...(typeof omeroData.version === "string"
      ? { version: omeroData.version }
      : {}),
  };
}

/**
 * Detect NGFF version from root attributes.
 */
export function detectVersion(
  rootAttrs: Record<string, unknown>,
): NgffVersion {
  let versionStr: string | undefined = undefined;

  if ("ome" in rootAttrs && rootAttrs.ome) {
    const ome = rootAttrs.ome as Record<string, unknown>;
    versionStr = ome.version as string | undefined;
  }

  // Fall back to multiscales if version not found in ome
  if (versionStr === undefined) {
    const multiscales = rootAttrs.multiscales as unknown[];
    if (multiscales && Array.isArray(multiscales) && multiscales.length > 0) {
      const firstMultiscale = multiscales[0] as Record<string, unknown>;
      versionStr = (firstMultiscale.version as string) ?? "0.4";
    }
  }

  if (versionStr === undefined) {
    throw new Error("Could not detect NGFF version from root attributes.");
  }

  // Any 0.6-family version, including pre-release tags such as `0.6.dev4`
  // and `0.6rc0`, is read as v0.6. Mirrors the Python port's
  // `version.startswith("0.6")` read check so dev releases remain readable.
  if (isV06Version(versionStr)) {
    return NgffVersion.V06;
  }

  if (!isSupportedVersion(versionStr)) {
    throw new Error(`Unsupported NGFF version: ${versionStr}`);
  }

  return versionStr as NgffVersion;
}
