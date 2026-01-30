// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Utilities for parsing OME-Zarr metadata from zarr attributes.
 */

import {
  isSupportedVersion,
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
    if (
      !channel ||
      typeof channel !== "object" ||
      !("window" in channel) ||
      !channel.window
    ) {
      continue;
    }

    const windowData = channel.window as Record<string, unknown>;
    if (typeof windowData !== "object") {
      continue;
    }

    // Handle backward compatibility for OMERO window metadata
    // Prefer start/end format, fall back to min/max, use one as the other if needed
    let start: number;
    let end: number;
    let minVal: number;
    let maxVal: number;

    if ("start" in windowData && "end" in windowData) {
      // New format with start/end
      start = Number(windowData.start);
      end = Number(windowData.end);
      // Use start/end as min/max if not present
      minVal = ("min" in windowData) ? Number(windowData.min) : start;
      maxVal = ("max" in windowData) ? Number(windowData.max) : end;
    } else if ("min" in windowData && "max" in windowData) {
      // Old format with min/max only
      minVal = Number(windowData.min);
      maxVal = Number(windowData.max);
      // Use min/max as start/end for backward compatibility
      start = minVal;
      end = maxVal;
    } else {
      // Invalid window data, skip this channel
      continue;
    }

    const window: OmeroWindow = {
      min: minVal,
      max: maxVal,
      start: start,
      end: end,
    };

    const omeroChannel: OmeroChannel = {
      color: String(channel.color),
      window,
      ...(channel.label !== undefined && channel.label !== null
        ? { label: String(channel.label) }
        : {}),
      ...(typeof channel.active === "boolean"
        ? { active: channel.active }
        : {}),
    };

    channels.push(omeroChannel);
  }

  if (channels.length === 0) {
    return undefined;
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

  if (!isSupportedVersion(versionStr)) {
    throw new Error(`Unsupported NGFF version: ${versionStr}`);
  }

  return versionStr as NgffVersion;
}
