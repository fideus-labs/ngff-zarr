// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { AxesType, SupportedDims, Units } from "./units.ts";
import { NgffVersion } from "./supported_versions.ts";
import type { NgffImage } from "./ngff_image.ts";

export interface Axis {
  name: SupportedDims;
  type: AxesType;
  unit: Units | undefined;
}

export interface Identity {
  type: "identity";
}

export interface Scale {
  scale: number[];
  type: "scale";
}

export interface Translation {
  translation: number[];
  type: "translation";
}

export type Transform = Scale | Translation;

export interface Dataset {
  path: string;
  coordinateTransformations: Transform[];
}

export interface OmeroWindow {
  min?: number;
  max?: number;
  start?: number;
  end?: number;
}

export interface OmeroChannel {
  color: string;
  window: OmeroWindow;
  label?: string;
  active?: boolean;
}

export interface Omero {
  channels: OmeroChannel[];
  version?: string;
}

export interface MethodMetadata {
  description: string;
  method: string;
  version: string;
}

/**
 * OME-Zarr Metadata interface for data transfer
 */
export interface MetadataInterface {
  axes: Axis[];
  datasets: Dataset[];
  coordinateTransformations: Transform[] | undefined;
  omero: Omero | undefined;
  name: string;
  version: string;
  type?: string;
  metadata?: MethodMetadata;
}

/**
 * Result from parsing zarr attributes
 */
export interface FromZarrAttrsResult {
  metadata: MetadataInterface;
  images: NgffImage[];
}

// Keep backward compatible alias
export type Metadata = MetadataInterface;

/**
 * Result from parsing zarr attributes
 */
export interface FromZarrAttrsResult {
  metadata: MetadataInterface;
  images: NgffImage[];
}

/**
 * Supported dimension names for backward compatibility
 */
export const SUPPORTED_DIMS: readonly SupportedDims[] = [
  "t",
  "c",
  "z",
  "y",
  "x",
] as const;

/**
 * Create a Metadata object with the specified version
 */
export function createMetadataWithVersion(
  metadata: MetadataInterface,
  targetVersion: string | NgffVersion,
): MetadataInterface {
  const version = typeof targetVersion === "string"
    ? (targetVersion as NgffVersion)
    : targetVersion;

  if (version === NgffVersion.V04) {
    return {
      ...metadata,
      version: "0.4",
    };
  } else if (version === NgffVersion.V05) {
    return {
      ...metadata,
      version: "0.5",
    };
  } else {
    throw new Error(
      `Unsupported version conversion: ${metadata.version} -> ${version}`,
    );
  }
}

/**
 * Get dimension names from metadata axes
 */
export function getDimensionNames(metadata: MetadataInterface): string[] {
  return metadata.axes.map((ax) => ax.name);
}

export function validateColor(color: string): void {
  if (!/^[0-9A-Fa-f]{6}$/.test(color)) {
    throw new Error(`Invalid color '${color}'. Must be 6 hex digits.`);
  }
}

export function createScale(scale: number[]): Scale {
  return { scale: [...scale], type: "scale" };
}

export function createTranslation(translation: number[]): Translation {
  return { translation: [...translation], type: "translation" };
}

export function createIdentity(): Identity {
  return { type: "identity" };
}
