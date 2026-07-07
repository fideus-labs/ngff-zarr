// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { AxesType, SupportedDims, Units } from "./units.ts";
import { NgffVersion } from "./supported_versions.ts";
import type { NgffImage } from "./ngff_image.ts";
import type { AnatomicalOrientation } from "./rfc4.ts";

/**
 * Name of the implicit coordinate system generated for a multiscale image,
 * matching the Python port. Datasets map their array index space into this
 * system via a scale + translation sequence.
 */
export const INTRINSIC_COORDINATE_SYSTEM_NAME = "intrinsic";

/**
 * Orientation metadata for spatial axes (RFC 4).
 * This interface represents the serialized form in the zarr metadata.
 */
export interface AxisOrientation {
  type: string;
  value: string;
}

export interface Axis {
  name: SupportedDims;
  type: AxesType;
  unit: Units | undefined;
  orientation?: AxisOrientation | AnatomicalOrientation | undefined;
  discrete?: boolean;
}

/**
 * RFC 5 / OME-Zarr v0.6 coordinate-system reference used by the `input` and
 * `output` fields of a transformation. Mirrors the Python
 * `CoordinateSystemIdentifier` dataclass and the spec `inputOutput` object: a
 * transformation references either a coordinate system by `name` or a dataset
 * array by `path`.
 */
export interface CoordinateSystemIdentifier {
  path?: string;
  name?: string;
}

/**
 * RFC 5 / OME-Zarr v0.6 coordinate system: a named set of axes. The implicit
 * "intrinsic" coordinate system is generated for the multiscale image.
 */
export interface CoordinateSystem {
  name: string;
  axes: Axis[];
}

export interface Identity {
  type: "identity";
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

export interface Scale {
  scale: number[];
  type: "scale";
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

export interface Translation {
  translation: number[];
  type: "translation";
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

/** RFC 5 rotation transformation (v0.6). */
export interface Rotation {
  rotation: number[][];
  type: "rotation";
  path?: string;
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

/** RFC 5 affine transformation (v0.6). */
export interface Affine {
  affine: number[][];
  type: "affine";
  path?: string;
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

/** RFC 5 coordinate-field transformation referencing a coordinate field (v0.6). */
export interface Coordinates {
  type: "coordinates";
  path: string;
  interpolation?: string;
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

/** RFC 5 displacement-field transformation referencing a displacement field (v0.6). */
export interface Displacements {
  type: "displacements";
  path: string;
  interpolation?: string;
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

/** RFC 5 sequence transformation, chaining sub-transformations (v0.6). */
export interface TransformSequence {
  transformations: V06Transform[];
  type: "sequence";
  input?: CoordinateSystemIdentifier;
  output?: CoordinateSystemIdentifier;
  name?: string;
}

/**
 * Simple per-dataset transform used by the v0.4/v0.5 in-memory model and as the
 * extracted scale/translation of a v0.6 dataset.
 */
export type Transform = Scale | Translation;

/**
 * Full RFC 5 / v0.6 transformation union. Used for top-level multiscale
 * `coordinateTransformations` and the per-dataset `sequence` on the wire.
 */
export type V06Transform =
  | Identity
  | Scale
  | Translation
  | Rotation
  | Affine
  | Coordinates
  | Displacements
  | TransformSequence;

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
  /**
   * Top-level multiscale transformations. For v0.4/v0.5 these are simple
   * {@link Transform}s; for RFC 5 / v0.6 they may be any {@link V06Transform}
   * (rotation, affine, sequence, ...) referencing coordinate systems.
   */
  coordinateTransformations: V06Transform[] | undefined;
  /**
   * RFC 5 / OME-Zarr v0.6 coordinate systems. Populated when reading a v0.6
   * store and by `toMultiscales` (the implicit "intrinsic" system). Unused when
   * serializing v0.4/v0.5, where axes live on the multiscale entry directly.
   */
  coordinateSystems?: CoordinateSystem[];
  omero: Omero | undefined;
  name: string;
  version: string;
  type?: string;
  metadata?: MethodMetadata;
  /**
   * Unrecognized keys that leaked into the `multiscales[]` entry (a malformed
   * or double-namespaced v0.5 document). Captured verbatim so the v0.5
   * namespacing rules (`zarr-format`, `ome-namespace`) can flag them; mirrors
   * the Rust port's `#[serde(flatten)]` `extra` passthrough. It is a read-side
   * validation aid only and is never serialized back to the store.
   */
  extra?: Record<string, unknown>;
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
  } else if (version === NgffVersion.V06) {
    // The in-memory model is version-agnostic (axes + per-dataset scale and
    // translation); v0.6 additionally exposes coordinate systems. Synthesize
    // the implicit "intrinsic" system from the axes when one is not present, so
    // a value converted to v0.6 carries the field the writer expects.
    return {
      ...metadata,
      version: "0.6",
      coordinateSystems: metadata.coordinateSystems ??
        [{ name: INTRINSIC_COORDINATE_SYSTEM_NAME, axes: metadata.axes }],
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

export function createRotation(rotation: number[][]): Rotation {
  return { rotation: rotation.map((row) => [...row]), type: "rotation" };
}

export function createAffine(affine: number[][]): Affine {
  return { affine: affine.map((row) => [...row]), type: "affine" };
}

export function createTransformSequence(
  transformations: V06Transform[],
): TransformSequence {
  return { transformations: [...transformations], type: "sequence" };
}

export function createCoordinateSystem(
  name: string,
  axes: Axis[],
): CoordinateSystem {
  return { name, axes: [...axes] };
}
