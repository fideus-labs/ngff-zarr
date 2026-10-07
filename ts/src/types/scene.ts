// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { NgffMultiscales } from "./multiscales.ts";
import type { CoordinateSystem, V06Transform } from "./zarr_metadata.ts";

export interface NgffSceneOptions {
  images: Record<string, NgffMultiscales>;
  coordinateTransformations: V06Transform[];
  coordinateSystems?: CoordinateSystem[];
}

/**
 * Images that share a spatial relationship, and the transformations between
 * them (OME-Zarr 0.6). Mirrors the Python `NgffScene`.
 *
 * `images` maps each image's path below the scene group to its multiscales.
 * `coordinateTransformations` relate the images' coordinate systems to each
 * other and to `coordinateSystems`, the systems the scene declares itself.
 * Each end of a transformation is a `CoordinateSystemIdentifier`: with a
 * `path` it names a coordinate system of that image, without one a system of
 * the scene. The first scene coordinate system is the reference a viewer
 * displays by default.
 */
export class NgffScene {
  public images: Record<string, NgffMultiscales>;
  public coordinateTransformations: V06Transform[];
  public coordinateSystems?: CoordinateSystem[];

  constructor(options: NgffSceneOptions) {
    this.images = { ...options.images };
    this.coordinateTransformations = [...options.coordinateTransformations];
    if (options.coordinateSystems !== undefined) {
      this.coordinateSystems = [...options.coordinateSystems];
    }
  }
}
