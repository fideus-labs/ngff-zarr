// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * A transformation on its own (OME-Zarr 0.6). Mirrors the Python port's
 * `ngff_zarr.transformation`.
 *
 * The spec's standalone transformation examples are documents that hold
 * `coordinateTransformations` and, optionally, `coordinateSystems`, without
 * an image: the shape of a scene without images. A transformation store is
 * a group whose `ome` metadata is such a document with one transformation.
 * Parameters that live in arrays, such as a `displacements` field, are nodes
 * below the store, written first, that the transformation references by
 * `path`. No Node.js dependency: both writers and both readers use it.
 */
import * as zarr from "zarrita";
import { NgffMultiscales } from "../types/multiscales.ts";
import { NgffScene } from "../types/scene.ts";
import { V06_ONDISK_VERSION } from "../types/supported_versions.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import { consolidateMetadata } from "../utils/consolidate_metadata.ts";
import { detectVersion } from "../utils/parse_metadata.ts";
import {
  serializeV06Transform,
  validateV06Transform,
} from "../utils/v06_metadata.ts";
import {
  checkImagePath,
  fieldPaths,
  SCENE_VERSIONS,
  sceneFromOmeValue,
} from "./scene_common.ts";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Whether `value` is a transformation object rather than an image or a scene. */
export function isV06Transform(value: unknown): value is V06Transform {
  return isRecord(value) && typeof value.type === "string" &&
    !(value instanceof NgffMultiscales) && !(value instanceof NgffScene);
}

/** Whether `rootAttrs` is the root document of a transformation store. */
export function hasTransformationMetadata(
  rootAttrs: Record<string, unknown>,
): boolean {
  const ome = rootAttrs.ome;
  return isRecord(ome) && "coordinateTransformations" in ome &&
    !("multiscales" in ome) && !("scene" in ome);
}

export interface WriteTransformationOptions {
  /** OME-Zarr specification version, 0.6 (the default) or later. */
  version?: "0.4" | "0.5" | "0.6" | "0.9.dev1";
  /**
   * With the default `true`, the root group's attributes are the
   * transformation's alone; with `false`, the attributes an existing root
   * group carries are kept beside it, as are the nodes the transformation
   * references by `path`, which have to be in the store already.
   */
  overwrite?: boolean;
}

/**
 * Write `transform` as the only transformation of `store`; `toOmeZarr`
 * calls this. The transformation lands in the root group's
 * `ome.coordinateTransformations`.
 */
export async function writeTransformation(
  store: zarr.Mutable,
  transform: V06Transform,
  options: WriteTransformationOptions = {},
): Promise<void> {
  const version = options.version ?? "0.6";
  if (!SCENE_VERSIONS.includes(version)) {
    throw new Error(
      "A transformation store is defined from OME-Zarr 0.6; got version " +
        `'${version}'.`,
    );
  }
  validateV06Transform(transform, [], version);
  const fields = [...fieldPaths([transform])].sort();
  for (const path of fields) {
    checkImagePath(path);
  }
  if (fields.length > 0 && (options.overwrite ?? true)) {
    throw new Error(
      `The transformation references the nodes ${
        JSON.stringify(fields)
      }; write those first with toOmeZarr() below the store, then the ` +
        "transformation with overwrite: false so they are kept.",
    );
  }
  for (const path of fields) {
    if ((await store.get(`/${path}/zarr.json`)) === undefined) {
      throw new Error(
        `The transformation references '${path}', which the store does ` +
          "not hold; write it first with toOmeZarr() below the store, then " +
          "the transformation with overwrite: false.",
      );
    }
  }
  const root = zarr.root(store);
  let attributes: Record<string, unknown> = {};
  if (options.overwrite === false) {
    try {
      const existing = await zarr.open(root, { kind: "group" });
      attributes = { ...(existing.attrs as Record<string, unknown>) };
    } catch {
      // No root group yet: nothing to keep.
    }
  }
  attributes.ome = {
    version: version === "0.6" ? V06_ONDISK_VERSION : version,
    coordinateTransformations: [serializeV06Transform(transform)],
  };
  await zarr.create(root, { attributes });
  const nodePaths = new Set<string>();
  for (const path of fields) {
    const segments = path.split("/");
    for (let end = 1; end < segments.length; end++) {
      const ancestor = segments.slice(0, end).join("/");
      nodePaths.add(ancestor);
      try {
        await zarr.open(root.resolve(ancestor), { kind: "group" });
      } catch {
        await zarr.create(root.resolve(ancestor));
      }
    }
    nodePaths.add(path);
  }
  await consolidateMetadata(store, [...nodePaths].sort());
}

/** The transformation `rootAttrs` declares; `fromOmeZarr` calls this. */
export function readTransformation(
  rootAttrs: Record<string, unknown>,
  options: { validate?: boolean; version?: string } = {},
): V06Transform {
  const version = options.version ?? detectVersion(rootAttrs);
  const { coordinateTransformations } = sceneFromOmeValue(
    rootAttrs.ome as Record<string, unknown>,
    version,
  );
  if (coordinateTransformations.length !== 1) {
    throw new Error(
      "A transformation store holds one coordinateTransformations entry; " +
        `got ${coordinateTransformations.length}.`,
    );
  }
  const transform = coordinateTransformations[0];
  if (options.validate ?? false) {
    for (const path of fieldPaths([transform])) {
      checkImagePath(path);
    }
  }
  return transform;
}
