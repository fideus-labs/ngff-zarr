// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Scene metadata: spatial relationships between images, mirroring the
 * Python port's `ngff_zarr.rfc8.scene`.
 *
 * RFC-5 introduced scene metadata for collections of images that share a
 * coordinate space, stored as `ome.scene` on a Zarr group of the 0.6 family
 * with name-based identifiers; RFC-8 embeds the same information as a
 * `scene` attribute of a collection node (OME-Zarr 0.9.dev3) with id-based
 * references. Both shapes parse into one in-memory {@link Scene}.
 *
 * The scene's transformations relate coordinate systems that mostly live in
 * other documents (the collection's member images); identifiers are
 * therefore preserved verbatim, unresolved names and ids included, and
 * dereferencing is left to the caller.
 */

import type {
  Axis,
  CoordinateSystem,
  CoordinateSystemIdentifier,
  V06Transform,
} from "../types/zarr_metadata.ts";
import type { OmeNode } from "../types/rfc8.ts";
import { parseV06Transforms, serializeV06Transform } from "./v06_metadata.ts";

/** The RFC-8 attribute key (and the RFC-5 `ome` key) this module owns. */
export const SCENE_KEY = "scene";

/**
 * Scene metadata: transformations between referenced coordinate systems.
 * `coordinateTransformations` is required by both the RFC-5 and RFC-8
 * shapes; `coordinateSystems` declares the systems the collection itself
 * owns (the first one is the recommended common reference).
 */
export interface Scene {
  coordinateTransformations: V06Transform[];
  coordinateSystems?: CoordinateSystem[] | undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function identifierFrom(
  raw: unknown,
): CoordinateSystemIdentifier | undefined {
  if (!isRecord(raw)) {
    return undefined;
  }
  const path = typeof raw.path === "string" ||
      (isRecord(raw.path) && typeof raw.path.path === "string")
    ? raw.path as string | { type: string; path: string }
    : undefined;
  return {
    ...(path !== undefined && { path }),
    ...(typeof raw.name === "string" && { name: raw.name }),
    ...(typeof raw.id === "string" && { id: raw.id }),
  };
}

/** Parse a raw scene object (either storage shape) into a {@link Scene}. */
export function sceneFromOmeValue(
  raw: Record<string, unknown>,
  version?: string,
): Scene {
  let systems: CoordinateSystem[] | undefined;
  if (Array.isArray(raw.coordinateSystems)) {
    systems = raw.coordinateSystems.filter(isRecord).map((entry) => ({
      ...(typeof entry.name === "string" && { name: entry.name }),
      axes: (Array.isArray(entry.axes) ? entry.axes : []) as Axis[],
      ...(typeof entry.id === "string" && { id: entry.id }),
    }));
  }
  const rawTransforms = Array.isArray(raw.coordinateTransformations)
    ? (raw.coordinateTransformations as Record<string, unknown>[])
    : [];
  const transforms = parseV06Transforms(
    rawTransforms,
    (systems ?? []).map((system) => system.name),
    systems ?? [],
    version,
  );
  // Restore the references verbatim: the wrapped parser resolves names
  // against the declared systems and drops the rest.
  for (const [index, parsed] of transforms.entries()) {
    const entry = rawTransforms[index];
    if (isRecord(entry)) {
      const input = identifierFrom(entry.input);
      const output = identifierFrom(entry.output);
      if (input !== undefined) {
        parsed.input = input;
      } else {
        delete parsed.input;
      }
      if (output !== undefined) {
        parsed.output = output;
      } else {
        delete parsed.output;
      }
    }
  }
  return {
    coordinateTransformations: transforms,
    ...(systems !== undefined && { coordinateSystems: systems }),
  };
}

/**
 * Serialize a {@link Scene} to its storage object for the shape `version`
 * selects. The RFC-8 shape (`"0.9.dev3"`, the default) keeps ids on systems
 * and references; the 0.6-family shape identifies coordinate systems by
 * name, so a system or reference without one cannot be expressed there and
 * throws. Mirrors the Python `_serialize_scene`.
 */
export function sceneToOmeValue(
  value: Scene,
  version: string = "0.9.dev3",
): Record<string, unknown> {
  const idBased = version === "0.9.dev3";
  const serialized: Record<string, unknown> = {};
  if (value.coordinateSystems !== undefined) {
    serialized.coordinateSystems = value.coordinateSystems.map((system) => {
      if (system.name === undefined && !idBased) {
        throw new Error(
          "A 0.6-family scene names its coordinate systems, but a declared " +
            "system carries no name. Give it a name, or write the scene at " +
            'version "0.9.dev3".',
        );
      }
      return {
        ...(idBased && system.id !== undefined && { id: system.id }),
        ...(system.name !== undefined && { name: system.name }),
        axes: system.axes,
      };
    });
  }
  serialized.coordinateTransformations = value.coordinateTransformations.map(
    (transform) => {
      const entry = serializeV06Transform(transform);
      if (!idBased) {
        for (const side of ["input", "output"]) {
          const reference = entry[side];
          if (!isRecord(reference)) {
            continue;
          }
          if (reference.name === undefined) {
            throw new Error(
              `A 0.6-family scene names its ${side} coordinate systems, ` +
                "but a transformation reference carries no name. Give it a " +
                'name, or write the scene at version "0.9.dev3".',
            );
          }
          if (isRecord(reference.path)) {
            throw new Error(
              `A 0.6-family scene ${side} reference path is a string, but ` +
                "this reference carries a typed path object. Use a string " +
                'path, or write the scene at version "0.9.dev3".',
            );
          }
          delete reference.id;
        }
      }
      return entry;
    },
  );
  return serialized;
}

/** The parsed `scene` attribute of `node`, or `undefined`. */
export function scene(
  node: OmeNode,
  version: string | undefined = "0.9.dev3",
): Scene | undefined {
  const raw = node.attributes?.[SCENE_KEY];
  if (raw === undefined || raw === null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new Error(
      `A 'scene' attribute must be an object; got ${JSON.stringify(raw)}.`,
    );
  }
  return sceneFromOmeValue(raw, version);
}

/** Write `value` into `node.attributes`, replacing the key. */
export function setScene(node: OmeNode, value: Scene | undefined): void {
  const attributes = { ...(node.attributes ?? {}) };
  if (value === undefined) {
    delete attributes[SCENE_KEY];
  } else {
    attributes[SCENE_KEY] = sceneToOmeValue(value);
  }
  node.attributes = attributes;
}

/**
 * The scene metadata in a store's root attributes, or `undefined`. Handles
 * both shapes: the RFC-5 `ome.scene` of the 0.6 family and the `scene`
 * attribute of an RFC-8 node document (OME-Zarr 0.9.dev3).
 */
export function readSceneFromAttrs(
  rootAttrs: Record<string, unknown>,
): Scene | undefined {
  const ome = rootAttrs.ome;
  if (!isRecord(ome)) {
    return undefined;
  }
  const declared = typeof ome.version === "string" ? ome.version : undefined;
  let raw = ome[SCENE_KEY];
  if ((raw === undefined || raw === null) && isRecord(ome.attributes)) {
    raw = ome.attributes[SCENE_KEY];
  }
  if (raw === undefined || raw === null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new Error(
      `A 'scene' entry must be an object; got ${JSON.stringify(raw)}.`,
    );
  }
  return sceneFromOmeValue(raw, declared);
}
