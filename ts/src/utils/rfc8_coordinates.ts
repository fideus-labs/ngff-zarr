// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Typed views over the RFC-8 coordinate attribute keys of a node.
 *
 * RFC-8 defines `coordinateSystems` and `coordinateTransformations` as
 * attribute keys: a multiscale node carries its systems, a singlescale node
 * its transformations, and a collection may carry either for its members.
 * The accessors here parse those keys into the RFC-5 model types (a
 * coordinate system's `id` and an identifier's `id` were added for RFC-8)
 * and write them back, leaving every other attribute key untouched. Mirrors
 * the Python port's `ngff_zarr.rfc8.coordinates`.
 */

import type {
  Axis,
  CoordinateSystem,
  V06Transform,
} from "../types/zarr_metadata.ts";
import type { OmeNode } from "../types/rfc8.ts";
import { parseV06Transforms, serializeV06Transform } from "./v06_metadata.ts";

/** The two RFC-8 attribute keys this module owns. */
export const COORDINATE_SYSTEMS_KEY = "coordinateSystems";
export const COORDINATE_TRANSFORMATIONS_KEY = "coordinateTransformations";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * The coordinate systems declared in `node.attributes`.
 *
 * RFC-8 makes a system's `name` optional (the required `id` carries the
 * identity), so `name` may come back as an empty string's absence --
 * `undefined`-like -- on the parsed object.
 */
export function coordinateSystems(node: OmeNode): CoordinateSystem[] {
  const raw = node.attributes?.[COORDINATE_SYSTEMS_KEY];
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.filter(isRecord).map((entry) => ({
    ...(typeof entry.name === "string" && { name: entry.name }),
    axes: (Array.isArray(entry.axes) ? entry.axes : []) as Axis[],
    ...(typeof entry.id === "string" && { id: entry.id }),
  }));
}

/** Write `systems` into `node.attributes`, replacing the key. */
export function setCoordinateSystems(
  node: OmeNode,
  systems: CoordinateSystem[],
): void {
  const serialized = systems.map((system) => ({
    ...(system.id !== undefined && { id: system.id }),
    ...(system.name !== undefined && { name: system.name }),
    axes: system.axes.map((axis) => ({
      name: axis.name,
      ...(axis.type !== undefined && { type: axis.type }),
      ...(axis.unit !== undefined && { unit: axis.unit }),
      ...(axis.orientation !== undefined && { orientation: axis.orientation }),
      ...(axis.discrete !== undefined && { discrete: axis.discrete }),
    })),
  }));
  node.attributes = {
    ...(node.attributes ?? {}),
    [COORDINATE_SYSTEMS_KEY]: serialized,
  };
}

/**
 * The coordinate transformations declared in `node.attributes`.
 *
 * `systems` are the coordinate systems the transformations' references
 * resolve against; they default to the ones declared on `node` itself.
 * Identifiers keep the RFC-8 `id` alongside any RFC-5 `name`/`path`.
 */
export function coordinateTransformations(
  node: OmeNode,
  systems?: CoordinateSystem[],
  version: string | undefined = "0.9.dev3",
): V06Transform[] {
  const raw = node.attributes?.[COORDINATE_TRANSFORMATIONS_KEY];
  if (!Array.isArray(raw) || raw.length === 0) {
    return [];
  }
  const resolved = systems ?? coordinateSystems(node);
  return parseV06Transforms(
    raw,
    resolved.map((cs) => cs.name).filter((name): name is string =>
      name !== undefined
    ),
    resolved,
    version,
  );
}

/** Write `transformations` into `node.attributes`, replacing the key. */
export function setCoordinateTransformations(
  node: OmeNode,
  transformations: V06Transform[],
): void {
  node.attributes = {
    ...(node.attributes ?? {}),
    [COORDINATE_TRANSFORMATIONS_KEY]: transformations.map((transform) =>
      serializeV06Transform(transform)
    ),
  };
}
