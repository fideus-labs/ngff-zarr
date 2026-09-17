// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Conversion between multiscales entries and RFC-8 multiscale node
 * documents, mirroring the Python port's `ngff_zarr.rfc8.multiscale`.
 *
 * At OME-Zarr 0.9.dev3 an image store's root `ome` value is a `multiscale`
 * node: the datasets become `singlescale` child nodes referenced by typed
 * paths, the coordinate systems move into the node's `attributes` and gain
 * required ids, and the per-level transformations reference the
 * singlescale's own id and a coordinate-system id in place of the RFC-5
 * path/name identifiers. Both directions are object-to-object, so the
 * readers, writers and upgraders keep one in-memory model.
 */

import * as zarr from "zarrita";

import { RFC8_ID_PATTERN } from "../types/rfc8.ts";

const INVALID_ID_CHARS = /[^a-zA-Z0-9\-_.]/g;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A unique document id derived from `candidate` (see the Python twin). */
function claimId(candidate: string, used: Set<string>): string {
  const cleaned = candidate.replace(INVALID_ID_CHARS, "-") || "n";
  let claimed = cleaned;
  let suffix = 2;
  while (used.has(claimed)) {
    claimed = `${cleaned}-${suffix}`;
    suffix += 1;
  }
  used.add(claimed);
  return claimed;
}

function referenceToIds(
  identifier: unknown,
  nameToId: Map<string | undefined, string>,
): unknown {
  if (!isRecord(identifier)) {
    return identifier;
  }
  const name = typeof identifier.name === "string"
    ? identifier.name
    : undefined;
  let reference: Record<string, unknown>;
  if (identifier.id !== undefined && identifier.id !== null) {
    reference = { id: identifier.id };
  } else if (nameToId.has(name)) {
    reference = { id: nameToId.get(name) };
  } else {
    return identifier;
  }
  if (identifier.path !== undefined && identifier.path !== null) {
    reference.path = identifier.path;
  }
  return reference;
}

/**
 * Build the 0.9.dev3 `ome` value for a serialized multiscales entry.
 *
 * `omero` is hoisted into the node's attributes (the RFC-8 root has no
 * `ome.omero` slot), every coordinate system gains an id (its own, else one
 * derived from its name), each dataset becomes a `singlescale` child node
 * whose transformations reference its id and a coordinate-system id, and
 * the remaining entry keys ride in the attributes verbatim.
 */
export function multiscaleOmeDict(
  entry: Record<string, unknown>,
  version: string,
  omero?: unknown,
): Record<string, unknown> {
  const rest = { ...entry };
  const name = typeof rest.name === "string" && rest.name !== ""
    ? rest.name
    : "image";
  delete rest.name;
  const systems =
    (Array.isArray(rest.coordinateSystems) ? rest.coordinateSystems : []).map((
      system,
    ) => ({ ...(system as Record<string, unknown>) }));
  delete rest.coordinateSystems;
  const datasets = Array.isArray(rest.datasets)
    ? (rest.datasets as Record<string, unknown>[])
    : [];
  delete rest.datasets;
  const topLevel = Array.isArray(rest.coordinateTransformations)
    ? (rest.coordinateTransformations as Record<string, unknown>[])
    : undefined;
  delete rest.coordinateTransformations;

  const usedIds = new Set<string>();
  for (const system of systems) {
    if (typeof system.id === "string") {
      usedIds.add(system.id);
    }
  }
  const nameToId = new Map<string | undefined, string>();
  for (const system of systems) {
    if (system.id === undefined || system.id === null) {
      system.id = claimId(
        typeof system.name === "string" && system.name !== ""
          ? system.name
          : "system",
        usedIds,
      );
    }
    const systemName = typeof system.name === "string"
      ? system.name
      : undefined;
    if (!nameToId.has(systemName)) {
      nameToId.set(systemName, system.id as string);
    }
  }

  const nodes = datasets.map((dataset) => {
    const path = dataset.path as string;
    const singlescaleId = claimId(path, usedIds);
    const transformations = (Array.isArray(dataset.coordinateTransformations)
      ? (dataset.coordinateTransformations as Record<string, unknown>[])
      : []).map((transform) => ({
        ...transform,
        input: { id: singlescaleId },
        output: referenceToIds(transform.output, nameToId),
      }));
    return {
      type: "singlescale",
      id: singlescaleId,
      name: path,
      path: { type: "zarr", path: `./${path}` },
      attributes: { coordinateTransformations: transformations },
    };
  });

  const attributes: Record<string, unknown> = {
    coordinateSystems: systems,
  };
  if (topLevel !== undefined && topLevel.length > 0) {
    attributes.coordinateTransformations = topLevel.map((transform) => ({
      ...transform,
      input: referenceToIds(transform.input, nameToId),
      output: referenceToIds(transform.output, nameToId),
    }));
  }
  if (omero !== undefined && omero !== null) {
    attributes.omero = omero;
  }
  Object.assign(attributes, rest);

  const ome: Record<string, unknown> = { version, type: "multiscale" };
  if (RFC8_ID_PATTERN.test(name) && !usedIds.has(name)) {
    ome.id = name;
  }
  ome.name = name;
  ome.nodes = nodes;
  ome.attributes = attributes;
  return ome;
}

function datasetPathFrom(node: Record<string, unknown>): string {
  const path = isRecord(node.path) && typeof node.path.path === "string"
    ? node.path.path
    : "";
  return path.startsWith("./") ? path.slice(2) : path;
}

function referenceToNames(
  identifier: unknown,
  idToName: Map<string, string>,
  datasetPath: string | undefined,
): unknown {
  if (!isRecord(identifier)) {
    return identifier;
  }
  const rewritten = { ...identifier };
  const value = rewritten.id;
  if (typeof value === "string" && idToName.has(value)) {
    if (rewritten.name === undefined) {
      rewritten.name = idToName.get(value);
    }
  } else if (
    datasetPath !== undefined &&
    (rewritten.path === undefined || rewritten.path === null)
  ) {
    // The singlescale's own id: the v0.6 shape records the dataset path.
    rewritten.path = datasetPath;
  }
  return rewritten;
}

/**
 * Rebuild the 0.9.dev1-shaped multiscales entry of a multiscale node.
 *
 * Returns the entry and the hoisted `omero`. Coordinate systems keep their
 * ids and gain a name when RFC-8 omitted it (the in-memory model resolves
 * systems by name), and each singlescale child becomes a dataset whose
 * transformations carry both the RFC-8 ids and the name/path fields the
 * 0.9.dev1 reader keys on. `singlescaleTransforms` supplies the
 * transformations of children that carry none inline (the RFC-8
 * distributed layout), keyed by dataset path.
 */
export function multiscalesEntryFromOme(
  ome: Record<string, unknown>,
  singlescaleTransforms?: Map<string, unknown[]>,
): { entry: Record<string, unknown>; omero?: unknown } {
  const attributes = { ...(isRecord(ome.attributes) ? ome.attributes : {}) };
  const systems =
    (Array.isArray(attributes.coordinateSystems)
      ? attributes.coordinateSystems
      : []).map((system) => ({ ...(system as Record<string, unknown>) }));
  delete attributes.coordinateSystems;
  const idToName = new Map<string, string>();
  for (const system of systems) {
    if (typeof system.name !== "string" || system.name === "") {
      system.name = system.id;
    }
    if (typeof system.id === "string") {
      idToName.set(system.id, system.name as string);
    }
  }

  const datasets = (Array.isArray(ome.nodes) ? ome.nodes : [])
    .filter(isRecord)
    .map((node) => {
      const path = datasetPathFrom(node);
      const nodeAttributes = isRecord(node.attributes) ? node.attributes : {};
      let transformations = Array.isArray(
          nodeAttributes.coordinateTransformations,
        )
        ? (nodeAttributes.coordinateTransformations as unknown[])
        : undefined;
      if (
        (transformations === undefined || transformations.length === 0) &&
        singlescaleTransforms?.has(path)
      ) {
        transformations = singlescaleTransforms.get(path);
      }
      const rewritten = (transformations ?? []).filter(isRecord).map(
        (transform) => ({
          ...transform,
          input: referenceToNames(transform.input, idToName, path),
          output: referenceToNames(transform.output, idToName, undefined),
        }),
      );
      const dataset: Record<string, unknown> = { path };
      if (rewritten.length > 0) {
        dataset.coordinateTransformations = rewritten;
      }
      return dataset;
    });

  const entry: Record<string, unknown> = {
    name: typeof ome.name === "string" ? ome.name : "image",
    coordinateSystems: systems,
    datasets,
  };
  const topLevel = attributes.coordinateTransformations;
  delete attributes.coordinateTransformations;
  if (Array.isArray(topLevel) && topLevel.length > 0) {
    entry.coordinateTransformations = topLevel.filter(isRecord).map(
      (transform) => ({
        ...transform,
        input: referenceToNames(transform.input, idToName, undefined),
        output: referenceToNames(transform.output, idToName, undefined),
      }),
    );
  }
  const omero = attributes.omero;
  delete attributes.omero;
  Object.assign(entry, attributes);
  return { entry, ...(omero !== undefined && { omero }) };
}

/**
 * Transformations of singlescale children that carry none inline: the RFC-8
 * distributed layout stores each level's singlescale document in the level's
 * own `zarr.json`. Collected keyed by dataset path.
 */
export async function distributedSinglescaleTransforms(
  omeValue: Record<string, unknown>,
  store: zarr.Readable,
): Promise<Map<string, unknown[]>> {
  const collected = new Map<string, unknown[]>();
  for (const node of Array.isArray(omeValue.nodes) ? omeValue.nodes : []) {
    if (
      typeof node !== "object" || node === null ||
      (node as Record<string, unknown>).type !== "singlescale"
    ) {
      continue;
    }
    const record = node as Record<string, unknown>;
    const attributes = record.attributes as
      | Record<string, unknown>
      | undefined;
    const inline = attributes?.coordinateTransformations;
    if (Array.isArray(inline) && inline.length > 0) {
      continue;
    }
    const rawPath = (record.path as Record<string, unknown> | undefined)
      ?.path;
    if (typeof rawPath !== "string") {
      continue;
    }
    const path = rawPath.startsWith("./") ? rawPath.slice(2) : rawPath;
    let child;
    try {
      child = await zarr.open(zarr.root(store).resolve(path));
    } catch (error) {
      // An absent child means this level has no distributed document; any
      // other failure (a corrupt document, transport) must surface rather
      // than silently dropping the level's transformations.
      if (error instanceof zarr.NodeNotFoundError) {
        continue;
      }
      throw error;
    }
    const childOme = (child.attrs as Record<string, unknown>)?.ome as
      | Record<string, unknown>
      | undefined;
    const transforms =
      (childOme?.attributes as Record<string, unknown> | undefined)
        ?.coordinateTransformations;
    if (Array.isArray(transforms) && transforms.length > 0) {
      collected.set(path, transforms);
    }
  }
  return collected;
}
