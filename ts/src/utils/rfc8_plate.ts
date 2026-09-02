// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Typed views over the RFC-8 HCS attributes, mirroring the Python port's
 * `ngff_zarr.rfc8.plate`.
 *
 * RFC-8 reworks the high-content-screening metadata as collection
 * attributes: a plate is a collection with a `plate` attribute declaring
 * acquisitions, columns and rows as id-carrying entries; a well is a
 * collection with a `well` attribute referencing one column and one row by
 * id; and an `acquisition` attribute (a reference to one of the plate's
 * acquisitions) sits on individual multiscale nodes (the "wide" layout) or
 * on sub-collections grouping one acquisition's images (the "tall" layout).
 */

import type { OmeNode, Reference } from "../types/rfc8.ts";

/** The RFC-8 attribute keys this module owns. */
export const PLATE_KEY = "plate";
export const WELL_KEY = "well";
export const ACQUISITION_KEY = "acquisition";

/**
 * One acquisition, column or row of a plate: an id plus a label. The RFC's
 * Acquisition, Column and Row interfaces share this shape; ids match
 * `[a-zA-Z0-9-_.]+` and are unique within the JSON document.
 */
export interface PlateEntry {
  id: string;
  name?: string | undefined;
}

/** The `plate` attribute of a plate-level collection. */
export interface PlateAttribute {
  columns: PlateEntry[];
  rows: PlateEntry[];
  acquisitions?: PlateEntry[] | undefined;
}

/**
 * The `well` attribute of a well-level collection. `column` and `row`
 * reference entries declared in the `plate` attribute of the enclosing
 * plate collection.
 */
export interface WellAttribute {
  column: Reference;
  row: Reference;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function entryFrom(value: unknown): PlateEntry {
  if (!isRecord(value)) {
    throw new Error(
      `A plate entry must be an object; got ${JSON.stringify(value)}.`,
    );
  }
  if (typeof value.id !== "string") {
    throw new Error(
      `A plate entry must declare a string 'id'; got ${JSON.stringify(value)}.`,
    );
  }
  if (value.name !== undefined && typeof value.name !== "string") {
    throw new Error(
      `A plate entry 'name' must be a string; got ${
        JSON.stringify(value.name)
      }.`,
    );
  }
  return {
    id: value.id,
    ...(value.name !== undefined && { name: value.name }),
  };
}

function entryToDict(entry: PlateEntry): Record<string, unknown> {
  return {
    id: entry.id,
    ...(entry.name !== undefined && { name: entry.name }),
  };
}

function referenceFrom(value: unknown): Reference {
  if (!isRecord(value)) {
    throw new Error(
      `A well reference must be an object; got ${JSON.stringify(value)}.`,
    );
  }
  if (typeof value.id !== "string") {
    throw new Error(
      `A well reference must declare a string 'id'; got ${
        JSON.stringify(value)
      }.`,
    );
  }
  if (
    value.path !== undefined &&
    (!isRecord(value.path) || typeof value.path.type !== "string" ||
      typeof value.path.path !== "string")
  ) {
    throw new Error(
      `A reference 'path' must be an object with string 'type' and 'path'; ` +
        `got ${JSON.stringify(value.path)}.`,
    );
  }
  return {
    id: value.id,
    ...(isRecord(value.path) && {
      path: {
        type: value.path.type as string,
        path: value.path.path as string,
      },
    }),
  };
}

function referenceToDict(reference: Reference): Record<string, unknown> {
  return {
    id: reference.id,
    ...(reference.path !== undefined && {
      path: { type: reference.path.type, path: reference.path.path },
    }),
  };
}

/** The parsed `plate` attribute of `node`, or `undefined`. */
export function plate(node: OmeNode): PlateAttribute | undefined {
  const raw = node.attributes?.[PLATE_KEY];
  if (raw === undefined || raw === null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new Error(
      `A 'plate' attribute must be an object; got ${JSON.stringify(raw)}.`,
    );
  }
  for (const field of ["columns", "rows", "acquisitions"]) {
    const value = raw[field];
    if (value !== undefined && value !== null && !Array.isArray(value)) {
      throw new Error(
        `A plate '${field}' must be an array; got ${JSON.stringify(value)}.`,
      );
    }
  }
  return {
    columns: ((raw.columns ?? []) as unknown[]).map(entryFrom),
    rows: ((raw.rows ?? []) as unknown[]).map(entryFrom),
    ...(Array.isArray(raw.acquisitions) && {
      acquisitions: raw.acquisitions.map(entryFrom),
    }),
  };
}

/** Write `value` into `node.attributes`, replacing the key. */
export function setPlate(
  node: OmeNode,
  value: PlateAttribute | undefined,
): void {
  const attributes = { ...(node.attributes ?? {}) };
  if (value === undefined) {
    delete attributes[PLATE_KEY];
  } else {
    attributes[PLATE_KEY] = {
      ...(value.acquisitions !== undefined && {
        acquisitions: value.acquisitions.map(entryToDict),
      }),
      columns: value.columns.map(entryToDict),
      rows: value.rows.map(entryToDict),
    };
  }
  node.attributes = attributes;
}

/** The parsed `well` attribute of `node`, or `undefined`. */
export function well(node: OmeNode): WellAttribute | undefined {
  const raw = node.attributes?.[WELL_KEY];
  if (raw === undefined || raw === null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new Error(
      `A 'well' attribute must be an object; got ${JSON.stringify(raw)}.`,
    );
  }
  return {
    column: referenceFrom(raw.column),
    row: referenceFrom(raw.row),
  };
}

/** Write `value` into `node.attributes`, replacing the key. */
export function setWell(
  node: OmeNode,
  value: WellAttribute | undefined,
): void {
  const attributes = { ...(node.attributes ?? {}) };
  if (value === undefined) {
    delete attributes[WELL_KEY];
  } else {
    attributes[WELL_KEY] = {
      column: referenceToDict(value.column),
      row: referenceToDict(value.row),
    };
  }
  node.attributes = attributes;
}

/** The parsed `acquisition` attribute of `node`, or `undefined`. */
export function acquisition(node: OmeNode): Reference | undefined {
  const raw = node.attributes?.[ACQUISITION_KEY];
  if (raw === undefined || raw === null) {
    return undefined;
  }
  return referenceFrom(raw);
}

/** Write `value` into `node.attributes`, replacing the key. */
export function setAcquisition(
  node: OmeNode,
  value: Reference | undefined,
): void {
  const attributes = { ...(node.attributes ?? {}) };
  if (value === undefined) {
    delete attributes[ACQUISITION_KEY];
  } else {
    attributes[ACQUISITION_KEY] = referenceToDict(value);
  }
  node.attributes = attributes;
}
