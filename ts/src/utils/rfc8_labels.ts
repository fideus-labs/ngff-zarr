// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Typed views over the RFC-8 `labels` attribute of a multiscale node,
 * mirroring the Python port's `ngff_zarr.rfc8.labels`.
 *
 * A multiscale node with a `labels` attribute is a label map:
 * `labelAttributes` describes individual label values (required
 * `labelValue`, optional RGBA `color`) and `source` references the
 * annotated multiscale images (m:n).
 */

import type { OmeNode, Reference } from "../types/rfc8.ts";

/** The RFC-8 attribute key this module owns. */
export const LABELS_KEY = "labels";

/** Attributes of one label value: the RFC-8 `Label Attributes` object. */
export interface LabelAttributes {
  labelValue: number;
  /** RGBA as four integers between 0 and 255. */
  color?: number[] | undefined;
  /** Additional keys the RFC permits, preserved verbatim. */
  extra?: Record<string, unknown> | undefined;
}

/**
 * The RFC-8 `labels` attribute of a multiscale node. Both fields are
 * optional and an empty object is meaningful: it denotes the node a label
 * map without saying more.
 */
export interface Labels {
  labelAttributes?: LabelAttributes[] | undefined;
  source?: Reference[] | undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Whether `node` declares the `labels` attribute. */
export function isLabelMap(node: OmeNode): boolean {
  return isRecord(node.attributes) && LABELS_KEY in node.attributes;
}

/**
 * The parsed `labels` attribute of `node`, or `undefined`. An empty
 * `labels` object parses to `{}`; use {@link isLabelMap} to tell it apart
 * from an absent attribute.
 */
export function labels(node: OmeNode): Labels | undefined {
  const attributes = node.attributes ?? {};
  if (!(LABELS_KEY in attributes)) {
    return undefined;
  }
  const raw = attributes[LABELS_KEY];
  if (!isRecord(raw)) {
    throw new Error(
      `A 'labels' attribute must be an object; got ${JSON.stringify(raw)}.`,
    );
  }
  const result: Labels = {};
  if (Array.isArray(raw.labelAttributes)) {
    result.labelAttributes = raw.labelAttributes.filter(isRecord).map(
      (entry) => {
        const { labelValue, color, ...extra } = entry;
        return {
          labelValue: labelValue as number,
          ...(color !== undefined && { color: color as number[] }),
          ...(Object.keys(extra).length > 0 && { extra }),
        };
      },
    );
  }
  if (Array.isArray(raw.source)) {
    result.source = raw.source.filter(isRecord).map((entry) => ({
      id: entry.id as string,
      ...(isRecord(entry.path) && {
        path: {
          type: entry.path.type as string,
          path: entry.path.path as string,
        },
      }),
    }));
  }
  return result;
}

/**
 * Write `value` into `node.attributes`, replacing the key. `{}` writes the
 * empty object that denotes a label map; `undefined` removes the attribute.
 */
export function setLabels(node: OmeNode, value: Labels | undefined): void {
  const attributes = { ...(node.attributes ?? {}) };
  if (value === undefined) {
    delete attributes[LABELS_KEY];
    node.attributes = attributes;
    return;
  }
  const serialized: Record<string, unknown> = {};
  if (value.labelAttributes !== undefined) {
    // Extra keys first: the typed fields stay authoritative when an extra
    // entry collides with one of them.
    serialized.labelAttributes = value.labelAttributes.map((entry) => ({
      ...structuredClone(entry.extra ?? {}),
      labelValue: entry.labelValue,
      ...(entry.color !== undefined && { color: [...entry.color] }),
    }));
  }
  if (value.source !== undefined) {
    serialized.source = value.source.map((reference) => ({
      id: reference.id,
      ...(reference.path !== undefined && {
        path: { type: reference.path.type, path: reference.path.path },
      }),
    }));
  }
  attributes[LABELS_KEY] = serialized;
  node.attributes = attributes;
}
