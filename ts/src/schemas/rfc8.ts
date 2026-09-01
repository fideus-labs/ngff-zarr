// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Zod schemas for the RFC-8 node model (OME-Zarr 0.9.dev3).
 *
 * Loose objects throughout: RFC-8 declares node types, attribute keys and
 * path types as extension points, so unknown and prefixed keys must pass
 * through untouched. The node schema is recursive (`nodes` holds nodes) and
 * therefore stays a plain `z.lazy` rather than joining the compiled-schema
 * set. The Python port validates the same shape with the repo-authored
 * `spec/rfc/8/node.schema.json`.
 */

import { z } from "zod";

/** The RFC-8 id production, shared by node ids and reference ids. */
export const Rfc8IdSchema = z.string().regex(
  /^[a-zA-Z0-9\-_.]+$/,
  "an RFC-8 id must match [a-zA-Z0-9-_.]+",
);

/** RFC-8 `Path`: `zarr`, `json`, or a prefixed extension type. */
export const OmePathSchema = z.strictObject({
  type: z.string().min(1).refine(
    (value) => value === "zarr" || value === "json" || /^[^:]+:.+$/.test(value),
    "a path type must be 'zarr', 'json', or a prefixed extension type",
  ),
  path: z.string().min(1),
});

/** RFC-8 `Reference`: an id plus, for external references, a path. */
export const ReferenceSchema = z.looseObject({
  id: Rfc8IdSchema,
  path: OmePathSchema.optional(),
});

/**
 * RFC-8 `Node`: `type` and `name` required, everything else optional and
 * open-world. `nodes`-XOR-`path` and id uniqueness are structural rules
 * (`validateCollection`), not schema constraints, matching the split in the
 * Python port.
 */
export const OmeNodeSchema: z.ZodType<unknown> = z.lazy(() =>
  z.looseObject({
    type: z.string().min(1),
    id: Rfc8IdSchema.optional(),
    name: z.string().min(1),
    attributes: z.record(z.string(), z.unknown()).optional(),
    nodes: z.array(OmeNodeSchema).optional(),
    path: OmePathSchema.optional(),
  })
);

/**
 * The root `ome` value of a stored 0.9.dev3 document: a node that also
 * carries the root-only `version` key.
 */
export const OmeNodeDocumentSchema = z.intersection(
  OmeNodeSchema,
  z.looseObject({ version: z.literal("0.9.dev3") }),
);
