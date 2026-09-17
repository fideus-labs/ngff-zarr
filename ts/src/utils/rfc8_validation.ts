// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Structural validation of RFC-8 node documents.
 *
 * The seven node/collection rules extend the canonical rule table in
 * `structural_validation.ts` (whose `SpecRule` object declares their
 * identifiers) and are dispatched by {@link validateCollection}, a sibling
 * of `validatePlate` / `validateWell` for the RFC-8 document kind. The rules
 * walk the raw `ome` object rather than the parsed model, so a document the
 * parser refuses (for instance one whose node lacks a `type`) can still be
 * diagnosed with a stable rule identifier.
 *
 * Rule, message and location bytes are kept identical to the Python port's
 * `ngff_zarr.rfc8.validation`, pinned by the shared
 * `py/test/rfc8_collection_cases.json` fixture both test suites drive.
 *
 * Shape errors beyond the rules, such as `nodes` not being an array, are
 * the schema pass's concern; the walker skips what it cannot traverse.
 */

import {
  isPrefixedIdentifier,
  type OmeNode,
  RFC8_ID_PATTERN,
} from "../types/rfc8.ts";
import { isRfc8NodeModel } from "../types/supported_versions.ts";
import { pyRepr } from "./py_format.ts";
import {
  SpecRule,
  ValidateOptions,
  ValidationError,
  ValidationLevel,
} from "./structural_validation.ts";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Render a raw metadata value the way Python's `repr()` renders it in the
 * rule messages: `None` for a missing value, quoted strings, bare numbers.
 */
function pyReprValue(value: unknown): string {
  if (value === undefined || value === null) {
    return "None";
  }
  if (typeof value === "string") {
    return pyRepr(value);
  }
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  return String(value);
}

/**
 * The node types whose schema declares the `nodes` / `path` pair, and for
 * which exactly one of the two must be present. Unknown and extension node
 * types are exempt (open-world). Issue #714's multiscale stage adds
 * `"multiscale"` here rather than introducing a new rule.
 */
export const XOR_NODE_TYPES: ReadonlySet<string> = new Set(["collection"]);

/** The unprefixed path types the core specification defines. */
export const CORE_PATH_TYPES: ReadonlySet<string> = new Set(["zarr", "json"]);

function* iterNodes(
  node: Record<string, unknown>,
  location: string,
): Generator<[string, Record<string, unknown>]> {
  yield [location, node];
  const children = node.nodes;
  if (Array.isArray(children)) {
    for (const [index, child] of children.entries()) {
      if (isRecord(child)) {
        yield* iterNodes(child, `${location}.nodes[${index}]`);
      }
    }
  }
}

function requireNonemptyString(
  document: Record<string, unknown>,
  key: string,
  rule: SpecRule,
): void {
  for (const [location, node] of iterNodes(document, "ome")) {
    const value = node[key];
    if (typeof value !== "string" || value === "") {
      throw new ValidationError(
        rule,
        `Node must declare a non-empty string ${pyRepr(key)}; got ` +
          `${pyReprValue(value)}.`,
        location,
      );
    }
  }
}

/** Every node declares a non-empty string `type` (`node-type-required`). */
export function validateNodeTypeRequired(
  document: Record<string, unknown>,
): void {
  requireNonemptyString(document, "type", SpecRule.NodeTypeRequired);
}

/** Every node declares a non-empty string `name` (`node-name-required`). */
export function validateNodeNameRequired(
  document: Record<string, unknown>,
): void {
  requireNonemptyString(document, "name", SpecRule.NodeNameRequired);
}

/**
 * Sibling nodes carry distinct names (`node-name-unique`); nodes of
 * different collections may share one.
 */
export function validateNodeNameUnique(
  document: Record<string, unknown>,
): void {
  for (const [location, node] of iterNodes(document, "ome")) {
    const children = node.nodes;
    if (!Array.isArray(children)) {
      continue;
    }
    const seen = new Set<string>();
    for (const [index, child] of children.entries()) {
      if (!isRecord(child)) {
        continue;
      }
      const name = child.name;
      if (typeof name !== "string") {
        continue;
      }
      if (seen.has(name)) {
        throw new ValidationError(
          SpecRule.NodeNameUnique,
          `Node name ${pyRepr(name)} is repeated; node names must be ` +
            `unique within the enclosing collection.`,
          `${location}.nodes[${index}]`,
        );
      }
      seen.add(name);
    }
  }
}

/**
 * Every declared id matches `[a-zA-Z0-9-_.]+` (`node-id-format`). The id
 * production is shared with references, so later RFC-8 stages report
 * reference ids under this same rule.
 */
export function validateNodeIdFormat(
  document: Record<string, unknown>,
): void {
  for (const [location, node] of iterNodes(document, "ome")) {
    const value = node.id;
    if (value === undefined || value === null) {
      continue;
    }
    if (typeof value !== "string" || !RFC8_ID_PATTERN.test(value)) {
      throw new ValidationError(
        SpecRule.NodeIdFormat,
        `Id ${pyReprValue(value)} does not match [a-zA-Z0-9-_.]+.`,
        `${location}.id`,
      );
    }
  }
}

/** Ids are unique within the JSON document (`node-id-unique`). */
export function validateNodeIdUnique(
  document: Record<string, unknown>,
): void {
  const seen = new Set<string>();
  for (const [location, node] of iterNodes(document, "ome")) {
    const value = node.id;
    if (typeof value !== "string") {
      continue;
    }
    if (seen.has(value)) {
      throw new ValidationError(
        SpecRule.NodeIdUnique,
        `Id ${pyRepr(value)} is already used in this document; ids must ` +
          `be unique within the JSON document.`,
        `${location}.id`,
      );
    }
    seen.add(value);
  }
}

/**
 * A collection carries exactly one of `nodes` / `path`
 * (`node-nodes-xor-path`). Applies to the node types in
 * {@link XOR_NODE_TYPES}; unknown and extension node types are exempt.
 */
export function validateNodeNodesXorPath(
  document: Record<string, unknown>,
): void {
  for (const [location, node] of iterNodes(document, "ome")) {
    if (typeof node.type !== "string" || !XOR_NODE_TYPES.has(node.type)) {
      continue;
    }
    const hasNodes = node.nodes !== undefined && node.nodes !== null;
    const hasPath = node.path !== undefined && node.path !== null;
    if (hasNodes && hasPath) {
      throw new ValidationError(
        SpecRule.NodeNodesXorPath,
        "Node declares both 'nodes' and 'path'; exactly one must be " +
          "present.",
        location,
      );
    }
    if (!hasNodes && !hasPath) {
      throw new ValidationError(
        SpecRule.NodeNodesXorPath,
        "Node declares neither 'nodes' nor 'path'; exactly one must be " +
          "present.",
        location,
      );
    }
  }
}

/**
 * Every path `type` is `zarr`, `json`, or a prefixed extension identifier
 * (`path-type-known`). Unprefixed identifiers are reserved for the core
 * specification; a prefixed identifier (`prefix:name`) belongs to the
 * extension that registered the prefix and passes as opaque.
 */
export function validatePathTypeKnown(
  document: Record<string, unknown>,
): void {
  for (const [location, node] of iterNodes(document, "ome")) {
    const path = node.path;
    if (!isRecord(path)) {
      continue;
    }
    const value = path.type;
    if (typeof value !== "string" || value === "") {
      throw new ValidationError(
        SpecRule.PathTypeKnown,
        `Path must declare a non-empty string 'type'; got ` +
          `${pyReprValue(value)}.`,
        `${location}.path.type`,
      );
    }
    if (CORE_PATH_TYPES.has(value) || isPrefixedIdentifier(value)) {
      continue;
    }
    throw new ValidationError(
      SpecRule.PathTypeKnown,
      `Path type ${pyRepr(value)} is not 'zarr', 'json', or a prefixed ` +
        `extension type.`,
      `${location}.path.type`,
    );
  }
}

/**
 * The RFC-8 rules in canonical evaluation order (the SpecRule declaration
 * order), dispatched fail-fast by {@link validateCollection}.
 */
const COLLECTION_RULES = [
  validateNodeTypeRequired,
  validateNodeNameRequired,
  validateNodeNameUnique,
  validateNodeIdFormat,
  validateNodeIdUnique,
  validateNodeNodesXorPath,
  validatePathTypeKnown,
] as const;

/**
 * Run the RFC-8 node/collection rules in canonical order, fail-fast.
 *
 * `root` is either the raw `ome` object or a parsed node tree (serialized
 * before checking, so both forms are judged by the same walker). Names and
 * ids are checked per document: nodes a `path` points at live in their own
 * documents and are validated when those are read.
 *
 * `version` gates the rules the RFC-4 way: they run when it is `undefined`
 * (a document with no declared version gets the strict reading) or when
 * `isRfc8NodeModel` says the version stores the node model, and are inert
 * when an earlier version is declared.
 */
export function validateCollection(
  root: Record<string, unknown> | OmeNode,
  version?: string,
  options?: ValidateOptions,
): void {
  const level = options?.level ?? ValidationLevel.Strict;
  if (level === ValidationLevel.SchemaOnly) {
    return;
  }
  if (version !== undefined && !isRfc8NodeModel(version)) {
    return;
  }
  // A parsed node is structurally a document already: it carries the same
  // modeled keys, and the keys the parser tucks under `extra` are ones no
  // rule reads. The Python port serializes its dataclass first; walking the
  // object directly yields the same verdicts, so both forms share one path.
  const document = root as Record<string, unknown>;
  for (const rule of COLLECTION_RULES) {
    rule(document);
  }
}
