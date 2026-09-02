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
 * types are exempt (open-world), and so is `singlescale`: its `path` is
 * optional and it declares no `nodes`.
 */
export const XOR_NODE_TYPES: ReadonlySet<string> = new Set([
  "collection",
  "multiscale",
]);

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

/**
 * Each container that may declare coordinate metadata: a node's
 * `attributes`, and its `scene` attribute, whose systems and
 * transformations share the document's id namespace.
 */
function* iterCoordinateContainers(
  document: Record<string, unknown>,
): Generator<[string, Record<string, unknown>]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    yield [`${location}.attributes`, attributes];
    const scene = attributes.scene;
    if (isRecord(scene)) {
      yield [`${location}.attributes.scene`, scene];
    }
  }
}

function* iterCoordinateSystems(
  document: Record<string, unknown>,
): Generator<[string, Record<string, unknown>]> {
  for (const [base, container] of iterCoordinateContainers(document)) {
    const systems = container.coordinateSystems;
    if (!Array.isArray(systems)) {
      continue;
    }
    for (const [index, system] of systems.entries()) {
      if (isRecord(system)) {
        yield [`${base}.coordinateSystems[${index}]`, system];
      }
    }
  }
}

function* iterTransformReferenceSites(
  document: Record<string, unknown>,
): Generator<[string, unknown]> {
  for (const [base, container] of iterCoordinateContainers(document)) {
    const transformations = container.coordinateTransformations;
    if (!Array.isArray(transformations)) {
      continue;
    }
    for (const [index, transformation] of transformations.entries()) {
      if (!isRecord(transformation)) {
        continue;
      }
      const prefix = `${base}.coordinateTransformations[${index}]`;
      for (const field of ["input", "output"]) {
        yield [`${prefix}.${field}`, transformation[field]];
      }
    }
  }
}

function* iterLabelEntries(
  document: Record<string, unknown>,
): Generator<[string, unknown]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const labels = attributes.labels;
    if (!isRecord(labels)) {
      continue;
    }
    const entries = labels.labelAttributes;
    if (!Array.isArray(entries)) {
      continue;
    }
    // Non-object entries are yielded too: the value rule rejects them, so
    // a `null` or string entry cannot slip past validation.
    for (const [index, entry] of entries.entries()) {
      yield [
        `${location}.attributes.labels.labelAttributes[${index}]`,
        entry,
      ];
    }
  }
}

function* iterLabelSourceSites(
  document: Record<string, unknown>,
): Generator<[string, unknown]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const labels = attributes.labels;
    if (!isRecord(labels)) {
      continue;
    }
    const sources = labels.source;
    if (!Array.isArray(sources)) {
      continue;
    }
    for (const [index, source] of sources.entries()) {
      yield [`${location}.attributes.labels.source[${index}]`, source];
    }
  }
}

/** Every reference site of the document: transformations, then labels. */
function* iterReferenceSites(
  document: Record<string, unknown>,
): Generator<[string, unknown]> {
  yield* iterTransformReferenceSites(document);
  yield* iterLabelSourceSites(document);
}

function* iterPlateAttributes(
  document: Record<string, unknown>,
): Generator<[string, Record<string, unknown>]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const plate = attributes.plate;
    if (isRecord(plate)) {
      yield [`${location}.attributes.plate`, plate];
    }
  }
}

/** The acquisition, column and row entries of every plate. */
function* iterPlateEntries(
  document: Record<string, unknown>,
): Generator<[string, Record<string, unknown>]> {
  for (const [base, plate] of iterPlateAttributes(document)) {
    for (const field of ["acquisitions", "columns", "rows"]) {
      const entries = plate[field];
      if (!Array.isArray(entries)) {
        continue;
      }
      for (const [index, entry] of entries.entries()) {
        if (isRecord(entry)) {
          yield [`${base}.${field}[${index}]`, entry];
        }
      }
    }
  }
}

/**
 * The well column/row and acquisition references of every node. These
 * resolve against the enclosing plate's entries, checked by their dedicated
 * rules; they join the id-format walk but not the generic reference-path
 * rule.
 */
function* iterHcsReferenceSites(
  document: Record<string, unknown>,
): Generator<[string, unknown]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const well = attributes.well;
    if (isRecord(well)) {
      for (const field of ["column", "row"]) {
        yield [`${location}.attributes.well.${field}`, well[field]];
      }
    }
    if (attributes.acquisition !== undefined) {
      yield [`${location}.attributes.acquisition`, attributes.acquisition];
    }
  }
}

/**
 * Every node with the nearest enclosing plate attribute. A node carrying
 * the `plate` attribute is its own nearest plate, so an acquisition
 * reference on the plate collection itself resolves too.
 */
function* iterNodesWithPlate(
  document: Record<string, unknown>,
): Generator<
  [string, Record<string, unknown>, Record<string, unknown> | undefined]
> {
  function* walk(
    node: Record<string, unknown>,
    location: string,
    plate: Record<string, unknown> | undefined,
  ): Generator<
    [string, Record<string, unknown>, Record<string, unknown> | undefined]
  > {
    const attributes = node.attributes;
    if (isRecord(attributes) && isRecord(attributes.plate)) {
      plate = attributes.plate;
    }
    yield [location, node, plate];
    const children = node.nodes;
    if (Array.isArray(children)) {
      for (const [index, child] of children.entries()) {
        if (isRecord(child)) {
          yield* walk(child, `${location}.nodes[${index}]`, plate);
        }
      }
    }
  }
  yield* walk(document, "ome", undefined);
}

function plateEntryIds(
  plate: Record<string, unknown>,
  field: string,
): Set<string> {
  const ids = new Set<string>();
  const entries = plate[field];
  if (Array.isArray(entries)) {
    for (const entry of entries) {
      if (isRecord(entry) && typeof entry.id === "string") {
        ids.add(entry.id);
      }
    }
  }
  return ids;
}

/**
 * Every typed path in the document: each node's own `path` and the `path`
 * of every reference site, so an extension path type is judged the same
 * wherever it appears.
 */
function* iterPathSites(
  document: Record<string, unknown>,
): Generator<[string, unknown]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    yield [`${location}.path`, node.path];
  }
  for (const [location, reference] of iterReferenceSites(document)) {
    if (isRecord(reference)) {
      yield [`${location}.path`, reference.path];
    }
  }
}

/**
 * Every id in the document: declared sites (node ids and coordinate-system
 * ids, which share the document-wide uniqueness namespace) and reference
 * sites (which reuse a declared id and are checked for format only).
 */
function* iterIdSites(
  document: Record<string, unknown>,
): Generator<[string, unknown, boolean]> {
  for (const [location, node] of iterNodes(document, "ome")) {
    if (node.id !== undefined && node.id !== null) {
      yield [`${location}.id`, node.id, true];
    }
  }
  for (const [location, system] of iterCoordinateSystems(document)) {
    if (system.id !== undefined && system.id !== null) {
      yield [`${location}.id`, system.id, true];
    }
  }
  for (const [location, entry] of iterPlateEntries(document)) {
    if (entry.id !== undefined && entry.id !== null) {
      yield [`${location}.id`, entry.id, true];
    }
  }
  for (const [location, reference] of iterReferenceSites(document)) {
    if (
      isRecord(reference) && reference.id !== undefined &&
      reference.id !== null
    ) {
      yield [`${location}.id`, reference.id, false];
    }
  }
  for (const [location, reference] of iterHcsReferenceSites(document)) {
    if (
      isRecord(reference) && reference.id !== undefined &&
      reference.id !== null
    ) {
      yield [`${location}.id`, reference.id, false];
    }
  }
}

/**
 * The ids of the document's multiscale nodes. RFC-8 defines labels `source`
 * entries as references to source multiscales, so only these ids satisfy a
 * pathless source.
 */
function declaredMultiscaleIds(
  document: Record<string, unknown>,
): Set<string> {
  const declared = new Set<string>();
  for (const [, node] of iterNodes(document, "ome")) {
    if (node.type === "multiscale" && typeof node.id === "string") {
      declared.add(node.id);
    }
  }
  return declared;
}

function declaredIds(document: Record<string, unknown>): Set<string> {
  const declared = new Set<string>();
  for (const [, value, isDeclared] of iterIdSites(document)) {
    if (isDeclared && typeof value === "string") {
      declared.add(value);
    }
  }
  return declared;
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
 * Every declared or referenced id matches `[a-zA-Z0-9-_.]+`
 * (`node-id-format`). The id production is shared across the document: node
 * ids, coordinate-system ids, and the reference ids of transformation
 * inputs and outputs are all checked here.
 */
export function validateNodeIdFormat(
  document: Record<string, unknown>,
): void {
  for (const [location, value] of iterIdSites(document)) {
    if (typeof value !== "string" || !RFC8_ID_PATTERN.test(value)) {
      throw new ValidationError(
        SpecRule.NodeIdFormat,
        `Id ${pyReprValue(value)} does not match [a-zA-Z0-9-_.]+.`,
        location,
      );
    }
  }
}

/**
 * Declared ids are unique within the JSON document (`node-id-unique`). Node
 * ids and coordinate-system ids share the namespace; the reference ids of
 * transformation inputs and outputs reuse a declared id and are exempt.
 */
export function validateNodeIdUnique(
  document: Record<string, unknown>,
): void {
  const seen = new Set<string>();
  for (const [location, value, declared] of iterIdSites(document)) {
    if (!declared || typeof value !== "string") {
      continue;
    }
    if (seen.has(value)) {
      throw new ValidationError(
        SpecRule.NodeIdUnique,
        `Id ${pyRepr(value)} is already used in this document; ids must ` +
          `be unique within the JSON document.`,
        location,
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
  for (const [location, path] of iterPathSites(document)) {
    if (!isRecord(path)) {
      continue;
    }
    const value = path.type;
    if (typeof value !== "string" || value === "") {
      throw new ValidationError(
        SpecRule.PathTypeKnown,
        `Path must declare a non-empty string 'type'; got ` +
          `${pyReprValue(value)}.`,
        `${location}.type`,
      );
    }
    if (CORE_PATH_TYPES.has(value) || isPrefixedIdentifier(value)) {
      continue;
    }
    throw new ValidationError(
      SpecRule.PathTypeKnown,
      `Path type ${pyRepr(value)} is not 'zarr', 'json', or a prefixed ` +
        `extension type.`,
      `${location}.type`,
    );
  }
}

/**
 * Every declared coordinate system carries an `id`
 * (`coordinate-system-id-required`). RFC-8 references coordinate systems by
 * id, so a system without one is unreachable; malformed ids are reported by
 * {@link validateNodeIdFormat} instead.
 */
export function validateCoordinateSystemIdRequired(
  document: Record<string, unknown>,
): void {
  for (const [location, system] of iterCoordinateSystems(document)) {
    if (system.id === undefined || system.id === null) {
      throw new ValidationError(
        SpecRule.CoordinateSystemIdRequired,
        "Coordinate system declares no 'id'; RFC-8 references coordinate " +
          "systems by id.",
        location,
      );
    }
  }
}

/**
 * Transformation inputs and outputs are id references
 * (`reference-id-required`). RFC-8 replaces the name-based RFC-5
 * identifiers: each top-level transformation's `input` and `output` must be
 * an object carrying an `id`.
 */
export function validateReferenceIdRequired(
  document: Record<string, unknown>,
): void {
  for (const [location, reference] of iterTransformReferenceSites(document)) {
    if (
      !isRecord(reference) || reference.id === undefined ||
      reference.id === null
    ) {
      throw new ValidationError(
        SpecRule.ReferenceIdRequired,
        "Transformation reference must be an object with an 'id'.",
        location,
      );
    }
  }
  for (const [location, reference] of iterLabelSourceSites(document)) {
    if (
      !isRecord(reference) || reference.id === undefined ||
      reference.id === null
    ) {
      throw new ValidationError(
        SpecRule.ReferenceIdRequired,
        "Label source reference must be an object with an 'id'.",
        location,
      );
    }
  }
}

/**
 * Unresolved references carry a `path` (`reference-path-required`). A
 * reference whose `id` is declared in this document is internal; any other
 * reference is external, and RFC-8 requires external references to locate
 * their document with a `path`. What the path points at is not resolved
 * here. A transformation reference resolves against node and
 * coordinate-system ids; a labels `source` reference designates an
 * annotated image, so only a multiscale node id satisfies it.
 */
export function validateReferencePathRequired(
  document: Record<string, unknown>,
): void {
  const siteGroups: Array<[Generator<[string, unknown]>, Set<string>]> = [
    [iterTransformReferenceSites(document), declaredIds(document)],
    [iterLabelSourceSites(document), declaredMultiscaleIds(document)],
  ];
  for (const [sites, resolvable] of siteGroups) {
    for (const [location, reference] of sites) {
      if (!isRecord(reference)) {
        continue;
      }
      const value = reference.id;
      if (typeof value !== "string" || resolvable.has(value)) {
        continue;
      }
      if (reference.path === undefined || reference.path === null) {
        throw new ValidationError(
          SpecRule.ReferencePathRequired,
          `Reference id ${pyRepr(value)} is not declared in this document, ` +
            `so the reference must carry a 'path'.`,
          location,
        );
      }
    }
  }
}

/**
 * Every label attributes entry declares a numeric `labelValue`
 * (`label-value-required`).
 */
export function validateLabelValueRequired(
  document: Record<string, unknown>,
): void {
  for (const [location, entry] of iterLabelEntries(document)) {
    const value = isRecord(entry) ? entry.labelValue : undefined;
    if (typeof value !== "number") {
      throw new ValidationError(
        SpecRule.LabelValueRequired,
        `Label attributes must declare a numeric 'labelValue'; got ` +
          `${pyReprValue(value)}.`,
        location,
      );
    }
  }
}

/**
 * A declared label `color` is four integers in 0..=255
 * (`label-color-format`): the uint8 red, green, blue and alpha values.
 */
export function validateLabelColorFormat(
  document: Record<string, unknown>,
): void {
  for (const [location, entry] of iterLabelEntries(document)) {
    const color = isRecord(entry) ? entry.color : undefined;
    if (color === undefined || color === null) {
      continue;
    }
    let valid = Array.isArray(color) && color.length === 4;
    if (valid) {
      for (const channel of color as unknown[]) {
        if (
          typeof channel !== "number" || !Number.isInteger(channel) ||
          channel < 0 || channel > 255
        ) {
          valid = false;
          break;
        }
      }
    }
    if (!valid) {
      throw new ValidationError(
        SpecRule.LabelColorFormat,
        "Label color must be an array of four integers between 0 and 255.",
        `${location}.color`,
      );
    }
  }
}

/**
 * A declared scene carries a non-empty `coordinateTransformations` array
 * (`scene-transformations-required`); a scene without one relates nothing.
 */
export function validateSceneTransformationsRequired(
  document: Record<string, unknown>,
): void {
  for (const [location, node] of iterNodes(document, "ome")) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const scene = attributes.scene;
    if (scene === undefined || scene === null) {
      continue;
    }
    const transformations = isRecord(scene)
      ? scene.coordinateTransformations
      : undefined;
    if (!Array.isArray(transformations) || transformations.length === 0) {
      throw new ValidationError(
        SpecRule.SceneTransformationsRequired,
        "Scene must declare a non-empty 'coordinateTransformations' array.",
        `${location}.attributes.scene`,
      );
    }
  }
}

/**
 * Every plate declares its columns and rows
 * (`plate-columns-rows-required`).
 */
export function validatePlateColumnsRowsRequired(
  document: Record<string, unknown>,
): void {
  for (const [location, plate] of iterPlateAttributes(document)) {
    for (const field of ["columns", "rows"]) {
      const entries = plate[field];
      if (!Array.isArray(entries) || entries.length === 0) {
        throw new ValidationError(
          SpecRule.PlateColumnsRowsRequired,
          `Plate must declare a non-empty ${pyRepr(field)} array.`,
          location,
        );
      }
    }
  }
}

/**
 * A well's column and row reference its plate's entries
 * (`well-reference-resolves`).
 */
export function validateWellReferenceResolves(
  document: Record<string, unknown>,
): void {
  for (const [location, node, plate] of iterNodesWithPlate(document)) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const well = attributes.well;
    if (!isRecord(well)) {
      continue;
    }
    for (
      const [field, plural] of [
        ["column", "columns"],
        ["row", "rows"],
      ] as const
    ) {
      const reference = well[field];
      const site = `${location}.attributes.well.${field}`;
      const value = isRecord(reference) ? reference.id : undefined;
      if (typeof value !== "string") {
        throw new ValidationError(
          SpecRule.WellReferenceResolves,
          `Well ${pyRepr(field)} must be a reference with an 'id'.`,
          site,
        );
      }
      if (plate === undefined) {
        throw new ValidationError(
          SpecRule.WellReferenceResolves,
          `Well ${pyRepr(field)} references ${pyRepr(value)}, but no ` +
            `enclosing collection declares a 'plate'.`,
          site,
        );
      }
      if (!plateEntryIds(plate, plural).has(value)) {
        throw new ValidationError(
          SpecRule.WellReferenceResolves,
          `Well ${pyRepr(field)} references ${pyRepr(value)}, which the ` +
            `enclosing plate's ${pyRepr(plural)} do not declare.`,
          site,
        );
      }
    }
  }
}

/**
 * An acquisition reference names a declared acquisition
 * (`acquisition-reference-resolves`).
 */
export function validateAcquisitionReferenceResolves(
  document: Record<string, unknown>,
): void {
  for (const [location, node, plate] of iterNodesWithPlate(document)) {
    const attributes = node.attributes;
    if (!isRecord(attributes)) {
      continue;
    }
    const reference = attributes.acquisition;
    if (reference === undefined || reference === null) {
      continue;
    }
    const site = `${location}.attributes.acquisition`;
    const value = isRecord(reference) ? reference.id : undefined;
    if (typeof value !== "string") {
      throw new ValidationError(
        SpecRule.AcquisitionReferenceResolves,
        "Acquisition must be a reference with an 'id'.",
        site,
      );
    }
    if (plate === undefined) {
      throw new ValidationError(
        SpecRule.AcquisitionReferenceResolves,
        `Acquisition references ${pyRepr(value)}, but no enclosing ` +
          `collection declares a 'plate'.`,
        site,
      );
    }
    if (!plateEntryIds(plate, "acquisitions").has(value)) {
      throw new ValidationError(
        SpecRule.AcquisitionReferenceResolves,
        `Acquisition references ${pyRepr(value)}, which the enclosing ` +
          `plate's 'acquisitions' do not declare.`,
        site,
      );
    }
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
  validateCoordinateSystemIdRequired,
  validateReferenceIdRequired,
  validateReferencePathRequired,
  validateLabelValueRequired,
  validateLabelColorFormat,
  validateSceneTransformationsRequired,
  validatePlateColumnsRowsRequired,
  validateWellReferenceResolves,
  validateAcquisitionReferenceResolves,
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
