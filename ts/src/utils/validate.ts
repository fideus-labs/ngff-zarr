// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Validation of OME-Zarr metadata, the twin of the Python `ngff_zarr.validate`:
 * each group a store holds is checked against the official JSON Schema of
 * each kind of OME-Zarr Metadata it carries. The schemas are the files the
 * Python package bundles, generated into `schemas/ome_zarr_json_schemas.ts`.
 */
import * as zarr from "zarrita";
import type { Schema, Validator } from "@cfworker/json-schema";
import { isV06Version, NgffVersion } from "../types/supported_versions.ts";
import { ruleProblems } from "./rules.ts";

/** One way a group of an OME-Zarr store breaks the specification. */
export interface Problem {
  /** The rule broken, or `schema` for the official JSON Schema. */
  code: string;
  message: string;
  /** The group, as a path below the store root. */
  path: string;
}

function formatProblem(problem: Problem): string {
  const where = problem.path ? ` at '${problem.path}'` : "";
  return `[${problem.code}]${where} ${problem.message}`;
}

/** The metadata of an OME-Zarr store breaks the specification. */
export class ValidationError extends Error {
  /** Every way it does. */
  problems: Problem[];

  constructor(problems: Problem[]) {
    super(problems.map(formatProblem).join("\n"));
    this.name = "ValidationError";
    this.problems = problems;
  }
}

/** The JSON Schema each kind of OME-Zarr Metadata is checked against. */
const SCHEMA_OF_KEY: Record<string, string> = {
  multiscales: "image",
  "image-label": "label",
  plate: "plate",
  well: "well",
  scene: "scene",
  "bioformats2raw.layout": "bf2raw",
};

/**
 * 0.6 pre-release tags that earlier releases wrote and that the bundled
 * schemas no longer list; such a store is checked as 0.6. Mirrors the Python
 * `V06_SUPERSEDED_TAGS`.
 */
const V06_SUPERSEDED_TAGS = ["0.6.dev4"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * `schema` with the `minContains: 1` that a `contains` implies spelled out
 * wherever a `maxContains` sits beside it. The meaning is unchanged.
 * ponytail: @cfworker/json-schema 4.1.1 skips that implicit minimum when
 * `maxContains` is present; 4.1.6 fixes it, drop this after upgrading.
 */
function withExplicitMinContains(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(withExplicitMinContains);
  }
  if (!isRecord(schema)) {
    return schema;
  }
  const copy: Record<string, unknown> = Object.fromEntries(
    Object.entries(schema).map(([key, value]) => [
      key,
      withExplicitMinContains(value),
    ]),
  );
  if ("contains" in copy && "maxContains" in copy && !("minContains" in copy)) {
    copy.minContains = 1;
  }
  return copy;
}

/**
 * The bundled schema tree a version reads from: its own, or for a
 * pre-release, the release it leads to (`0.6rc0` and `0.9.dev1` read `0.6`
 * and `0.9`). Mirrors the Python `_schemas_dir`.
 */
function schemaTree(version: string, trees: Record<string, unknown>): string {
  if (version in trees) {
    return version;
  }
  const preRelease = /^(\d+(?:\.\d+)*)(?:\.?(?:a|b|rc|dev)\d*)+$/.exec(version);
  if (preRelease !== null && preRelease[1] in trees) {
    return preRelease[1];
  }
  throw new Error(
    `No JSON Schema is bundled for OME-Zarr version '${version}'. Bundled ` +
      `versions: ${Object.keys(trees).sort().join(", ")}.`,
  );
}

const validators = new Map<string, Validator | null>();

/** The validator of `model` in a bundled tree, or null if it has none. */
async function validator(
  tree: string,
  model: string,
): Promise<Validator | null> {
  const key = `${tree}/${model}`;
  if (!validators.has(key)) {
    // Loaded on the first validated read only.
    const [{ Validator }, { OME_ZARR_JSON_SCHEMAS }] = await Promise.all([
      import("@cfworker/json-schema"),
      import("../schemas/ome_zarr_json_schemas.ts"),
    ]);
    const schemas = OME_ZARR_JSON_SCHEMAS[tree];
    const root = schemas.find((schema) =>
      String(schema.$id).endsWith(`/${model}.schema`),
    );
    let made: Validator | null = null;
    if (root !== undefined) {
      made = new Validator(withExplicitMinContains(root) as Schema, "2020-12");
      for (const schema of schemas) {
        if (schema !== root) {
          made.addSchema(withExplicitMinContains(schema) as Schema);
        }
      }
    }
    validators.set(key, made);
  }
  return validators.get(key) ?? null;
}

/**
 * Check a group's attributes against the JSON Schema of each kind of OME-Zarr
 * Metadata they hold, at `version`. `expected` names the schema the group is
 * read as (`image`, `scene`): it is checked even when the attributes lack its
 * key.
 */
export async function schemaProblems(
  attributes: Record<string, unknown>,
  version: string,
  path = "",
  expected?: string,
): Promise<Problem[]> {
  const found = "ome" in attributes ? attributes.ome : attributes;
  if (!isRecord(found)) {
    return [{ code: "schema", message: "'ome' is not a JSON object", path }];
  }
  let ome = found;
  if ("ome" in attributes && !("version" in ome)) {
    return [
      {
        code: "NGFF-GN-001",
        message: "the 'ome' metadata has no 'version'",
        path,
      },
    ];
  }
  if (V06_SUPERSEDED_TAGS.includes(ome.version as string)) {
    ome = { ...ome, version: "0.6" };
    attributes = { ...attributes, ome };
    version = "0.6";
  }
  const { OME_ZARR_JSON_SCHEMAS } =
    await import("../schemas/ome_zarr_json_schemas.ts");
  const tree = schemaTree(version, OME_ZARR_JSON_SCHEMAS);
  // A JSON round trip gives the document as stored: a reader may hand over
  // keys it left undefined, which the validator refuses as a type.
  const instance = JSON.parse(JSON.stringify(attributes));
  const problems: Problem[] = [];
  for (const [key, model] of Object.entries(SCHEMA_OF_KEY)) {
    const check =
      key in ome || model === expected ? await validator(tree, model) : null;
    if (check === null && model === expected) {
      problems.push({
        code: "schema",
        message: `OME-Zarr ${version} defines no ${model} metadata`,
        path,
      });
    }
    const result = check?.validate(instance);
    if (result !== undefined && !result.valid) {
      // The errors run from the document down to the value at fault; the
      // last one names it.
      const error = result.errors[result.errors.length - 1];
      problems.push({
        code: "schema",
        message: `${model}.schema, ${error.instanceLocation}: ${error.error}`,
        path,
      });
    }
  }
  return problems;
}

/**
 * Check a group's attributes against the JSON Schema and, once the schema
 * accepts them, against the rules beyond it ({@link ruleProblems}).
 */
export async function groupProblems(
  attributes: Record<string, unknown>,
  version: string,
  path = "",
  expected?: string,
): Promise<Problem[]> {
  const problems = await schemaProblems(attributes, version, path, expected);
  if (problems.length > 0) {
    return problems;
  }
  return ruleProblems(
    attributes,
    isV06Version(version) ? NgffVersion.V06 : version,
  ).map(([code, message]) => ({ code, message, path }));
}

/** The group `name` below `node`, or undefined. */
async function group(
  node: zarr.Group<zarr.Readable>,
  name: string,
): Promise<zarr.Group<zarr.Readable> | undefined> {
  // A name from the store's own metadata must stay below the group, also
  // where a backslash or a drive separates paths.
  if (
    name.replaceAll("\\", "/").split("/").includes("..") ||
    /^([A-Za-z]:|[\\/])/.test(name)
  ) {
    return undefined;
  }
  try {
    return await zarr.open(node.resolve(name), { kind: "group" });
  } catch (error) {
    // No group there; any other failure is a read error to report.
    if (zarr.isZarritaError(error, "NotFoundError")) {
      return undefined;
    }
    throw error;
  }
}

/**
 * Check the image group `root` and the label images its `labels` group lists.
 */
export async function storeProblems(
  root: zarr.Group<zarr.Readable>,
  version: string,
): Promise<Problem[]> {
  const problems = await groupProblems(
    root.attrs as Record<string, unknown>,
    version,
    "",
    "image",
  );
  const labels = await group(root, "labels");
  if (labels === undefined) {
    return problems;
  }
  const attributes = labels.attrs as Record<string, unknown>;
  const names = (isRecord(attributes.ome) ? attributes.ome : attributes).labels;
  for (const name of Array.isArray(names) ? names : []) {
    const label =
      typeof name === "string" ? await group(labels, name) : undefined;
    if (label !== undefined) {
      problems.push(
        ...(await groupProblems(
          label.attrs as Record<string, unknown>,
          version,
          `labels/${name}`,
          "image",
        )),
      );
    }
  }
  return problems;
}
