// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC 5 / OME-Zarr v0.6 coordinate-transformation serialization helpers.
 *
 * The in-memory {@link MetadataInterface} model is version-agnostic: axes plus a
 * per-dataset scale and translation. On the wire, v0.6 instead carries
 * `coordinateSystems` and represents each dataset's mapping into the implicit
 * "intrinsic" system as a `sequence` of scale + translation. These helpers
 * convert between the two, mirroring `py/ngff_zarr/v06/zarr_metadata.py`.
 */

import {
  isRfc3AxisModelAllowed,
  NgffVersion,
} from "../types/supported_versions.ts";
import type {
  ByDimensionItem,
  CoordinateSystem,
  CoordinateSystemIdentifier,
  MetadataInterface,
  Transform,
  V06Transform,
} from "../types/zarr_metadata.ts";
import { INTRINSIC_COORDINATE_SYSTEM_NAME } from "../types/zarr_metadata.ts";

/**
 * Build the on-disk v0.6 `multiscales[0]` entry from the version-agnostic
 * in-memory metadata. `processedAxes` are the RFC-processed axes (orientation
 * filtered per enabled RFCs); they back the intrinsic coordinate system so v0.6
 * matches the v0.4/v0.5 writers.
 */
export function buildV06MultiscalesEntry(
  metadata: MetadataInterface,
  processedAxes: Array<Record<string, unknown>>,
  version?: string,
): Record<string, unknown> {
  const coordinateSystems: CoordinateSystem[] =
    metadata.coordinateSystems && metadata.coordinateSystems.length > 0
      ? metadata.coordinateSystems
      : [{ name: INTRINSIC_COORDINATE_SYSTEM_NAME, axes: metadata.axes }];
  const intrinsicName = coordinateSystems[0].name;

  // The first (intrinsic) coordinate system uses the RFC-processed axes; any
  // additional systems (e.g. a user-supplied output space) are serialized
  // verbatim. The RFC-8 `id` rides along whenever the model carries one,
  // matching the Python serializer, so id-based references stay resolvable.
  const serializedCoordinateSystems = coordinateSystems.map((cs, index) => ({
    ...(cs.name !== undefined && { name: cs.name }),
    axes: index === 0 ? processedAxes : cs.axes,
    ...(cs.id !== undefined && { id: cs.id }),
  }));

  const datasets = metadata.datasets.map((ds, index) => {
    let scale: number[] | undefined;
    let translation: number[] | undefined;
    for (const transform of ds.coordinateTransformations) {
      if (transform.type === "scale") {
        scale = transform.scale;
      } else if (transform.type === "translation") {
        translation = transform.translation;
      }
    }

    const transformations: Array<Record<string, unknown>> = [];
    if (scale !== undefined) {
      transformations.push({ type: "scale", scale });
    }
    if (translation !== undefined) {
      transformations.push({ type: "translation", translation });
    }

    return {
      path: ds.path,
      coordinateTransformations: [
        {
          type: "sequence",
          name: `scale${index}_to_${intrinsicName}`,
          input: { path: ds.path },
          output: { name: intrinsicName },
          transformations,
        },
      ],
    };
  });

  const entry: Record<string, unknown> = {
    name: metadata.name,
    coordinateSystems: serializedCoordinateSystems,
    datasets,
  };

  if (
    metadata.coordinateTransformations &&
    metadata.coordinateTransformations.length > 0
  ) {
    // From 0.6 a transform on the multiscales entry maps between two named
    // coordinate systems, and the schema requires both `input` and `output`
    // to name one. Serializing whatever the model holds would produce a store
    // the validated reader rejects.
    // RFC-8 (0.9.dev3) references coordinate systems by id, so an id-only
    // reference is complete there; 0.6 and 0.9.dev1 need the name the
    // schema requires.
    const allowIdReferences = version === NgffVersion.V09dev3;
    metadata.coordinateTransformations.forEach((transform, index) => {
      for (const side of ["input", "output"] as const) {
        const named = transform[side]?.name !== undefined;
        const hasId = transform[side]?.id !== undefined;
        if (!named && !(allowIdReferences && hasId)) {
          throw new Error(
            `multiscales coordinateTransformations[${index}] ` +
              `(${transform.type}) names no ${side} coordinate system; ` +
              "OME-Zarr 0.6 requires every multiscale-level transformation " +
              "to name both its input and its output coordinate system",
          );
        }
      }
      // The reader's own check, against the systems that get serialized: the
      // intrinsic one built above when `metadata.coordinateSystems` is absent.
      try {
        validateV06Transform(transform, coordinateSystems);
      } catch (invalid) {
        throw new Error(
          `multiscales coordinateTransformations[${index}] ` +
            `(${transform.type}) would be written as a transform this ` +
            `package cannot read back: ${
              invalid instanceof Error ? invalid.message : String(invalid)
            }`,
        );
      }
    });
    entry.coordinateTransformations = metadata.coordinateTransformations.map(
      serializeV06Transform,
    );
  }
  if (metadata.type !== undefined) {
    entry.type = metadata.type;
  }
  if (metadata.metadata !== undefined) {
    entry.metadata = metadata.metadata;
  }

  return entry;
}

/** Serialize a single v0.6 transformation to its on-disk dictionary form. */
export function serializeV06Transform(
  transform: V06Transform,
): Record<string, unknown> {
  const out: Record<string, unknown> = { type: transform.type };
  if (transform.name !== undefined) {
    out.name = transform.name;
  }
  if (transform.input !== undefined) {
    const input = identifierToDict(transform.input);
    if (input !== undefined) {
      out.input = input;
    }
  }
  if (transform.output !== undefined) {
    const output = identifierToDict(transform.output);
    if (output !== undefined) {
      out.output = output;
    }
  }

  switch (transform.type) {
    case "identity":
      break;
    case "scale":
      out.scale = transform.scale;
      break;
    case "translation":
      out.translation = transform.translation;
      break;
    case "rotation":
      out.rotation = transform.rotation;
      if (transform.path !== undefined) {
        out.path = transform.path;
      }
      break;
    case "affine":
      out.affine = transform.affine;
      if (transform.path !== undefined) {
        out.path = transform.path;
      }
      break;
    case "coordinates":
    case "displacements":
      out.path = transform.path;
      if (transform.interpolation !== undefined) {
        out.interpolation = transform.interpolation;
      }
      break;
    case "mapAxis":
      out.mapAxis = transform.mapAxis;
      break;
    case "projectAxis":
      if (transform.droppedInputs !== undefined) {
        out.droppedInputs = transform.droppedInputs;
      }
      if (transform.createdOutputs !== undefined) {
        out.createdOutputs = transform.createdOutputs;
      }
      break;
    case "byDimension":
      out.transformations = transform.transformations.map((item) => ({
        transformation: serializeV06Transform(item.transformation),
        inputAxes: item.inputAxes,
        outputAxes: item.outputAxes,
      }));
      break;
    case "bijection":
      out.forward = serializeV06Transform(transform.forward);
      out.inverse = serializeV06Transform(transform.inverse);
      break;
    case "sequence":
      out.transformations = transform.transformations.map(
        serializeV06Transform,
      );
      break;
  }

  return out;
}

/**
 * Parse a list of (possibly nested) v0.6 transformation dictionaries into typed
 * {@link V06Transform}s. `coordinateSystemNames` gates which `input`/`output`
 * `name` references are retained, matching the Python `_parse_transforms`: a
 * `name` is kept only when it identifies a known coordinate system, while a
 * `path` (a dataset array reference) is always kept.
 */
export function parseV06Transforms(
  raw: Array<Record<string, unknown>>,
  coordinateSystemNames: (string | undefined)[],
  coordinateSystems?: CoordinateSystem[],
  version?: string,
): V06Transform[] {
  return raw.map((entry) =>
    parseV06Transform(entry, coordinateSystemNames, coordinateSystems, version)
  );
}

function parseV06Transform(
  entry: Record<string, unknown>,
  coordinateSystemNames: (string | undefined)[],
  coordinateSystems?: CoordinateSystem[],
  version?: string,
): V06Transform {
  const type = String(entry.type);
  let transform: V06Transform;

  switch (type) {
    case "identity":
      transform = { type: "identity" };
      break;
    case "scale":
      transform = { type: "scale", scale: asNumberArray(entry.scale, "scale") };
      break;
    case "translation":
      transform = {
        type: "translation",
        translation: asNumberArray(entry.translation, "translation"),
      };
      break;
    case "rotation":
      transform = {
        type: "rotation",
        rotation: asNumberMatrix(entry.rotation, "rotation"),
      };
      if (typeof entry.path === "string") {
        transform.path = entry.path;
      }
      break;
    case "affine":
      transform = {
        type: "affine",
        affine: asNumberMatrix(entry.affine, "affine"),
      };
      if (typeof entry.path === "string") {
        transform.path = entry.path;
      }
      break;
    case "coordinates":
    case "displacements": {
      if (typeof entry.path !== "string") {
        throw new Error(
          `Invalid ${type} transform: 'path' must be a string`,
        );
      }
      transform = type === "coordinates"
        ? { type: "coordinates", path: entry.path }
        : { type: "displacements", path: entry.path };
      if (typeof entry.interpolation === "string") {
        transform.interpolation = entry.interpolation;
      }
      break;
    }
    case "mapAxis":
      transform = {
        type: "mapAxis",
        mapAxis: asIntegerArray(entry.mapAxis, "mapAxis"),
      };
      break;
    case "projectAxis": {
      transform = { type: "projectAxis" };
      if (entry.droppedInputs !== undefined) {
        transform.droppedInputs = asIntegerArray(
          entry.droppedInputs,
          "droppedInputs",
        );
      }
      if (entry.createdOutputs !== undefined) {
        transform.createdOutputs = asIntegerArray(
          entry.createdOutputs,
          "createdOutputs",
        );
      }
      break;
    }
    case "byDimension": {
      if (!Array.isArray(entry.transformations)) {
        throw new Error(
          "Invalid byDimension transform: 'transformations' must be an array",
        );
      }
      const items: ByDimensionItem[] = entry.transformations.map(
        (rawItem: unknown) => {
          const item = rawItem as Record<string, unknown>;
          if (item === null || typeof item !== "object") {
            throw new Error(
              "Invalid byDimension transform: each item must be an object " +
                "holding 'transformation', 'inputAxes' and 'outputAxes'",
            );
          }
          return {
            transformation: parseV06Transform(
              item.transformation as Record<string, unknown>,
              coordinateSystemNames,
              coordinateSystems,
              version,
            ),
            // ngff-zarr 0.29.0 wrote these two keys in snake_case; the spec
            // and the 0.6rc0 schema spell them inputAxes and outputAxes, which
            // is what is written now. Both spellings are read.
            inputAxes: asIntegerArray(
              item.inputAxes ?? item.input_axes,
              "byDimension",
            ),
            outputAxes: asIntegerArray(
              item.outputAxes ?? item.output_axes,
              "byDimension",
            ),
          };
        },
      );
      transform = { type: "byDimension", transformations: items };
      break;
    }
    case "bijection":
      transform = {
        type: "bijection",
        forward: parseV06Transform(
          entry.forward as Record<string, unknown>,
          coordinateSystemNames,
          coordinateSystems,
          version,
        ),
        inverse: parseV06Transform(
          entry.inverse as Record<string, unknown>,
          coordinateSystemNames,
          coordinateSystems,
          version,
        ),
      };
      break;
    case "sequence":
      if (!Array.isArray(entry.transformations)) {
        throw new Error(
          "Invalid sequence transform: 'transformations' must be an array",
        );
      }
      transform = {
        type: "sequence",
        transformations: parseV06Transforms(
          entry.transformations as Array<Record<string, unknown>>,
          coordinateSystemNames,
          coordinateSystems,
          version,
        ),
      };
      break;
    default:
      throw new Error(`Unsupported transform type: ${type}`);
  }

  const input = dictToIdentifier(entry.input, coordinateSystemNames);
  if (input !== undefined) {
    transform.input = input;
  }
  const output = dictToIdentifier(entry.output, coordinateSystemNames);
  if (output !== undefined) {
    transform.output = output;
  }
  if (typeof entry.name === "string") {
    transform.name = entry.name;
  }

  validateV06Transform(transform, coordinateSystems ?? [], version);
  return transform;
}

/** Number of axes of the referenced coordinate system, when resolvable. */
function axisCount(
  identifier: CoordinateSystemIdentifier | undefined,
  coordinateSystems: CoordinateSystem[],
): number | undefined {
  if (identifier === undefined) {
    return undefined;
  }
  // An `id` resolves first (RFC-8), then a `name` (RFC-5), mirroring the
  // Python `CoordinateSystemIdentifier.axis_count`.
  if (identifier.id !== undefined) {
    const byId = coordinateSystems.find((cs) => cs.id === identifier.id);
    if (byId !== undefined) {
      return byId.axes.length;
    }
  }
  if (identifier.name === undefined) {
    return undefined;
  }
  return coordinateSystems.find((cs) => cs.name === identifier.name)?.axes
    .length;
}

/**
 * `projectAxis` may add or drop at most this many axes at once, which is the
 * `maxItems` its schema sets on `droppedInputs` and `createdOutputs`. The
 * companion `maximum: 4` on each index is deliberately not mirrored: it follows
 * from the five-axis cap of 0.4 through 0.6, which RFC-3 lifts at 0.9.dev1, so
 * an index is bounded by the coordinate system it points into rather than by a
 * constant.
 */
const PROJECT_AXIS_MAX_OPERATIONS = 3;

/** Dimensionality a byDimension item's axis lists must have, if knowable. */
function itemDimensions(transformation: V06Transform): number | undefined {
  switch (transformation.type) {
    case "scale":
      return transformation.scale.length;
    case "translation":
      return transformation.translation.length;
    case "mapAxis":
      return transformation.mapAxis.length;
    default:
      return undefined;
  }
}

/**
 * Check the RFC 5 constraints the schema cannot express, mirroring the Python
 * `validate_transform`: a `mapAxis` must be a permutation, a `byDimension`
 * must cover every output axis exactly once with dimensionally consistent
 * items, and a `bijection` must join coordinate systems of equal
 * dimensionality. Dimension checks against `input`/`output` run only when
 * those resolve to a known coordinate system.
 */
export function validateV06Transform(
  transform: V06Transform,
  coordinateSystems: CoordinateSystem[],
  version?: string,
): void {
  if (transform.type === "mapAxis") {
    const indices = transform.mapAxis;
    if (!indices.every((axis) => Number.isInteger(axis))) {
      throw new Error(
        `mapAxis axis indices must be integers; got [${indices}]`,
      );
    }
    // The 2-to-5 arity mirrors the minItems and maxItems the 0.4 through 0.6
    // schemas set, and follows from their five-axis cap. RFC-3 lifts that cap
    // at 0.9.dev1, whose mapAxis definition sets neither bound, so a
    // permutation over six axes is valid there and the check stands down.
    if (
      !isRfc3AxisModelAllowed(version) &&
      (indices.length < 2 || indices.length > 5)
    ) {
      throw new Error(
        "mapAxis must hold between 2 and 5 indices, one per axis of the " +
          `coordinate systems it permutes; got [${indices}]`,
      );
    }
    const sorted = [...indices].sort((a, b) => a - b);
    if (!sorted.every((value, position) => value === position)) {
      throw new Error(
        "mapAxis must be a permutation holding every zero-based input axis " +
          `index exactly once; got [${indices}]`,
      );
    }
    for (const identifier of [transform.input, transform.output]) {
      const count = axisCount(identifier, coordinateSystems);
      if (count !== undefined && count !== indices.length) {
        throw new Error(
          `mapAxis length ${indices.length} does not match the ${count} ` +
            `axes of coordinate system '${identifier?.name}'`,
        );
      }
    }
  } else if (transform.type === "projectAxis") {
    const dropped = transform.droppedInputs;
    const created = transform.createdOutputs;
    if (dropped === undefined && created === undefined) {
      throw new Error(
        "projectAxis must declare droppedInputs, createdOutputs, or both",
      );
    }
    for (
      const [name, indices] of [
        ["droppedInputs", dropped],
        ["createdOutputs", created],
      ] as const
    ) {
      if (indices === undefined) {
        continue;
      }
      if (!indices.every((axis) => Number.isInteger(axis))) {
        throw new Error(
          `${name} axis indices must be integers; got [${indices}]`,
        );
      }
      if (indices.length === 0) {
        throw new Error(`${name} must hold at least one index`);
      }
      if (indices.some((axis) => axis < 0)) {
        throw new Error(
          `${name} indices are zero-based positions and must not be ` +
            `negative; got [${indices}]`,
        );
      }
      if (new Set(indices).size !== indices.length) {
        throw new Error(`${name} indices must be unique; got [${indices}]`);
      }
      if (indices.length > PROJECT_AXIS_MAX_OPERATIONS) {
        throw new Error(
          `${name} may name at most ${PROJECT_AXIS_MAX_OPERATIONS} axes; ` +
            `got [${indices}]`,
        );
      }
    }
    const droppedCount = dropped?.length ?? 0;
    const createdCount = created?.length ?? 0;
    const inputCount = axisCount(transform.input, coordinateSystems);
    const outputCount = axisCount(transform.output, coordinateSystems);
    if (inputCount !== undefined) {
      if (droppedCount > inputCount) {
        throw new Error(
          `projectAxis drops ${droppedCount} axes, more than the ` +
            `${inputCount} axes of coordinate system ` +
            `'${transform.input?.name}'`,
        );
      }
      const beyond = (dropped ?? []).filter((axis) => axis >= inputCount);
      if (beyond.length > 0) {
        throw new Error(
          `droppedInputs indices [${beyond}] are beyond the ${inputCount} ` +
            `axes of coordinate system '${transform.input?.name}'`,
        );
      }
    }
    if (outputCount !== undefined) {
      if (createdCount > outputCount) {
        throw new Error(
          `projectAxis creates ${createdCount} axes, more than the ` +
            `${outputCount} axes of coordinate system ` +
            `'${transform.output?.name}'`,
        );
      }
      const beyond = (created ?? []).filter((axis) => axis >= outputCount);
      if (beyond.length > 0) {
        throw new Error(
          `createdOutputs indices [${beyond}] are beyond the ${outputCount} ` +
            `axes of coordinate system '${transform.output?.name}'`,
        );
      }
    }
    if (inputCount !== undefined && outputCount !== undefined) {
      const expected = inputCount - droppedCount + createdCount;
      if (expected !== outputCount) {
        throw new Error(
          `projectAxis maps ${inputCount} input axes to ${expected} output ` +
            `axes, but coordinate system '${transform.output?.name}' ` +
            `declares ${outputCount}`,
        );
      }
    }
  } else if (transform.type === "sequence") {
    for (const nested of transform.transformations) {
      validateV06Transform(nested, coordinateSystems);
    }
  } else if (transform.type === "byDimension") {
    const inputCount = axisCount(transform.input, coordinateSystems);
    const seenOutputAxes = new Set<number>();
    for (const item of transform.transformations) {
      // The reader validates each child as it parses it.
      validateV06Transform(item.transformation, coordinateSystems);
      const axes = [...item.inputAxes, ...item.outputAxes];
      if (!axes.every((axis) => Number.isInteger(axis))) {
        throw new Error(
          `byDimension axis indices must be integers; got [${axes}]`,
        );
      }
      if (axes.some((axis) => axis < 0)) {
        throw new Error(
          `byDimension axis indices must be non-negative; got [${axes}]`,
        );
      }
      if (
        inputCount !== undefined &&
        item.inputAxes.some((axis) => axis >= inputCount)
      ) {
        throw new Error(
          `byDimension input axes [${item.inputAxes}] exceed the ` +
            `${inputCount} axes of coordinate system '${transform.input?.name}'`,
        );
      }
      for (const axis of item.outputAxes) {
        if (seenOutputAxes.has(axis)) {
          throw new Error(
            "byDimension output axes must each be produced by exactly one " +
              `transformation; axis ${axis} appears more than once`,
          );
        }
        seenOutputAxes.add(axis);
      }
      const dimensions = itemDimensions(item.transformation);
      if (
        dimensions !== undefined &&
        (item.inputAxes.length !== dimensions ||
          item.outputAxes.length !== dimensions)
      ) {
        throw new Error(
          `byDimension item of type '${item.transformation.type}' is ` +
            `${dimensions}-dimensional but maps ${item.inputAxes.length} ` +
            `input axes to ${item.outputAxes.length} output axes`,
        );
      }
    }
    const count = axisCount(transform.output, coordinateSystems);
    if (count !== undefined) {
      const expected = Array.from({ length: count }, (_, axis) => axis);
      if (
        seenOutputAxes.size !== count ||
        !expected.every((axis) => seenOutputAxes.has(axis))
      ) {
        throw new Error(
          "byDimension items must cover every output axis exactly once; " +
            `coordinate system '${transform.output?.name}' has ${count} axes ` +
            `but the items produce [${
              [...seenOutputAxes].sort((a, b) => a - b)
            }]`,
        );
      }
    }
  } else if (transform.type === "bijection") {
    validateV06Transform(transform.forward, coordinateSystems);
    validateV06Transform(transform.inverse, coordinateSystems);
    const inputCount = axisCount(transform.input, coordinateSystems);
    const outputCount = axisCount(transform.output, coordinateSystems);
    if (
      inputCount !== undefined && outputCount !== undefined &&
      inputCount !== outputCount
    ) {
      throw new Error(
        "bijection input and output coordinate systems must have the same " +
          `dimensionality; got ${inputCount} and ${outputCount}`,
      );
    }
  }
}

/**
 * Extract the effective scale and translation from a dataset's v0.6
 * transformations (unwrapping a `sequence`), defaulting to identity. Mirrors
 * the convenience extraction in the Python `_from_zarr_attrs`.
 */
export function extractScaleTranslation(
  transforms: V06Transform[],
  dims: string[],
): { scale: number[]; translation: number[] } {
  let scale = dims.map(() => 1.0);
  let translation = dims.map(() => 0.0);

  for (const transform of transforms) {
    if (transform.type === "sequence") {
      for (const sub of transform.transformations) {
        if (sub.type === "scale") {
          scale = sub.scale;
        } else if (sub.type === "identity") {
          // Within a sequence an identity resets only the scale, not the
          // translation, matching the Python reference `_from_zarr_attrs`. A
          // sibling translation composes with the identity (identity ∘
          // translation = translation), so it must be preserved. Spec-conformant
          // dataset sequences are `[scale, translation]` and never include an
          // identity, so this branch is defensive.
          scale = dims.map(() => 1.0);
        } else if (sub.type === "translation") {
          translation = sub.translation;
        }
      }
    } else if (transform.type === "scale") {
      scale = transform.scale;
    } else if (transform.type === "translation") {
      translation = transform.translation;
    } else if (transform.type === "identity") {
      scale = dims.map(() => 1.0);
      translation = dims.map(() => 0.0);
    }
  }

  return { scale, translation };
}

/**
 * Reduce v0.6 top-level transformations to the simple scale/translation subset
 * permitted by v0.4/v0.5, dropping richer transforms (rotation, affine,
 * sequence) and the coordinate-system references they carry. Used when writing
 * a value read at v0.6 back out at an older version.
 */
export function legacyTopLevelTransforms(
  transforms: V06Transform[] | undefined,
): Transform[] | undefined {
  if (!transforms) {
    return undefined;
  }
  const simple: Transform[] = [];
  for (const transform of transforms) {
    if (transform.type === "scale") {
      simple.push({ type: "scale", scale: transform.scale });
    } else if (transform.type === "translation") {
      simple.push({ type: "translation", translation: transform.translation });
    }
  }
  return simple.length > 0 ? simple : undefined;
}

/**
 * Validate that a parsed transform field is a flat array of numbers. The reader
 * consumes untrusted on-disk metadata, so a malformed value (missing, a string,
 * a nested array) is rejected with a clear message rather than passed through an
 * unchecked `as number[]` cast that surfaces as a confusing error downstream.
 */
function asNumberArray(value: unknown, field: string): number[] {
  if (
    !Array.isArray(value) || !value.every((v) => typeof v === "number")
  ) {
    throw new Error(
      `Invalid ${field} transform: '${field}' must be an array of numbers`,
    );
  }
  return value as number[];
}

/** Validate that a parsed transform field is a flat array of integers. */
function asIntegerArray(value: unknown, field: string): number[] {
  if (
    !Array.isArray(value) ||
    !value.every((v) => typeof v === "number" && Number.isInteger(v))
  ) {
    throw new Error(
      `Invalid ${field} transform: expected an array of integers`,
    );
  }
  return value as number[];
}

/** Validate that a parsed transform field is a 2D array of numbers. */
function asNumberMatrix(value: unknown, field: string): number[][] {
  if (
    !Array.isArray(value) ||
    !value.every(
      (row) => Array.isArray(row) && row.every((v) => typeof v === "number"),
    )
  ) {
    throw new Error(
      `Invalid ${field} transform: '${field}' must be a 2D array of numbers`,
    );
  }
  return value as number[][];
}

function identifierToDict(
  id: CoordinateSystemIdentifier,
): Record<string, unknown> | undefined {
  const out: Record<string, unknown> = {};
  if (id.path !== undefined) {
    out.path = id.path;
  }
  if (id.name !== undefined) {
    out.name = id.name;
  }
  if (id.id !== undefined) {
    out.id = id.id;
  }
  // An identifier carrying no field at all is meaningless and invalid at
  // every version (the v0.6 schema requires a path or a name, RFC-8 an id);
  // omit it rather than emitting `input: {}` / `output: {}`.
  return Object.keys(out).length > 0 ? out : undefined;
}

function dictToIdentifier(
  value: unknown,
  coordinateSystemNames: (string | undefined)[],
): CoordinateSystemIdentifier | undefined {
  if (value === null || typeof value !== "object") {
    return undefined;
  }
  const dict = value as Record<string, unknown>;
  const id: CoordinateSystemIdentifier = {};
  if (
    typeof dict.name === "string" && coordinateSystemNames.includes(dict.name)
  ) {
    id.name = dict.name;
  }
  if (typeof dict.path === "string") {
    id.path = dict.path;
  } else if (dict.path !== null && typeof dict.path === "object") {
    // An RFC-8 external reference carries a typed path object.
    id.path = dict.path as { type: string; path: string };
  }
  if (typeof dict.id === "string") {
    id.id = dict.id;
  }
  return id.name !== undefined || id.path !== undefined || id.id !== undefined
    ? id
    : undefined;
}
