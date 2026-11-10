// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The rules of the OME-Zarr specification that ngff-zarr checks beyond the
 * official JSON Schemas, the twin of the Python `ngff_zarr.rules`.
 *
 * The rule table is the Python package's `rules.json`. `schemas/ngff_rules.ts`
 * is generated from it and holds what the checks read: each rule's code and
 * the versions whose text states it. The checks run on a document the JSON
 * Schema accepted, so they rely on the shapes the schema enforces.
 */
import { NGFF_RULES, type NgffRule } from "../schemas/ngff_rules.ts";

export const RULES: ReadonlyMap<string, NgffRule> = new Map(
  NGFF_RULES.map((rule) => [rule.code, rule]),
);

/** How far a stored rotation may be from orthonormal, per matrix entry. */
export const ROTATION_TOLERANCE = 1e-5;

/** A broken rule: its code and what breaks it. */
export type Finding = [code: string, message: string];

type Json = Record<string, unknown>;

function records(value: unknown): Json[] {
  return Array.isArray(value) ? (value as Json[]) : [];
}

/**
 * The rules that the OME-Zarr Metadata in `attributes` breaks at `version`.
 */
export function ruleProblems(attributes: Json, version: string): Finding[] {
  const ome = ("ome" in attributes ? attributes.ome : attributes) as Json;
  return [...transformationFindings(ome)].filter(([code]) =>
    RULES.get(code)!.versions.includes(version),
  );
}

function systemsOf(document: Json): Map<string, Json> {
  return new Map(
    records(document.coordinateSystems).map((system) => [
      system.name as string,
      system,
    ]),
  );
}

/**
 * The axis count of the coordinate system a reference names in the same
 * document, or undefined when it points elsewhere.
 */
function axisCount(
  reference: unknown,
  systems: Map<string, Json>,
): number | undefined {
  if (typeof reference !== "object" || reference === null) return undefined;
  const { path, name } = reference as Json;
  if (path) return undefined;
  const system = systems.get(name as string);
  return system === undefined ? undefined : records(system.axes).length;
}

/**
 * Check each coordinate transformation of a document against the axis counts
 * of the coordinate systems it maps between.
 */
function* transformationFindings(ome: Json): Generator<Finding> {
  const scene = ome.scene;
  if (typeof scene === "object" && scene !== null) {
    const systems = systemsOf(scene as Json);
    for (const [index, transform] of records(
      (scene as Json).coordinateTransformations,
    ).entries()) {
      yield* dimensionFindings(
        `scene.coordinateTransformations[${index}]`,
        transform,
        axisCount(transform.input, systems),
        axisCount(transform.output, systems),
      );
    }
  }
  for (const [m, entry] of records(ome.multiscales).entries()) {
    const systems = systemsOf(entry);
    for (const [index, transform] of records(
      entry.coordinateTransformations,
    ).entries()) {
      yield* dimensionFindings(
        `multiscales[${m}].coordinateTransformations[${index}]`,
        transform,
        axisCount(transform.input, systems),
        axisCount(transform.output, systems),
      );
    }
    for (const [d, dataset] of records(entry.datasets).entries()) {
      for (const [index, transform] of records(
        dataset.coordinateTransformations,
      ).entries()) {
        // A dataset's array has as many dimensions as the system it maps to
        // has axes.
        const count = axisCount(transform.output, systems);
        yield* dimensionFindings(
          `multiscales[${m}].datasets[${d}].coordinateTransformations[${index}]`,
          transform,
          count,
          count,
        );
      }
    }
  }
}

/** The axis count after a step of a sequence, when it follows from the step. */
function axesAfter(step: Json, count: number | undefined): number | undefined {
  const kind = step.type;
  if (
    ["identity", "scale", "translation", "rotation", "mapAxis"].includes(
      kind as string,
    )
  ) {
    return count;
  }
  if (kind === "affine" && Array.isArray(step.affine)) {
    return step.affine.length;
  }
  return undefined;
}

/** The determinant of a square matrix, by Gaussian elimination. */
function determinant(matrix: number[][]): number {
  const rows = matrix.map((row) => [...row]);
  let result = 1;
  for (let column = 0; column < rows.length; column++) {
    let pivot = column;
    for (let row = column + 1; row < rows.length; row++) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) {
        pivot = row;
      }
    }
    if (rows[pivot][column] === 0) return 0;
    if (pivot !== column) {
      [rows[pivot], rows[column]] = [rows[column], rows[pivot]];
      result = -result;
    }
    result *= rows[column][column];
    for (let row = column + 1; row < rows.length; row++) {
      const factor = rows[row][column] / rows[column][column];
      for (let k = column; k < rows.length; k++) {
        rows[row][k] -= factor * rows[column][k];
      }
    }
  }
  return result;
}

function orthonormal(matrix: number[][]): boolean {
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      const dot = matrix[i].reduce(
        (sum, value, k) => sum + value * matrix[j][k],
        0,
      );
      if (Math.abs(dot - (i === j ? 1 : 0)) > ROTATION_TOLERANCE) return false;
    }
  }
  return Math.abs(determinant(matrix) - 1) <= ROTATION_TOLERANCE;
}

/**
 * Check a transformation's parameters against `inputCount` input and `outputCount` output
 * axes, either of which may be unknown.
 */
function* dimensionFindings(
  where: string,
  transform: Json,
  inputCount: number | undefined,
  outputCount: number | undefined,
): Generator<Finding> {
  const kind = transform.type;
  const known = inputCount !== undefined && outputCount !== undefined;
  if (kind === "identity") {
    if (known && inputCount !== outputCount) {
      yield [
        "NGFF-CT-015",
        `${where}: an identity maps ${inputCount} axes to ${outputCount}`,
      ];
    }
  } else if (kind === "scale" || kind === "translation") {
    const values = transform[kind];
    if (Array.isArray(values)) {
      const counts = [...new Set([inputCount, outputCount])]
        .filter((count): count is number => count !== undefined)
        .sort((a, b) => a - b);
      for (const count of counts) {
        if (values.length !== count) {
          yield [
            "NGFF-CT-015",
            `${where}: ${kind} gives ${values.length} values for ${count} axes`,
          ];
        }
      }
    }
  } else if (kind === "rotation") {
    if (known && inputCount !== outputCount) {
      yield [
        "NGFF-CT-026",
        `${where}: a rotation maps ${inputCount} axes to ${outputCount}`,
      ];
    }
    const matrix = transform.rotation;
    if (Array.isArray(matrix)) {
      const count = inputCount ?? outputCount ?? matrix.length;
      if (
        matrix.length !== count ||
        matrix.some((row) => (row as unknown[]).length !== count)
      ) {
        yield [
          "NGFF-CT-030",
          `${where}: a rotation of ${count} axes needs a ${count} x ${count} matrix`,
        ];
      } else if (!orthonormal(matrix as number[][])) {
        yield [
          "NGFF-CT-027",
          `${where}: the rotation matrix is not orthonormal with determinant one`,
        ];
      }
    }
  } else if (kind === "affine") {
    const matrix = transform.affine;
    if (Array.isArray(matrix)) {
      const needs: string[] = [];
      if (outputCount !== undefined && matrix.length !== outputCount) {
        needs.push(`${outputCount} rows for ${outputCount} output axes`);
      }
      if (
        inputCount !== undefined &&
        matrix.some((row) => (row as unknown[]).length !== inputCount + 1)
      ) {
        needs.push(
          `rows of ${inputCount + 1} values for ${inputCount} input axes`,
        );
      }
      if (needs.length > 0) {
        yield [
          "NGFF-CT-025",
          `${where}: the affine matrix needs ${needs.join(" and ")}`,
        ];
      }
    }
  } else if (kind === "sequence") {
    let count = inputCount;
    for (const [index, step] of records(transform.transformations).entries()) {
      const after = axesAfter(step, count);
      yield* dimensionFindings(
        `${where}.transformations[${index}]`,
        step,
        count,
        after,
      );
      count = after;
    }
    if (
      count !== undefined &&
      outputCount !== undefined &&
      count !== outputCount
    ) {
      yield [
        "NGFF-CT-015",
        `${where}: the sequence ends with ${count} axes, its output has ${outputCount}`,
      ];
    }
  } else if (kind === "bijection") {
    if (known && inputCount !== outputCount) {
      yield [
        "NGFF-CT-040",
        `${where}: a bijection maps ${inputCount} axes to ${outputCount}`,
      ];
    }
    const directions: [string, number | undefined, number | undefined][] = [
      ["forward", inputCount, outputCount],
      ["inverse", outputCount, inputCount],
    ];
    for (const [direction, from, to] of directions) {
      const inner = transform[direction];
      if (typeof inner === "object" && inner !== null) {
        yield* dimensionFindings(
          `${where}.${direction}`,
          inner as Json,
          from,
          to,
        );
      }
    }
  }
}
