// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Rotation and affine matrices stored as Zarr arrays.
 *
 * RFC-5 lets a `rotation` or `affine` carry its matrix inline in the JSON
 * metadata or in a 2D Zarr array the transform names by `path`, relative to
 * the multiscales group; the schema accepts one or the other, never both. The
 * writer always takes the array form. A JSON number is decimal text, kept only
 * to the precision of every tool that parses and re-serializes the document,
 * while a float64 array holds each entry bit for bit.
 *
 * The reader loads the array back into the inline field and keeps `path`, so
 * the in-memory model always carries the values and a rewrite reuses the
 * array. Mirrors `py/ngff_zarr/_matrix_transform_arrays.py`.
 */

import * as zarr from "zarrita";
import { bytesOnlyCodecs } from "./codecs.ts";

/**
 * The group generated matrix arrays are written under, as in the
 * specification's own example.
 */
export const MATRIX_ARRAY_GROUP = "coordinateTransformations";

/**
 * The transformation types parameterized by a matrix. Each holds its matrix
 * inline under the key that is also its type.
 */
const MATRIX_TYPES = new Set(["rotation", "affine"]);

const NODE_NAME_CHARACTERS = /^[A-Za-z0-9._-]+$/;

type JsonObject = Record<string, unknown>;

/** A row-major float64 matrix to write as a Zarr array. */
export interface MatrixArray {
  shape: [number, number];
  data: Float64Array;
}

/** Matrices to write, keyed by path relative to the multiscales group. */
export type MatrixArrays = Map<string, MatrixArray>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Every rotation and affine in `transforms`, depth first in document order.
 *
 * Descends into the wrapper transforms: `sequence` and `byDimension` items,
 * and a `bijection`'s `forward` then `inverse`.
 */
function* iterMatrixTransforms(transforms: unknown): Generator<JsonObject> {
  if (!Array.isArray(transforms)) {
    return;
  }
  for (const transform of transforms) {
    if (!isObject(transform)) {
      continue;
    }
    const kind = transform.type;
    if (typeof kind === "string" && MATRIX_TYPES.has(kind)) {
      yield transform;
    } else if (kind === "sequence") {
      yield* iterMatrixTransforms(transform.transformations);
    } else if (kind === "byDimension") {
      const items = Array.isArray(transform.transformations)
        ? transform.transformations
        : [];
      yield* iterMatrixTransforms(
        items.filter(isObject).map((item) => item.transformation),
      );
    } else if (kind === "bijection") {
      yield* iterMatrixTransforms([transform.forward, transform.inverse]);
    }
  }
}

function describe(transform: JsonObject): string {
  const name = transform.name;
  return `${transform.type} transformation` +
    (typeof name === "string" && name !== "" ? ` '${name}'` : "");
}

function namedPath(transform: JsonObject): string | undefined {
  const path = transform.path;
  return typeof path === "string" && path !== "" ? path : undefined;
}

/**
 * Whether `name` is a portable Zarr node name: the Zarr v3 node name rules
 * with the recommended character set, so no `/`, not only periods, and no
 * reserved `__` prefix.
 */
function isNodeName(name: unknown): name is string {
  return typeof name === "string" && NODE_NAME_CHARACTERS.test(name) &&
    name.replace(/\./g, "") !== "" && !name.startsWith("__");
}

/** `path`, refused when it could resolve outside the multiscales group. */
function checkedPath(path: string, transform: JsonObject): string {
  if (
    path.startsWith("/") ||
    path.split("/").some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new Error(
      `${describe(transform)} names the array path '${path}'; a matrix ` +
        "array sits at a relative path inside the multiscales group, with " +
        "no empty, '.' or '..' segments",
    );
  }
  return path;
}

/** Whether `path` is taken, or is an ancestor or descendant of a taken one. */
function isClaimed(path: string, claimed: Set<string>): boolean {
  for (const other of claimed) {
    if (
      other === path || other.startsWith(`${path}/`) ||
      path.startsWith(`${other}/`)
    ) {
      return true;
    }
  }
  return false;
}

function generatedPath(transform: JsonObject, claimed: Set<string>): string {
  const base = isNodeName(transform.name)
    ? transform.name
    : String(transform.type);
  let path = `${MATRIX_ARRAY_GROUP}/${base}`;
  let suffix = 0;
  while (isClaimed(path, claimed)) {
    suffix += 1;
    path = `${MATRIX_ARRAY_GROUP}/${base}_${suffix}`;
  }
  return path;
}

function toMatrixArray(values: unknown, transform: JsonObject): MatrixArray {
  if (
    !Array.isArray(values) ||
    !values.every((row) =>
      Array.isArray(row) && row.every((entry) => typeof entry === "number")
    )
  ) {
    throw new Error(
      `${describe(transform)} does not hold a matrix of numbers`,
    );
  }
  const rows = values as number[][];
  const columns = rows[0]?.length ?? 0;
  if (rows.some((row) => row.length !== columns)) {
    throw new Error(
      `${describe(transform)} does not hold a rectangular matrix: its rows ` +
        `have lengths ${rows.map((row) => row.length).join(", ")}`,
    );
  }
  if (rows.length === 0 || columns === 0) {
    throw new Error(
      `${describe(transform)} must hold a non-empty 2D matrix; got shape ` +
        `${rows.length}x${columns}`,
    );
  }
  return {
    shape: [rows.length, columns],
    data: Float64Array.from(rows.flat()),
  };
}

function sameMatrix(a: MatrixArray, b: MatrixArray): boolean {
  return a.shape[0] === b.shape[0] && a.shape[1] === b.shape[1] &&
    a.data.every((value, index) => value === b.data[index]);
}

/**
 * The array paths the rotations and affines in a serialized
 * `coordinateTransformations` list already name.
 */
export function namedMatrixPaths(transforms: unknown): Set<string> {
  const paths = new Set<string>();
  for (const transform of iterMatrixTransforms(transforms)) {
    const path = namedPath(transform);
    if (path !== undefined) {
      paths.add(path);
    }
  }
  return paths;
}

/**
 * Move each rotation and affine matrix in `transforms` into an array.
 *
 * `transforms` is a serialized `coordinateTransformations` list and is edited
 * in place: a transform holding its matrix inline loses that field and gains
 * `path`. Returns the matrices to write, keyed by their path relative to the
 * multiscales group.
 *
 * A transform keeps a `path` it already names, as one read back from a store
 * does, so a rewrite updates that array. Otherwise the path is
 * `coordinateTransformations/<name>`, with the transform type standing in for
 * a name that is absent or not a portable Zarr node name, and a numeric suffix
 * added when an earlier transform claimed the path. A transform that names a
 * `path` but holds no matrix refers to an array already in the store and is
 * left as it is.
 */
export function externalizeMatrixTransforms(transforms: unknown): MatrixArrays {
  const matrices = [...iterMatrixTransforms(transforms)];
  const claimed = new Set<string>();
  for (const transform of matrices) {
    const path = namedPath(transform);
    if (path !== undefined) {
      claimed.add(checkedPath(path, transform));
    }
  }
  const arrays: MatrixArrays = new Map();
  for (const transform of matrices) {
    const kind = String(transform.type);
    const values = transform[kind];
    delete transform[kind];
    if (
      values === undefined || values === null ||
      (Array.isArray(values) && values.length === 0)
    ) {
      if (namedPath(transform) === undefined) {
        throw new Error(
          `${describe(transform)} holds no matrix and names no path to an ` +
            "array holding one",
        );
      }
      continue;
    }
    const matrix = toMatrixArray(values, transform);
    let path = namedPath(transform);
    if (path === undefined) {
      path = generatedPath(transform, claimed);
      claimed.add(path);
      transform.path = path;
    }
    const existing = arrays.get(path);
    if (existing !== undefined && !sameMatrix(existing, matrix)) {
      throw new Error(
        `Two matrix transformations name the array path '${path}' but hold ` +
          "different matrices",
      );
    }
    arrays.set(path, matrix);
  }
  return arrays;
}

/**
 * Write each matrix as a single-chunk, uncompressed float64 Zarr v3 array.
 *
 * `arrays` is keyed by path relative to `root`, the multiscales group; missing
 * parent groups are created and existing ones left as they are. An existing
 * array at a path is replaced.
 */
export async function writeMatrixArrays<Store extends zarr.Mutable>(
  root: zarr.Location<Store>,
  arrays: MatrixArrays,
): Promise<void> {
  for (const [path, matrix] of arrays) {
    const segments = path.split("/");
    for (let end = 1; end < segments.length; end++) {
      const group = root.resolve(segments.slice(0, end).join("/"));
      if (
        await group.store.get(group.resolve("zarr.json").path) === undefined
      ) {
        await zarr.create(group, { attributes: {} });
      }
    }
    const array = await zarr.create(root.resolve(path), {
      shape: matrix.shape,
      dtype: "float64",
      chunkShape: matrix.shape,
      fillValue: 0,
      codecs: bytesOnlyCodecs(),
    });
    await zarr.set(array, null, {
      data: matrix.data,
      shape: matrix.shape,
      stride: [matrix.shape[1], 1],
    });
  }
}

function matrixValues(
  chunk: { data: unknown; shape: number[]; stride: number[] },
  transform: JsonObject,
): number[][] {
  const { data, shape, stride } = chunk;
  if (shape.length !== 2 || shape.includes(0)) {
    throw new Error(
      `${describe(transform)} must hold a non-empty 2D matrix; got shape ` +
        `${shape.join("x")}`,
    );
  }
  if (!ArrayBuffer.isView(data) || data instanceof DataView) {
    throw new Error(`${describe(transform)} does not hold numeric values`);
  }
  const values = data as unknown as ArrayLike<number | bigint>;
  const rows: number[][] = [];
  for (let i = 0; i < shape[0]; i++) {
    const row: number[] = [];
    for (let j = 0; j < shape[1]; j++) {
      row.push(Number(values[i * stride[0] + j * stride[1]]));
    }
    rows.push(row);
  }
  return rows;
}

/**
 * `transforms` with each array-stored rotation and affine matrix inline.
 *
 * Returns a deep copy of the serialized `coordinateTransformations` list in
 * which every rotation and affine that names a `path` and holds no inline
 * matrix carries the array's values beside that `path`. `path` resolves
 * against `root`, the multiscales group, and is refused when it could reach
 * outside that group.
 */
export async function resolveMatrixTransforms(
  transforms: Array<Record<string, unknown>>,
  root: zarr.Location<zarr.Readable>,
): Promise<Array<Record<string, unknown>>> {
  const resolved = structuredClone(transforms);
  for (const transform of iterMatrixTransforms(resolved)) {
    const kind = String(transform.type);
    const path = transform.path;
    if (
      (transform[kind] !== undefined && transform[kind] !== null) ||
      typeof path !== "string"
    ) {
      continue;
    }
    // Read only inside the group: a crafted path could otherwise reach past
    // the store, or past `root`, which an absolute path discards.
    checkedPath(path, transform);
    let chunk;
    try {
      const array = await zarr.open(root.resolve(path), { kind: "array" });
      chunk = await zarr.get(array);
    } catch (error) {
      throw new Error(
        `${describe(transform)} stores its matrix in the array at '${path}', ` +
          `which could not be read: ${
            error instanceof Error ? error.message : String(error)
          }`,
      );
    }
    transform[kind] = matrixValues(chunk, transform);
  }
  return resolved;
}
