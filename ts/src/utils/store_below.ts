// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The part of a store below a path, as a store of its own.
 *
 * A group below a store's root is written and read as if it were a store's
 * root: `toOmeZarr(store, image, { path })` writes a displacement field below
 * the path a transformation references, `fromOmeZarr(store, { path })` reads
 * it back, and a scene's images are read below their paths. A directory path
 * is joined with the path instead; a store object, such as a `MemoryStore`
 * or a zipped `.ozx` store, is viewed through {@link storeBelow}. No Node.js
 * dependency.
 */
import * as zarr from "zarrita";

/**
 * Whether `path` is a plain path below a store's root: `/`-separated
 * segments, none empty, `.` or `..`, that a store reads as written. zarrita
 * resolves a path through the URL parser, once to reach the node and again
 * at every node below it, so a path that parser rewrites, cuts or drops
 * could name another node: a backslash (read as `/`), a `%` (decoded, or
 * refused when malformed), a `?` or `#` (cut with what follows), a control
 * character or a leading or trailing space (dropped), or a scheme or drive
 * prefix such as `file:`, `C:` or `C|` (which replaces the root). None is
 * allowed, and the Python port refuses the same, so both ports accept the
 * same documents.
 */
export function isPathBelowRoot(path: unknown): path is string {
  return (
    typeof path === "string" &&
    !/[\\%?#\p{Cc}]|^ | $/u.test(path) &&
    !/^[A-Za-z](?:[A-Za-z0-9+.-]*:|\|)/.test(path) &&
    !path.split("/").some((segment) => ["", ".", ".."].includes(segment))
  );
}

/** Throw unless `path` is a plain path below a store's root. */
export function checkPathBelowRoot(path: unknown): asserts path is string {
  if (!isPathBelowRoot(path)) {
    throw new Error(
      `The path '${String(path)}' must be a relative path below the store's ` +
        "root, with no empty, '.' or '..' segments.",
    );
  }
}

/**
 * The part of `store` below `path`, as a store of its own: every key is
 * looked up, and written when `store` is writable, under `/<path>`. Range
 * reads are passed through when `store` has them, so a sharded array below
 * `path` still opens.
 */
export function storeBelow(
  store: zarr.Mutable,
  path: string,
): zarr.AsyncMutable;
export function storeBelow(
  store: zarr.Readable,
  path: string,
): zarr.AsyncReadable;
export function storeBelow(
  store: zarr.Readable & Partial<zarr.Writable>,
  path: string,
): zarr.AsyncReadable & Partial<zarr.AsyncWritable> {
  checkPathBelowRoot(path);
  const below: zarr.AsyncReadable & Partial<zarr.AsyncWritable> = {
    get: async (key, options) => await store.get(`/${path}${key}`, options),
  };
  if (typeof store.getRange === "function") {
    const getRange = store.getRange.bind(store);
    below.getRange = async (key, range, options) =>
      await getRange(`/${path}${key}`, range, options);
  }
  if (typeof store.set === "function") {
    const set = store.set.bind(store);
    below.set = async (key, value) => {
      await set(`/${path}${key}`, value);
    };
  }
  return below;
}

/**
 * The kind of node the metadata at `location` declares, Zarr v3 or v2, or
 * `undefined` when there is none. Malformed metadata throws rather than
 * reading as no node, which `zarr.open` does not tell apart.
 */
async function nodeTypeAt(
  location: zarr.Location<zarr.Mutable>,
): Promise<unknown> {
  const get = async (key: string) =>
    await location.store.get(location.resolve(key).path);
  const v3 = await get("zarr.json");
  if (v3 !== undefined) {
    return (
      JSON.parse(new TextDecoder().decode(v3)) as {
        node_type?: unknown;
      }
    ).node_type;
  }
  if ((await get(".zarray")) !== undefined) return "array";
  if ((await get(".zgroup")) !== undefined) return "group";
  return undefined;
}

/**
 * Create each group above `path` that the store below `root` lacks, empty,
 * so the hierarchy stays navigable from the root; a group it holds is left
 * as it is, and an array there is refused rather than replaced. Returns
 * their paths, the shallowest first.
 */
export async function ensureAncestorGroups(
  root: zarr.Location<zarr.Mutable>,
  path: string,
): Promise<string[]> {
  const segments = path.split("/");
  const ancestors: string[] = [];
  for (let end = 1; end < segments.length; end++) {
    const ancestor = segments.slice(0, end).join("/");
    ancestors.push(ancestor);
    const location = root.resolve(ancestor);
    const nodeType = await nodeTypeAt(location);
    if (nodeType === undefined) {
      await zarr.create(location);
    } else if (nodeType !== "group") {
      throw new Error(
        `'${ancestor}' holds a Zarr ${
          nodeType === "array" ? "array" : `node of type ${String(nodeType)}`
        }, not a group, so nothing is written below it at '${path}'.`,
      );
    }
  }
  return ancestors;
}

/**
 * `store` below `path`, to write a group into as if it were a store's root:
 * {@link storeBelow}, with the groups above `path` created first.
 */
export async function writableStoreBelow(
  store: zarr.Mutable,
  path: string,
): Promise<zarr.AsyncMutable> {
  checkPathBelowRoot(path);
  await ensureAncestorGroups(zarr.root(store), path);
  return storeBelow(store, path);
}
