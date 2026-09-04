// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Zarr v3 consolidated metadata for the OME-Zarr writers.
 *
 * Consolidation inlines every child node document into the root `zarr.json`,
 * so a reader that understands the block resolves the whole hierarchy from one
 * document instead of one request per node. zarr-python understands it, and so
 * therefore does the Python side of this package, which consolidates every
 * store it writes -- writing it here is what keeps the two writers' output
 * equivalent.
 *
 * zarrita has no consolidation API: it exports the `withConsolidated` /
 * `tryWithConsolidated` readers and nothing that writes the block, and those
 * readers understand only the Zarr v2 `.zmetadata` sidecar, so `fromOmeZarr`
 * does not yet benefit from what is written here. The block is instead
 * assembled from the documents the writer just wrote, mirroring
 * `_zarrista_utils.consolidate_metadata` on the Python side, which in turn
 * matches what `zarr.consolidate_metadata` produces.
 *
 * That v2 sidecar has no counterpart here: the TypeScript writer emits Zarr v3
 * documents only.
 */

/** A store key, which zarrita always spells with a leading slash. */
type AbsolutePath = `/${string}`;

/**
 * The store surface consolidation needs: read a document, write a document.
 *
 * Deliberately narrower than zarrita's `Mutable` so both a `MemoryStore`
 * (a plain `Map`, synchronous) and a `FileSystemStore` (asynchronous) satisfy
 * it, and so this module pulls in no store implementation -- it has to stay
 * importable from the browser build.
 */
export interface ConsolidatableStore {
  get(
    key: AbsolutePath,
  ): Promise<Uint8Array | undefined> | Uint8Array | undefined;
  set(key: AbsolutePath, value: Uint8Array): unknown;
}

const ROOT_DOCUMENT: AbsolutePath = "/zarr.json";

type JsonObject = Record<string, unknown>;

/** The `consolidated_metadata` block wrapping `metadata`. */
function inlineBlock(metadata: JsonObject): JsonObject {
  return { kind: "inline", must_understand: false, metadata };
}

async function readDocument(
  store: ConsolidatableStore,
  key: AbsolutePath,
): Promise<JsonObject | undefined> {
  const bytes = await store.get(key);
  if (bytes === undefined) {
    return undefined;
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as JsonObject;
}

async function writeDocument(
  store: ConsolidatableStore,
  key: AbsolutePath,
  document: JsonObject,
): Promise<void> {
  // Two-space indent is what zarrita's own `json_encode_object` writes, so a
  // document rewritten here is formatted like the ones around it.
  await store.set(
    key,
    new TextEncoder().encode(JSON.stringify(document, null, 2)),
  );
}

/**
 * Normalize a child document into the entry zarr-python would consolidate.
 *
 * zarr-python's consolidation re-serializes every entry through its metadata
 * model, which always materializes these optional fields, and marks each
 * non-root group entry as (vacuously) consolidated. Matching it keeps a block
 * written here indistinguishable from one written by zarr-python or by the
 * Python package.
 */
function consolidatedEntry(document: JsonObject): JsonObject {
  const entry: JsonObject = { ...document };
  entry.attributes ??= {};
  if (entry.node_type === "group") {
    entry.consolidated_metadata = inlineBlock({});
  } else {
    entry.storage_transformers ??= [];
  }
  return entry;
}

/**
 * The node paths a dataset list implies, root-relative and without a leading
 * slash -- the keys a consolidated block is indexed by.
 *
 * A dataset path may be nested (`"scale0/image"`), in which case each ancestor
 * group is a node in its own right and gets its own entry, the way
 * zarr-python's consolidation walks the hierarchy.
 */
export function datasetNodePaths(datasetPaths: string[]): string[] {
  const nodePaths = new Set<string>();
  for (const datasetPath of datasetPaths) {
    const segments = datasetPath.split("/").filter((segment) => segment !== "");
    for (let end = 1; end <= segments.length; end++) {
      nodePaths.add(segments.slice(0, end).join("/"));
    }
  }
  return [...nodePaths].sort();
}

/**
 * Inline the metadata of `nodePaths` into the root `zarr.json` of `store`.
 *
 * A node without a document of its own is skipped rather than invented, so an
 * ancestor group the writer never materialized does not turn into an entry
 * that points at nothing.
 *
 * @param store - Store holding the already-written documents
 * @param nodePaths - Root-relative node paths, e.g. from {@link datasetNodePaths}
 */
export async function consolidateMetadata(
  store: ConsolidatableStore,
  nodePaths: string[],
): Promise<void> {
  const metadata: JsonObject = {};
  for (const nodePath of nodePaths) {
    const document = await readDocument(store, `/${nodePath}/zarr.json`);
    if (document === undefined) {
      continue;
    }
    metadata[nodePath] = consolidatedEntry(document);
  }

  const root = await readDocument(store, ROOT_DOCUMENT);
  if (root === undefined) {
    throw new Error(
      "Cannot consolidate metadata: the store has no root zarr.json.",
    );
  }
  root.consolidated_metadata = inlineBlock(metadata);
  await writeDocument(store, ROOT_DOCUMENT, root);
}

/**
 * Whether the store's root `zarr.json` carries a consolidated metadata block.
 *
 * The Zarr v3 counterpart of the Python `has_consolidated_metadata`, used to
 * decide whether an in-place metadata rewrite has consolidation to refresh.
 */
export async function hasConsolidatedMetadata(
  store: ConsolidatableStore,
): Promise<boolean> {
  const root = await readDocument(store, ROOT_DOCUMENT);
  return root !== undefined && root.consolidated_metadata !== undefined;
}
