// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Shared internal utilities for OME-Zarr writing (regular Zarr stores and
 * RFC-9 .ozx). Used by the Node and browser writers — and the in-place
 * `upgradeOmeZarr` metadata rewrite — to avoid duplicating the root-attribute
 * and axis-processing logic.
 */

import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import type { Axis, MetadataInterface } from "../types/zarr_metadata.ts";
import { type MemoryStore, memoryStoreToZip } from "./rfc9_zip.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import {
  consolidateMetadata,
  datasetNodePaths,
} from "../utils/consolidate_metadata.ts";
import {
  buildV06MultiscalesEntry,
  legacyTopLevelTransforms,
} from "../utils/v06_metadata.ts";
import {
  externalizeMatrixTransforms,
  type MatrixArrays,
  writeMatrixArrays,
} from "../utils/matrix_transform_arrays.ts";
import {
  NgffVersion,
  V06_ONDISK_VERSION,
} from "../types/supported_versions.ts";
import { detectVersion } from "../utils/parse_metadata.ts";
import {
  SpecRule,
  validateAxisCount,
  validateAxisNamesUnique,
  validateAxisOrder,
  validateAxisType,
  validateSpatialAxisOrder,
  ValidationError,
} from "../utils/structural_validation.ts";
import { pyRepr, pyReprOptional } from "../utils/py_format.ts";
import {
  type ChunksPerShard,
  ensureRangeReads,
  outerChunkShape,
} from "../utils/sharding.ts";

/**
 * Every axis list `metadata` will serialize at `version`, paired with its
 * location.
 *
 * Only the v0.6 family writes `coordinateSystems`, and
 * `coordinate_systems.schema` applies `axes.schema` to each one, so every
 * system is returned there. A v0.4/v0.5 target writes a single flat `axes` and
 * drops the systems, so that is the only list to gate: refusing a system the
 * downgrade discards would reject a document the writer can emit, and name a
 * node absent from it. Python reaches the same set by gating after
 * `Metadata.to_version`.
 */
function axisViews(
  metadata: MetadataInterface,
  version: string,
): Array<{ location: string; axes: Axis[] }> {
  const systems = metadata.coordinateSystems ?? [];
  const serializesSystems = version === "0.6" ||
    version === NgffVersion.V09dev1;
  if (serializesSystems && systems.length > 0) {
    // `buildV06MultiscalesEntry` serializes the first system from
    // `metadata.axes` and the later ones verbatim, and `MetadataInterface`
    // does not tie `axes` to `coordinateSystems[0].axes`. Gate what is
    // written, not what is declared.
    return systems.map((system, index) => ({
      location: `multiscales[0].coordinateSystems[${index}].axes`,
      axes: index === 0 ? metadata.axes : system.axes,
    }));
  }
  return [{ location: "multiscales[0].axes", axes: metadata.axes }];
}

/**
 * Refuse to serialize an axis model the target `version` cannot express.
 *
 * Reuses the axis rules of the structural pass, but this is a second
 * dispatcher, not the same pass as {@link validateStructural}, and it differs
 * from it two ways:
 *
 * 1. It runs only the five axis rules, not the full image cascade.
 * 2. Where the target version serializes coordinate systems, it checks *every*
 *    one of them (see {@link axisViews}), while the structural pass reduces the
 *    metadata to the intrinsic system's axes.
 */
function gateAxisModel(
  metadata: MetadataInterface,
  version: string,
): void {
  gateAxisViews(axisViews(metadata, version), version);
}

/** Apply the axis rules of `version` to each `{ location, axes }` view. */
export function gateAxisViews(
  views: Array<{ location: string; axes: Axis[] }>,
  version: string,
): void {
  const rules = [
    validateAxisCount,
    validateAxisType,
    validateAxisOrder,
    validateSpatialAxisOrder,
    validateAxisNamesUnique,
  ];
  for (const view of views) {
    for (const rule of rules) {
      try {
        rule({ axes: view.axes }, version);
      } catch (error) {
        if (!(error instanceof ValidationError)) {
          throw error;
        }
        const rendered = view.axes
          .map((ax) => `${pyRepr(ax.name)}(type=${pyReprOptional(ax.type)})`)
          .join(", ");
        if (error.rule === SpecRule.AxisNamesUnique) {
          // Required at every version, 0.9.dev1 included.
          throw new Error(
            `Cannot write OME-Zarr version="${version}": ${error.detail} ` +
              `Axes at ${view.location}: [${rendered}].`,
          );
        }
        throw new Error(
          `Cannot write OME-Zarr version="${version}": this axis model violates ` +
            `that version's [${error.rule}] rule. ${error.detail} ` +
            `Axes at ${view.location}: [${rendered}]. ` +
            `Pass version="${NgffVersion.V09dev1}" to write it: 0.9.dev1 is the ` +
            `only OME-Zarr version that adopts RFC-3 (arbitrary axis count, ` +
            `names, types and ordering).`,
        );
      }
    }
  }
}

/**
 * Refuse a scale or translation whose vector does not span the axes it
 * applies to: `transform`'s own vectors, then those of its sequence members,
 * which span what the sequence spans. `systems` maps a coordinate system
 * name to its axis count, so a transform naming one spans that count;
 * otherwise it spans `inherited`. Mirrors the Python `_gate_spans`.
 */
export function gateSpans(
  transform: unknown,
  where: string,
  systems: Map<string, number>,
  inherited: Set<number>,
): void {
  const record = transform as Record<string, unknown>;
  const named = new Set<number>();
  for (const side of ["input", "output"]) {
    const reference = record[side] as { name?: unknown } | undefined;
    const count = typeof reference?.name === "string"
      ? systems.get(reference.name)
      : undefined;
    if (count !== undefined) {
      named.add(count);
    }
  }
  const spans = named.size > 0 ? named : inherited;
  for (const kind of ["scale", "translation"]) {
    const vector = record[kind];
    if (Array.isArray(vector) && !spans.has(vector.length)) {
      const axes = [...spans].sort((a, b) => a - b).join(" or ");
      throw new Error(
        `${where} (${record.type}) gives ${vector.length} ${kind} values ` +
          `for the ${axes} axes it applies to; a transform that does not ` +
          "span its axes cannot be applied by a reader.",
      );
    }
  }
  const members = record.transformations;
  if (Array.isArray(members)) {
    members.forEach((member, position) =>
      gateSpans(member, `${where}.transformations[${position}]`, systems, spans)
    );
  }
}

/**
 * Refuse a scale or translation whose vector does not span the axes it
 * applies to, at the multiscales level and the dataset level alike. A
 * transform naming a coordinate system spans the axes the target `version`
 * writes for it (see {@link axisViews}: the first system is written from
 * `metadata.axes`, and a 0.4/0.5 target writes no system at all); any other
 * spans the intrinsic axes. Mirrors the Python `_gate_transform_arity`.
 */
function gateTransformArity(
  metadata: MetadataInterface,
  version: string,
): void {
  const systems = new Map<string, number>();
  if (version === "0.6" || version === NgffVersion.V09dev1) {
    (metadata.coordinateSystems ?? []).forEach((system, index) => {
      systems.set(
        system.name,
        index === 0 ? metadata.axes.length : system.axes.length,
      );
    });
  }
  const intrinsic = metadata.axes.length;
  if (intrinsic === 0) {
    return;
  }
  const levels: Array<[string, unknown[]]> = [
    ["the multiscales", metadata.coordinateTransformations ?? []],
  ];
  for (const dataset of metadata.datasets) {
    levels.push([
      `dataset '${dataset.path}'`,
      dataset.coordinateTransformations ?? [],
    ]);
  }
  for (const [where, transforms] of levels) {
    transforms.forEach((transform, index) =>
      gateSpans(
        transform,
        `${where} coordinateTransformations[${index}]`,
        systems,
        new Set([intrinsic]),
      )
    );
  }
}

/**
 * Process axes for serialization.
 * Anatomical orientation (RFC 4) is included whenever an axis carries it;
 * axes without orientation simply omit the field.
 */
export function processAxes(
  axes: Axis[],
): Record<string, unknown>[] {
  return axes.map((axis) => {
    const result: Record<string, unknown> = {
      name: axis.name,
      type: axis.type,
    };

    // Include unit if present
    if (axis.unit !== undefined) {
      result.unit = axis.unit;
    }

    // Include the discrete flag whenever present (RFC-5 vector-field axes)
    if (axis.discrete !== undefined) {
      result.discrete = axis.discrete;
    }

    // Include orientation whenever it is present
    if (axis.orientation) {
      result.orientation = {
        type: axis.orientation.type,
        value: axis.orientation.value,
      };
    }

    return result;
  });
}

/** The root-group attributes and the matrix arrays they reference. */
export interface RootDocument {
  attributes: Record<string, unknown>;
  /**
   * The rotation and affine matrices the attributes name by `path`, to write
   * as arrays beside them; see {@link externalizeMatrixTransforms}.
   */
  matrixArrays: MatrixArrays;
}

/**
 * Build the root-group attributes for an OME-Zarr store at a given spec version
 * from the version-agnostic in-memory metadata. This is the single source of
 * truth for the on-disk root shape shared by the Node and browser writers
 * ({@link toOmeZarr}) and the in-place metadata rewrite in `upgradeOmeZarr`:
 *
 * - **0.6 (RFC 5):** coordinate systems + per-dataset `sequence` transforms,
 *   wrapped under the `ome` namespace and tagged {@link V06_ONDISK_VERSION}
 *   (`0.6`). Each rotation and affine names its matrix by `path`; the matrices
 *   come back in `matrixArrays` for the caller to write with
 *   {@link writeMatrixArrays}.
 * - **0.5:** axes carried directly on the multiscale entry, wrapped under `ome`.
 * - **0.4:** axes carried directly on the multiscale entry at the root (no
 *   `ome` wrapper).
 *
 * Richer v0.6-only top-level transforms are reduced to the simple
 * scale/translation subset for 0.4/0.5 via {@link legacyTopLevelTransforms}.
 */
export function buildRootDocument(
  metadata: MetadataInterface,
  version: "0.4" | "0.5" | "0.6" | "0.9.dev1",
): RootDocument {
  gateAxisModel(metadata, version);
  gateTransformArity(metadata, version);

  // Process axes (orientation included when present).
  const processedAxes = processAxes(metadata.axes);

  if (version === "0.9.dev1") {
    // "0.9.dev1" is already the on-disk string.
    const v09Entry = buildV06MultiscalesEntry(metadata, processedAxes);
    const matrixArrays = externalizeMatrixTransforms(
      v09Entry.coordinateTransformations,
      metadata.datasets.map((dataset) => dataset.path),
    );
    return {
      attributes: {
        ome: {
          version: NgffVersion.V09dev1,
          multiscales: [v09Entry],
          ...(metadata.omero && { omero: metadata.omero }),
        },
      },
      matrixArrays,
    };
  }

  if (version === "0.6") {
    const v06Entry = buildV06MultiscalesEntry(metadata, processedAxes);
    const matrixArrays = externalizeMatrixTransforms(
      v06Entry.coordinateTransformations,
      metadata.datasets.map((dataset) => dataset.path),
    );
    return {
      attributes: {
        ome: {
          // Tag the store with V06_ONDISK_VERSION, which is `0.6` since the
          // release; the constant stays the single place the tag lives.
          version: V06_ONDISK_VERSION,
          multiscales: [v06Entry],
          ...(metadata.omero && { omero: metadata.omero }),
        },
      },
      matrixArrays,
    };
  }

  // v0.4/v0.5 only support the simple scale/translation subset of top-level
  // transformations; drop richer v0.6 transforms when downgrading.
  const legacyTransforms = legacyTopLevelTransforms(
    metadata.coordinateTransformations,
  );
  const multiscalesMetadata = {
    version,
    name: metadata.name,
    axes: processedAxes,
    datasets: metadata.datasets,
    ...(legacyTransforms && { coordinateTransformations: legacyTransforms }),
    ...(metadata.type && { type: metadata.type }),
    ...(metadata.metadata && { metadata: metadata.metadata }),
  };

  // Neither version carries a rotation or an affine, so no matrix arrays.
  const attributes = version === "0.5"
    ? {
      ome: {
        version,
        multiscales: [multiscalesMetadata],
        ...(metadata.omero && { omero: metadata.omero }),
      },
    }
    : {
      multiscales: [multiscalesMetadata],
      ...(metadata.omero && { omero: metadata.omero }),
    };
  return { attributes, matrixArrays: new Map() };
}

/**
 * The OME-Zarr versions an RFC-9 `.ozx` archive can hold. RFC-9 is defined on
 * Zarr v3 -- the archive leads with the root `zarr.json` -- so every version
 * stored in Zarr v3 qualifies: 0.5, 0.6, and the opt-in 0.9.dev1. OME-Zarr
 * 0.4 lives in Zarr v2 and cannot be zipped. Mirrors the Python port, where
 * `.ozx` output requires `_zarr_format_for_version(version) == 3`.
 */
export type OzxVersion = "0.5" | "0.6" | "0.9.dev1";

/** The version an `.ozx` archive is written at when none is requested. */
export const DEFAULT_OZX_VERSION: OzxVersion = "0.5";

/**
 * Chunks per shard, along every axis, when an `.ozx` write requests none --
 * the Python writer's default, so the two lay out the same archive.
 */
export const DEFAULT_OZX_CHUNKS_PER_SHARD = 2;

/**
 * Refuse a version an RFC-9 archive cannot hold; see {@link OzxVersion}.
 * Accepts `undefined` so a caller can pass an option through as given.
 */
export function gateOzxVersion(version: string | undefined): OzxVersion {
  if (version === undefined) {
    return DEFAULT_OZX_VERSION;
  }
  if (version === "0.5" || version === "0.6" || version === "0.9.dev1") {
    return version;
  }
  throw new Error(
    "RFC-9 (.ozx) requires OME-Zarr version 0.5 or later (Zarr v3). " +
      `Got version "${version}". ` +
      "For .ozx files, omit the version option or set it to '0.5', '0.6' " +
      "or '0.9.dev1'.",
  );
}

/** Options for packing a store into an `.ozx` archive with `storeToZip`. */
export interface StoreToZipOptions {
  /**
   * OME-Zarr version to record in the archive's ZIP comment. Omitted, the
   * version the store's root document declares, as the readers detect it,
   * or `0.5` when it declares none.
   */
  version?: OzxVersion | undefined;
}

/**
 * The contents of `store` packed into an RFC-9 `.ozx` archive, as
 * {@link memoryStoreToZip} lays them out: the root `zarr.json` first, the
 * other `zarr.json` documents breadth first, nothing compressed again. A
 * store staged with `toOmeZarr` -- a displacement field written below its
 * `path`, then the scene or transformation referencing it with `overwrite:
 * false` -- is packed whole, the way the Python `write_store_to_zip` packs
 * a directory. Throws on a store with no root `zarr.json` or a Zarr v2 one:
 * RFC-9 holds a Zarr v3 store.
 */
export function storeToZipData(
  store: MemoryStore,
  options: StoreToZipOptions = {},
): Uint8Array {
  const rootBytes = store.get("/zarr.json") ?? store.get("zarr.json");
  if (rootBytes === undefined) {
    throw new Error(
      "The store has no root zarr.json, so it is not a Zarr v3 store an " +
        ".ozx archive (RFC-9) can hold.",
    );
  }
  const root = JSON.parse(new TextDecoder().decode(rootBytes)) as {
    zarr_format?: unknown;
    attributes?: Record<string, unknown>;
  };
  if (root.zarr_format !== 3) {
    throw new Error(
      "An .ozx archive (RFC-9) holds a Zarr v3 store; the root zarr.json " +
        `declares zarr_format ${JSON.stringify(root.zarr_format)}.`,
    );
  }
  // The version as the readers detect it: `ome.version`, or a bare
  // multiscales entry's own, which is how a 0.4 image written in a Zarr v3
  // container declares itself. A plain Zarr group declares none; a version
  // the readers do not support throws.
  const attributes = root.attributes ?? {};
  const ome = attributes.ome as { version?: unknown } | undefined;
  const multiscales = attributes.multiscales;
  const declaresVersion = (ome?.version !== undefined) ||
    (Array.isArray(multiscales) && multiscales.length > 0);
  const declared = options.version === undefined && declaresVersion
    ? detectVersion(attributes)
    : undefined;
  const version = gateOzxVersion(options.version ?? declared);
  return memoryStoreToZip(store, { version });
}

/**
 * Get chunks from an NgffImage, falling back to default chunk size if not specified.
 *
 * @param image - NgffImage to get chunks from
 * @returns Array of chunk sizes for each dimension
 */
function getChunksFromImage(image: NgffImage): number[] {
  if (image.data.chunks && image.data.chunks.length > 0) {
    return image.data.chunks;
  }
  return image.data.shape.map((s: number) => Math.min(s, 1024));
}

/** Reports cumulative writes: `completedChunks` of `totalChunks`. */
export type ProgressCallback = (
  completedChunks: number,
  totalChunks: number,
) => void;

/**
 * Writes one array of a multiscales group, below `group` at `path`. The Node
 * and browser writers each supply their own, bound to their codec and
 * sharding options.
 */
export type MultiscalesArrayWriter = (
  group: zarr.Group<MemoryStore>,
  image: NgffImage,
  path: string,
  onProgress?: ProgressCallback | null,
) => Promise<void>;

/**
 * Makes a writer's {@link MultiscalesArrayWriter} for a codec pipeline and a
 * sharding layout, so code shared by the Node and browser writers, such as
 * the scene writer, writes arrays the way the calling writer does.
 */
export type ArrayWriterFactory = (options: {
  codecs?: ZarrCodec[] | undefined;
  chunksPerShard?: ChunksPerShard | undefined;
}) => MultiscalesArrayWriter;

/** How many writes one level takes: chunks, or whole shards when sharded. */
function countImageWrites(
  image: NgffImage,
  chunksPerShard: ChunksPerShard | undefined,
): number {
  const shape = image.data.shape;
  const chunks = outerChunkShape(
    shape,
    image.dims,
    getChunksFromImage(image),
    chunksPerShard,
  );
  let writes = 1;
  for (let d = 0; d < shape.length; d++) {
    writes *= Math.ceil(shape[d] / chunks[d]);
  }
  return writes;
}

/**
 * How many writes {@link writeMultiscalesGroup} makes for `multiscales`, the
 * total its progress counts toward.
 */
export function countMultiscalesWrites(
  multiscales: NgffMultiscales,
  chunksPerShard?: ChunksPerShard,
): number {
  return multiscales.images.reduce(
    (total, image) => total + countImageWrites(image, chunksPerShard),
    0,
  );
}

/**
 * Write `multiscales` as the OME-Zarr group at `location`: the root document
 * at `version`, the matrix arrays it names, and each level through
 * `writeArray`. `location` may be the root of a store or a group below it, as
 * the images of a scene are.
 *
 * Returns the node paths below `location` that a consolidated metadata block
 * lists, the datasets and the matrix arrays with their ancestor groups;
 * consolidation is left to the caller, which knows where the store's root
 * is. `onProgress` counts this group's writes ({@link countMultiscalesWrites}).
 */
export async function writeMultiscalesGroup(
  location: zarr.Location<zarr.Mutable>,
  multiscales: NgffMultiscales,
  writeArray: MultiscalesArrayWriter,
  version: "0.4" | "0.5" | "0.6" | "0.9.dev1",
  onProgress?: ProgressCallback | null,
  chunksPerShard?: ChunksPerShard,
): Promise<string[]> {
  // The same version-specific root document whatever the container: a zipped
  // store differs from a directory store only in how it is packaged.
  const { attributes, matrixArrays } = buildRootDocument(
    multiscales.metadata,
    version,
  );

  const group = await zarr.create(location, { attributes });
  await writeMatrixArrays(location, matrixArrays);

  const totalChunks = onProgress
    ? countMultiscalesWrites(multiscales, chunksPerShard)
    : 0;
  let completedChunks = 0;
  for (let i = 0; i < multiscales.images.length; i++) {
    const image = multiscales.images[i];
    const dataset = multiscales.metadata.datasets[i];

    if (!dataset) {
      throw new Error(`No dataset configuration found for image ${i}`);
    }

    // Report each level's writes on top of the levels before it.
    const offset = completedChunks;
    await writeArray(
      group as zarr.Group<MemoryStore>,
      image,
      dataset.path,
      onProgress
        ? (completed: number) => onProgress(offset + completed, totalChunks)
        : null,
    );
    if (onProgress) {
      completedChunks += countImageWrites(image, chunksPerShard);
    }
  }

  return datasetNodePaths([
    ...multiscales.metadata.datasets.map((dataset) => dataset.path),
    ...matrixArrays.keys(),
  ]);
}

/**
 * Write multiscales data to a memory store for RFC-9 export.
 * This is the shared implementation used by both Node and browser versions.
 *
 * @param store - Memory store to write to
 * @param multiscales - NgffMultiscales data to write
 * @param writeImage - Function to write individual images
 * @param onProgress - Optional progress callback for cumulative chunk
 *   progress across all scale levels
 * @param consolidate - Inline the array documents into the root `zarr.json`
 *   once they are written (default `true`)
 * @param version - OME-Zarr version to write the store at; see
 *   {@link OzxVersion} (default `0.5`)
 * @param chunksPerShard - Chunks per shard, when `writeImage` shards; the
 *   progress count is of the writes, which are then whole shards
 */
export async function writeNgffMultiscalesToMemoryStore(
  store: MemoryStore,
  multiscales: NgffMultiscales,
  writeImage: MultiscalesArrayWriter,
  onProgress?: ProgressCallback | null,
  consolidate: boolean = true,
  version: OzxVersion = DEFAULT_OZX_VERSION,
  chunksPerShard?: ChunksPerShard,
): Promise<void> {
  // A sharded array needs range reads, which the in-memory `Map` lacks.
  const root = zarr.root(
    chunksPerShard === undefined ? store : ensureRangeReads(store),
  );
  const nodePaths = await writeMultiscalesGroup(
    root,
    multiscales,
    writeImage,
    version,
    onProgress,
    chunksPerShard,
  );

  // Consolidate last: the block inlines the array documents, so they have to
  // exist first. The root key keeps its insertion position in the Map, so the
  // RFC-9 zip writer still lays the root document down as the first entry.
  if (consolidate) {
    await consolidateMetadata(store, nodePaths);
  }
}
