// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Scenes: images that share a spatial relationship (OME-Zarr 0.6).
 *
 * A scene is the Zarr group above a set of multiscale images. Its `ome.scene`
 * metadata declares the coordinate transformations between the images'
 * coordinate systems and, optionally, coordinate systems of its own, such as
 * a common world system the images map into. Mirrors the Python port's
 * `ngff_zarr.scene`. The layout is specified at
 * https://ngff.openmicroscopy.org/0.6/#scene-md.
 *
 * This module has no Node.js dependency: the Node and browser readers both
 * read scenes through it, and the Node and browser writers both write them
 * through it, each with its own array writer, to a directory (Node), a
 * `MemoryStore`, or an RFC-9 `.ozx` archive.
 */
import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import { NgffScene } from "../types/scene.ts";
import {
  isV06Version,
  NgffVersion,
  V06_ONDISK_VERSION,
} from "../types/supported_versions.ts";
import {
  type Axis,
  type CoordinateSystem,
  type CoordinateSystemIdentifier,
  INTRINSIC_COORDINATE_SYSTEM_NAME,
  type V06Transform,
} from "../types/zarr_metadata.ts";
import { detectVersion } from "../utils/parse_metadata.ts";
import {
  parseV06Transforms,
  serializeV06Transform,
  validateV06Transform,
} from "../utils/v06_metadata.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import {
  type ConsolidatableStore,
  consolidateMetadata,
} from "../utils/consolidate_metadata.ts";
import { type ChunksPerShard, ensureRangeReads } from "../utils/sharding.ts";
import type { ChunkCache } from "../utils/worker_pool.ts";
import { memoryStoreToZip } from "./rfc9_zip.ts";
import {
  type ArrayWriterFactory,
  countMultiscalesWrites,
  DEFAULT_OZX_CHUNKS_PER_SHARD,
  gateAxisViews,
  gateOzxVersion,
  gateSpans,
  type OzxVersion,
  type ProgressCallback,
  writeMultiscalesGroup,
} from "./to_ngff_zarr_ozx_common.ts";

/** The versions whose metadata model defines scenes. */
export const SCENE_VERSIONS: readonly string[] = [
  NgffVersion.V06,
  NgffVersion.V09dev1,
];

/** The scene subset of `FromOmeZarrOptions`, forwarded by `fromOmeZarr`. */
export interface ReadSceneOptions {
  /**
   * Check that every transformation end resolves and that the coordinate
   * systems and images form one connected graph, and validate each image.
   */
  validate?: boolean;
  /** OME-Zarr version, if known. */
  version?: "0.4" | "0.5" | "0.6" | "0.9.dev1";
  /** Passed to the image reader for every image. */
  cache?: ChunkCache;
}

/** The scene subset of `ToOmeZarrOptions`, forwarded by `toOmeZarr`. */
export interface WriteSceneOptions {
  /** OME-Zarr specification version, 0.6 (the default) or later. */
  version?: "0.4" | "0.5" | "0.6" | "0.9.dev1";
  /**
   * With the default `true`, the root group's attributes are the scene's
   * alone. With `false`, the attributes an existing root group carries are
   * kept beside the scene metadata. The images are written over whatever
   * their paths hold either way. A transformation that references an array
   * or a field group by `path`, such as a `displacements` field, needs that
   * node written first, below the scene's store, and the scene written with
   * `overwrite: false`.
   */
  overwrite?: boolean;
  /**
   * Write consolidated metadata for the whole store at the root once the
   * images are written (default `true`).
   */
  consolidateMetadata?: boolean;
  /** How every image's arrays are sharded. */
  chunksPerShard?: ChunksPerShard;
  /** The codec pipeline of every image's arrays. */
  codecs?: ZarrCodec[];
  /**
   * Reports the writes of all the images together, chunks or whole shards,
   * as they complete.
   */
  onProgress?: ProgressCallback;
}

/** Reads the image below a scene: the port's `fromOmeZarr`. */
export type ImageReader = (
  store: string | zarr.Readable,
  options: { validate: boolean; cache?: ChunkCache },
) => Promise<NgffMultiscales>;

/** Whether `value` is a plain JSON object. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Reject an image path that is not a plain path below the scene group.
 *
 * The path is joined to the store, so a backslash counts as a separator
 * (Windows does) and a percent-encoded part is read as a URL store does.
 */
export function checkImagePath(path: unknown): asserts path is string {
  const decode = (part: string): string => {
    try {
      return decodeURIComponent(part);
    } catch {
      return part;
    }
  };
  if (
    typeof path !== "string" ||
    path.split(/[\\/]/).some((part) => ["", ".", ".."].includes(decode(part)))
  ) {
    throw new Error(
      `Image path '${
        String(path)
      }' must be a relative path below the scene group.`,
    );
  }
}

/** The coordinate systems the image carries once written at 0.6. */
function imageSystems(multiscales: NgffMultiscales): CoordinateSystem[] {
  const systems = multiscales.metadata.coordinateSystems;
  // The 0.6 writer declares the intrinsic system on an image that carries
  // no coordinate systems of its own.
  return systems !== undefined && systems.length > 0 ? systems : [{
    name: INTRINSIC_COORDINATE_SYSTEM_NAME,
    axes: multiscales.metadata.axes,
  }];
}

/** The paths of the arrays and field groups the transformations reference. */
export function fieldPaths(transforms: V06Transform[]): Set<string> {
  const paths = new Set<string>();
  for (const transform of transforms) {
    const path = (transform as { path?: unknown }).path;
    if (typeof path === "string") {
      paths.add(path);
    }
    const nested: V06Transform[] = [];
    if ("transformations" in transform) {
      for (const member of transform.transformations) {
        nested.push(
          "transformation" in member ? member.transformation : member,
        );
      }
    }
    if (transform.type === "bijection") {
      nested.push(transform.forward, transform.inverse);
    }
    for (const path of fieldPaths(nested)) {
      paths.add(path);
    }
  }
  return paths;
}

type Node = ["scene" | "image", string];

/**
 * Throw unless the scene is one OME-Zarr `version` can express.
 *
 * Every image path stays below the scene group; the scene's coordinate
 * systems carry unique names and an axis model the version allows; every
 * transformation names its ends, they resolve, to a system the scene
 * declares or to one of the image at its path, and the transformation holds
 * for the two systems it joins; and the coordinate systems and images form
 * one connected graph, as the spec requires. An image's own systems are
 * connected through its multiscales already, so each image counts as one
 * node.
 */
export function checkScene(scene: NgffScene, version: string = "0.6"): void {
  for (const path of Object.keys(scene.images)) {
    checkImagePath(path);
  }
  if (scene.coordinateTransformations.length === 0) {
    throw new Error("A scene declares at least one coordinate transformation.");
  }

  const local = new Map<string, CoordinateSystem>();
  for (const system of scene.coordinateSystems ?? []) {
    if (system.name === "" || local.has(system.name)) {
      throw new Error(
        "Scene coordinateSystems names must be non-empty and unique; got " +
          `${JSON.stringify(scene.coordinateSystems!.map((s) => s.name))}.`,
      );
    }
    local.set(system.name, system);
  }
  gateAxisViews(
    (scene.coordinateSystems ?? []).map((system, index) => ({
      location: `scene.coordinateSystems[${index}].axes`,
      axes: system.axes,
    })),
    version,
  );

  const key = ([kind, name]: Node): string => `${kind}\0${name}`;
  const parent = new Map<string, string>();
  const labels = new Map<string, string>();
  for (const name of local.keys()) {
    parent.set(key(["scene", name]), key(["scene", name]));
    labels.set(key(["scene", name]), `coordinate system '${name}'`);
  }
  for (const path of Object.keys(scene.images)) {
    parent.set(key(["image", path]), key(["image", path]));
    labels.set(key(["image", path]), `image '${path}'`);
  }
  const find = (node: string): string => {
    let current = node;
    while (parent.get(current) !== current) {
      const up = parent.get(current)!;
      parent.set(current, parent.get(up)!);
      current = parent.get(current)!;
    }
    return current;
  };

  scene.coordinateTransformations.forEach((transform, index) => {
    const ends: string[] = [];
    const systems: CoordinateSystem[] = [];
    for (const side of ["input", "output"] as const) {
      const where = `coordinateTransformations[${index}].${side}`;
      const reference = transform[side];
      if (reference === undefined || reference.name === undefined) {
        throw new Error(
          `${where} must name a coordinate system: a ` +
            "CoordinateSystemIdentifier with a name, and the path of the " +
            "image that declares it unless the scene does.",
        );
      }
      if (reference.path === undefined) {
        const system = local.get(reference.name);
        if (system === undefined) {
          throw new Error(
            `${where} names coordinate system '${reference.name}', which ` +
              "the scene does not declare. Declare it in coordinateSystems, " +
              "or give the reference the path of the image that declares it.",
          );
        }
        ends.push(key(["scene", reference.name]));
        systems.push(system);
        continue;
      }
      const image = Object.hasOwn(scene.images, reference.path)
        ? scene.images[reference.path]
        : undefined;
      if (image === undefined) {
        throw new Error(
          `${where} references image '${reference.path}', which the scene ` +
            `does not contain; its images are ${
              JSON.stringify(Object.keys(scene.images).sort())
            }.`,
        );
      }
      const declared = imageSystems(image);
      const system = declared.find((s) => s.name === reference.name);
      if (system === undefined) {
        throw new Error(
          `${where} names coordinate system '${reference.name}' of image ` +
            `'${reference.path}', which declares ${
              JSON.stringify(declared.map((s) => s.name).sort())
            }.`,
        );
      }
      ends.push(key(["image", reference.path]));
      systems.push(system);
    }

    const where = `coordinateTransformations[${index}]`;
    gateSpans(
      transform,
      where,
      new Map(),
      new Set(systems.map((s) => s.axes.length)),
    );
    // The reader's own checks, against the two systems the transformation
    // joins. They resolve a reference by name, so the probe names its ends
    // apart when both refer to systems that share a name.
    let names = [transform.input!.name!, transform.output!.name!];
    if (names[0] === names[1] && systems[0] !== systems[1]) {
      names = ["input", "output"];
    }
    const probe = {
      ...transform,
      input: { name: names[0] },
      output: { name: names[1] },
    } as V06Transform;
    try {
      validateV06Transform(probe, [
        { name: names[0], axes: systems[0].axes },
        { name: names[1], axes: systems[1].axes },
      ], version);
    } catch (invalid) {
      throw new Error(
        `${where} (${transform.type}) would be written as a transform this ` +
          `package cannot read back: ${
            invalid instanceof Error ? invalid.message : String(invalid)
          }`,
      );
    }
    parent.set(find(ends[0]), find(ends[1]));
  });

  const groups = new Map<string, string[]>();
  for (const node of parent.keys()) {
    const root = find(node);
    groups.set(root, [...(groups.get(root) ?? []), labels.get(node)!]);
  }
  if (groups.size > 1) {
    const listing = [...groups.values()].map((group) => group.sort()).sort();
    throw new Error(
      "The scene's coordinate systems and images must form one connected " +
        `graph of transformations, but they fall into ${groups.size} ` +
        `unconnected groups: ${JSON.stringify(listing)}.`,
    );
  }
}

/** Serialize a scene's metadata to its `ome.scene` object. */
export function sceneToOmeValue(scene: NgffScene): Record<string, unknown> {
  const document: Record<string, unknown> = {};
  if (scene.coordinateSystems !== undefined) {
    document.coordinateSystems = scene.coordinateSystems.map((system) => ({
      name: system.name,
      axes: system.axes,
    }));
  }
  document.coordinateTransformations = scene.coordinateTransformations.map(
    serializeV06Transform,
  );
  return document;
}

/** Parse an `ome.scene` object into a scene's metadata. */
export function sceneFromOmeValue(
  document: Record<string, unknown>,
  version?: string,
): Pick<NgffScene, "coordinateTransformations" | "coordinateSystems"> {
  let systems: CoordinateSystem[] | undefined;
  if (Array.isArray(document.coordinateSystems)) {
    systems = document.coordinateSystems.filter(isRecord).map((entry) => ({
      name: String(entry.name),
      axes: (Array.isArray(entry.axes) ? entry.axes : []) as Axis[],
    }));
  }
  const raw = Array.isArray(document.coordinateTransformations)
    ? document.coordinateTransformations.filter(isRecord)
    : [];
  const transforms = parseV06Transforms(
    raw,
    (systems ?? []).map((system) => system.name),
    systems ?? [],
    version,
  );
  // The multiscales parser keeps a reference only when its name resolves in
  // the systems it is given. A scene's references mostly name systems of its
  // images, so they are restored as written.
  transforms.forEach((transform, index) => {
    for (const side of ["input", "output"] as const) {
      const reference = raw[index][side];
      if (isRecord(reference)) {
        const identifier: CoordinateSystemIdentifier = {};
        if (typeof reference.path === "string") {
          identifier.path = reference.path;
        }
        if (typeof reference.name === "string") {
          identifier.name = reference.name;
        }
        transform[side] = identifier;
      } else {
        delete transform[side];
      }
    }
  });
  return {
    coordinateTransformations: transforms,
    ...(systems !== undefined && { coordinateSystems: systems }),
  };
}

/**
 * The part of `store` below `path`, as a store of its own: every key is
 * looked up under `/<path>`, range reads included when `store` has them, so
 * a sharded array below `path` still opens.
 */
function storeBelow(store: zarr.Readable, path: string): zarr.AsyncReadable {
  const below: zarr.AsyncReadable = {
    get: async (key, options) => await store.get(`/${path}${key}`, options),
  };
  if (typeof store.getRange === "function") {
    const getRange = store.getRange.bind(store);
    below.getRange = async (key, range, options) =>
      await getRange(`/${path}${key}`, range, options);
  }
  return below;
}

/**
 * The store of the image at `path` below the scene group: the path or URL
 * joined to the scene's, or for a store object, such as a `MemoryStore` or
 * a zipped `.ozx` store, the part of it below `path`.
 */
export function childStore(store: string, path: string): string;
export function childStore(
  store: string | zarr.Readable,
  path: string,
): string | zarr.Readable;
export function childStore(
  store: string | zarr.Readable,
  path: string,
): string | zarr.Readable {
  if (typeof store !== "string") {
    return storeBelow(store, path);
  }
  return `${store.replace(/\/+$/, "")}/${path}`;
}

/**
 * Read the scene `rootAttrs` declares, with the images it references;
 * `fromOmeZarr` calls this. `store` is the path, URL, or store object the
 * caller passed, below which `readImage` reads each referenced image.
 * `validate` validates each image and runs the spec checks of
 * {@link checkScene} on the result.
 */
export async function readScene(
  store: string | zarr.Readable,
  rootAttrs: Record<string, unknown>,
  readImage: ImageReader,
  options: ReadSceneOptions = {},
): Promise<NgffScene> {
  const validate = options.validate ?? false;
  const document = (rootAttrs.ome as Record<string, unknown>).scene;
  if (!isRecord(document)) {
    throw new Error(
      `The 'ome.scene' entry must be an object; got ${
        JSON.stringify(document)
      }.`,
    );
  }
  const version = readVersion(rootAttrs, options);
  const { coordinateTransformations, coordinateSystems } = sceneFromOmeValue(
    document,
    version,
  );
  const images: Record<string, NgffMultiscales> = {};
  for (const transform of coordinateTransformations) {
    for (const reference of [transform.input, transform.output]) {
      const path = reference?.path;
      if (path !== undefined && !Object.hasOwn(images, path)) {
        // The path comes from the store's own metadata and is joined to
        // the store, so it must not reach outside the scene group.
        checkImagePath(path);
        images[path] = await readImage(childStore(store, path), {
          validate,
          ...(options.cache !== undefined && { cache: options.cache }),
        });
      }
    }
  }
  const scene = new NgffScene({
    images,
    coordinateTransformations,
    ...(coordinateSystems !== undefined && { coordinateSystems }),
  });
  if (validate) {
    checkScene(scene, version);
  }
  return scene;
}

/**
 * The version to read `rootAttrs` with: the requested one, checked against
 * the stored one when validating (the 0.6 family counts as one version), or
 * the stored one.
 */
export function readVersion(
  rootAttrs: Record<string, unknown>,
  options: { validate?: boolean; version?: string },
): string {
  const detected = detectVersion(rootAttrs);
  const requested = options.version;
  if ((options.validate ?? false) && requested !== undefined) {
    const versionsMatch = detected === requested ||
      (isV06Version(detected) && isV06Version(requested));
    if (!versionsMatch) {
      throw new Error(
        `Expected OME-Zarr version ${requested}, but found ${detected}`,
      );
    }
  }
  return requested ?? detected;
}

/** Whether `rootAttrs` is the root document of a scene group. */
export function hasSceneMetadata(rootAttrs: Record<string, unknown>): boolean {
  return isRecord(rootAttrs.ome) && "scene" in rootAttrs.ome;
}

/**
 * Write a scene and its images to `store`; the Node and browser
 * `toOmeZarr` call this, and `toOmeZarrOzx` through {@link sceneToOzx}.
 *
 * The scene metadata lands in the root group's `ome.scene` and each image is
 * written below its path, at `options.version`, by the calling writer's
 * arrays (`arrayWriter`, given the codecs and sharding of `options`). The
 * scene is checked against the spec first ({@link checkScene}); nothing is
 * written when it fails. A transformation that references an array or a
 * field group by `path`, such as a `displacements` field, needs that node
 * written first, below the scene's store, and the scene written with
 * `overwrite: false`, which also keeps the root group's other attributes.
 * The consolidated block at the root lists every node the scene wrote: each
 * image's datasets and matrix arrays, and the groups above them.
 */
export async function writeSceneToStore(
  store: zarr.Mutable,
  scene: NgffScene,
  arrayWriter: ArrayWriterFactory,
  options: WriteSceneOptions = {},
): Promise<void> {
  const version = options.version ?? "0.6";
  if (!SCENE_VERSIONS.includes(version)) {
    throw new Error(
      `Scene metadata is defined from OME-Zarr 0.6; got version '${version}'.`,
    );
  }
  checkScene(scene, version);

  // An array-backed transformation points at a node of the store, which the
  // scene writer does not produce: it has to be there already, so it is
  // written first and the scene keeps it.
  const fields = [...fieldPaths(scene.coordinateTransformations)].sort();
  for (const path of fields) {
    checkImagePath(path);
  }
  if (fields.length > 0 && (options.overwrite ?? true)) {
    throw new Error(
      `The scene's transformations reference the nodes ${
        JSON.stringify(fields)
      }; write those first with toOmeZarr() below the scene's store, then ` +
        "the scene with overwrite: false so they are kept.",
    );
  }
  for (const path of fields) {
    if ((await store.get(`/${path}/zarr.json`)) === undefined) {
      throw new Error(
        `The scene's transformations reference '${path}', which the store ` +
          "does not hold; write it first with toOmeZarr() below the scene's " +
          "store, then the scene with overwrite: false.",
      );
    }
  }

  // A sharded array needs range reads, which an in-memory `Map` lacks.
  const root = zarr.root(ensureRangeReads(store));
  let attributes: Record<string, unknown> = {};
  if (options.overwrite === false) {
    try {
      const existing = await zarr.open(root, { kind: "group" });
      attributes = { ...(existing.attrs as Record<string, unknown>) };
    } catch {
      // No root group yet: nothing to keep.
    }
  }
  const ondisk = version === "0.6" ? V06_ONDISK_VERSION : version;
  attributes.ome = { version: ondisk, scene: sceneToOmeValue(scene) };
  await zarr.create(root, { attributes });

  const nodePaths = new Set<string>();
  const ensureAncestors = async (path: string): Promise<void> => {
    const segments = path.split("/");
    for (let end = 1; end < segments.length; end++) {
      const ancestor = segments.slice(0, end).join("/");
      nodePaths.add(ancestor);
      try {
        await zarr.open(root.resolve(ancestor), { kind: "group" });
      } catch {
        await zarr.create(root.resolve(ancestor));
      }
    }
    nodePaths.add(path);
  };
  for (const path of fields) {
    await ensureAncestors(path);
  }

  const writeArray = arrayWriter({
    codecs: options.codecs,
    chunksPerShard: options.chunksPerShard,
  });
  const { onProgress } = options;
  const totalChunks = onProgress
    ? Object.values(scene.images).reduce(
      (total, multiscales) =>
        total + countMultiscalesWrites(multiscales, options.chunksPerShard),
      0,
    )
    : 0;
  let completedChunks = 0;
  for (const [path, multiscales] of Object.entries(scene.images)) {
    await ensureAncestors(path);
    // Report each image's writes on top of the images before it.
    const offset = completedChunks;
    const written = await writeMultiscalesGroup(
      root.resolve(path),
      multiscales,
      writeArray,
      version,
      onProgress
        ? (completed: number) => onProgress(offset + completed, totalChunks)
        : null,
      options.chunksPerShard,
    );
    for (const node of written) {
      nodePaths.add(`${path}/${node}`);
    }
    if (onProgress) {
      completedChunks += countMultiscalesWrites(
        multiscales,
        options.chunksPerShard,
      );
    }
  }
  if (options.consolidateMetadata ?? true) {
    await consolidateMetadata(
      store as unknown as ConsolidatableStore,
      [...nodePaths].sort(),
    );
  }
}

/** The `toOmeZarrOzx` subset of the options a scene is zipped with. */
export interface SceneOzxOptions {
  /** OME-Zarr version, 0.6 (the default) or 0.9.dev1. */
  version?: OzxVersion | undefined;
  /**
   * How every image's arrays are sharded; 2 chunks a shard along every axis
   * unless given, the `.ozx` default.
   */
  chunksPerShard?: ChunksPerShard | undefined;
  /** Consolidate the whole archive's metadata at its root (default `true`). */
  consolidateMetadata?: boolean | undefined;
  /** Reports the writes of all the images together. */
  onProgress?: ProgressCallback | undefined;
}

/**
 * Refuse to write `transforms` in one piece, as an `.ozx` archive is, when
 * they reference stored nodes by `path`: such a node has to be in the store
 * before the scene or transformation is, and a fresh archive holds none.
 */
export function refuseStoredNodes(
  transforms: V06Transform[],
  what: "scene" | "transformation",
): void {
  const fields = [...fieldPaths(transforms)].sort();
  if (fields.length > 0) {
    throw new Error(
      `An .ozx archive is written in one piece, so a ${what} whose ` +
        `transformations reference the stored nodes ${
          JSON.stringify(fields)
        } cannot be zipped with toOmeZarrOzx(). Write it with those nodes ` +
        "to a store with toOmeZarr() instead.",
    );
  }
}

/**
 * `scene` and its images as an RFC-9 `.ozx` archive; the Node and browser
 * `toOmeZarrOzx` call this. The scene is written to a fresh `MemoryStore`
 * with {@link writeSceneToStore}, its images sharded as `.ozx` images are,
 * and zipped with the version in the archive's comment.
 */
export async function sceneToOzx(
  scene: NgffScene,
  arrayWriter: ArrayWriterFactory,
  options: SceneOzxOptions = {},
): Promise<Uint8Array> {
  const version = gateOzxVersion(options.version ?? "0.6");
  refuseStoredNodes(scene.coordinateTransformations, "scene");
  const store = new Map<string, Uint8Array>();
  await writeSceneToStore(store, scene, arrayWriter, {
    version,
    chunksPerShard: options.chunksPerShard ?? DEFAULT_OZX_CHUNKS_PER_SHARD,
    ...(options.consolidateMetadata !== undefined &&
      { consolidateMetadata: options.consolidateMetadata }),
    ...(options.onProgress !== undefined &&
      { onProgress: options.onProgress }),
  });
  return memoryStoreToZip(store, { version });
}
