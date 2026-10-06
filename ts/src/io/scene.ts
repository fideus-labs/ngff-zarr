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
 */
import * as zarr from "zarrita";
import type { NgffMultiscales } from "../types/multiscales.ts";
import {
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
import type { ZarrCodec } from "../utils/codecs.ts";
import {
  type ConsolidatableStore,
  consolidateMetadata,
  datasetNodePaths,
} from "../utils/consolidate_metadata.ts";
import { detectVersion } from "../utils/parse_metadata.ts";
import type { ChunksPerShard } from "../utils/sharding.ts";
import {
  parseV06Transforms,
  serializeV06Transform,
  validateV06Transform,
} from "../utils/v06_metadata.ts";
import type { ChunkCache } from "../utils/worker_pool.ts";
import { fromOmeZarr } from "./from_ngff_zarr.ts";
import { toOmeZarr } from "./to_ngff_zarr.ts";
import { gateAxisViews, gateSpans } from "./to_ngff_zarr_ozx_common.ts";

/** The versions whose metadata model defines scenes. */
export const SCENE_VERSIONS: readonly string[] = [
  NgffVersion.V06,
  NgffVersion.V09dev1,
];

/**
 * Images that share a spatial relationship, and the transformations between
 * them.
 *
 * `images` maps each image's path below the scene group to its multiscales.
 * `coordinateTransformations` relate the images' coordinate systems to each
 * other and to `coordinateSystems`, the systems the scene declares itself.
 * Each end of a transformation is a {@link CoordinateSystemIdentifier}: with
 * a `path` it names a coordinate system of that image, without one a system
 * of the scene. The first scene coordinate system is the reference a viewer
 * displays by default.
 */
export interface NgffScene {
  images: Record<string, NgffMultiscales>;
  coordinateTransformations: V06Transform[];
  coordinateSystems?: CoordinateSystem[];
}

export interface ToSceneZarrOptions {
  /** OME-Zarr specification version, 0.6 (the default) or later. */
  version?: "0.6" | "0.9.dev1";
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
  /** Passed to {@link toOmeZarr} for every image. */
  chunksPerShard?: ChunksPerShard;
  /** Passed to {@link toOmeZarr} for every image. */
  codecs?: ZarrCodec[];
}

export interface FromSceneZarrOptions {
  /**
   * Check that every transformation end resolves and that the coordinate
   * systems and images form one connected graph, and validate each image.
   */
  validate?: boolean;
  /** OME-Zarr version, if known. */
  version?: "0.6" | "0.9.dev1";
  /** Passed to {@link fromOmeZarr} for every image. */
  cache?: ChunkCache;
}

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
function checkImagePath(path: unknown): asserts path is string {
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
function fieldPaths(transforms: V06Transform[]): Set<string> {
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

/** The store of the image at `path` below the scene group. */
function childStore(store: string, path: string): string {
  return `${store.replace(/\/+$/, "")}/${path}`;
}

/** The file system store at `store`, in Node.js and Deno. */
async function localStore(store: string): Promise<zarr.Mutable> {
  if (typeof window !== "undefined") {
    throw new Error(
      "Local file paths are not supported in browser environments.",
    );
  }
  const { FileSystemStore } = await import("@zarrita/storage");
  return new FileSystemStore(
    store.replace(/^\/([A-Za-z]:)/, "$1"),
  ) as unknown as zarr.Mutable;
}

/**
 * Write a scene and its images to an OME-Zarr store.
 *
 * The scene metadata lands in the root group's `ome.scene` and each image is
 * written below its path with {@link toOmeZarr}, at `options.version`. Every
 * transformation end has to resolve, to a coordinate system the scene
 * declares or to one of the image at its path, and the coordinate systems and
 * images have to form one connected graph; otherwise an error is thrown and
 * nothing is written.
 *
 * @param store - Path to a directory in the file system
 * @param scene - The scene to write
 * @param options - Writing options
 */
export async function toSceneZarr(
  store: string,
  scene: NgffScene,
  options: ToSceneZarrOptions = {},
): Promise<void> {
  const version = options.version ?? "0.6";
  if (!SCENE_VERSIONS.includes(version)) {
    throw new Error(
      `Scene metadata is defined from OME-Zarr 0.6; got version '${version}'.`,
    );
  }
  checkScene(scene, version);
  if (store.startsWith("http://") || store.startsWith("https://")) {
    throw new Error(
      "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use a local file path instead.",
    );
  }

  const fsStore = await localStore(store);
  const root = zarr.root(fsStore);
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
    if ((await fsStore.get(`/${path}/zarr.json`)) === undefined) {
      throw new Error(
        `The scene's transformations reference '${path}', which the store ` +
          "does not hold; write it first with toOmeZarr() below the scene's " +
          "store, then the scene with overwrite: false.",
      );
    }
  }
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
  for (const [path, multiscales] of Object.entries(scene.images)) {
    await ensureAncestors(path);
    await toOmeZarr(childStore(store, path), multiscales, {
      version,
      consolidateMetadata: false,
      ...(options.codecs !== undefined && { codecs: options.codecs }),
      ...(options.chunksPerShard !== undefined &&
        { chunksPerShard: options.chunksPerShard }),
    });
    for (
      const node of datasetNodePaths(
        multiscales.metadata.datasets.map((dataset) => dataset.path),
      )
    ) {
      nodePaths.add(`${path}/${node}`);
    }
  }
  if (options.consolidateMetadata ?? true) {
    await consolidateMetadata(
      fsStore as unknown as ConsolidatableStore,
      [...nodePaths].sort(),
    );
  }
}

/**
 * Read a scene and the images its transformations reference.
 *
 * @param store - Path to a directory in the file system, or an HTTP(S) URL
 * @param options - Reading options
 * @returns The scene, with the images its transformations reference by path;
 *   pixel data is read lazily
 */
export async function fromSceneZarr(
  store: string,
  options: FromSceneZarrOptions = {},
): Promise<NgffScene> {
  const validate = options.validate ?? false;
  const resolvedStore =
    store.startsWith("http://") || store.startsWith("https://")
      ? new zarr.FetchStore(store)
      : await localStore(store);
  const root = await zarr.open(zarr.root(resolvedStore), { kind: "group" });
  const rootAttrs = root.attrs as Record<string, unknown>;
  const ome = rootAttrs.ome;
  const document = isRecord(ome) ? ome.scene : undefined;
  if (!isRecord(document)) {
    throw new Error(
      `No scene metadata at '${store}': the root group carries no ` +
        "'ome.scene' entry. An image store is read with fromOmeZarr().",
    );
  }
  const version = options.version ?? detectVersion(rootAttrs);
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
        images[path] = await fromOmeZarr(childStore(store, path), {
          validate,
          ...(options.cache !== undefined && { cache: options.cache }),
        });
      }
    }
  }
  const scene: NgffScene = {
    images,
    coordinateTransformations,
    ...(coordinateSystems !== undefined && { coordinateSystems }),
  };
  if (validate) {
    checkScene(scene, version);
  }
  return scene;
}
