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
} from "../utils/v06_metadata.ts";
import type { ChunkCache } from "../utils/worker_pool.ts";
import { fromOmeZarr } from "./from_ngff_zarr.ts";
import { toOmeZarr } from "./to_ngff_zarr.ts";

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
   * their paths hold either way.
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

/** Reject an image path that is not a plain path below the scene group. */
function checkImagePath(path: unknown): asserts path is string {
  if (
    typeof path !== "string" ||
    path === "" ||
    path.split("/").some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new Error(
      `Image path '${
        String(path)
      }' must be a relative path below the scene group.`,
    );
  }
}

/** The coordinate system names the image carries once written at 0.6. */
function imageSystemNames(multiscales: NgffMultiscales): string[] {
  const systems = multiscales.metadata.coordinateSystems;
  // The 0.6 writer declares the intrinsic system on an image that carries
  // no coordinate systems of its own.
  return systems !== undefined && systems.length > 0
    ? systems.map((system) => system.name)
    : [INTRINSIC_COORDINATE_SYSTEM_NAME];
}

type Node = ["scene" | "image", string];

/**
 * Throw unless every reference resolves and the graph is connected.
 *
 * The coordinate systems of the scene and of its images are the nodes of a
 * graph whose edges are the transformations, and the spec requires that
 * graph to be connected. An image's own systems are connected through its
 * multiscales already, so each image counts as one node.
 */
export function checkScene(scene: NgffScene): void {
  for (const path of Object.keys(scene.images)) {
    checkImagePath(path);
  }
  if (scene.coordinateTransformations.length === 0) {
    throw new Error("A scene declares at least one coordinate transformation.");
  }

  const local = new Set((scene.coordinateSystems ?? []).map((s) => s.name));
  const key = ([kind, name]: Node): string => `${kind}\0${name}`;
  const parent = new Map<string, string>();
  const labels = new Map<string, string>();
  for (const name of local) {
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
        if (!local.has(reference.name)) {
          throw new Error(
            `${where} names coordinate system '${reference.name}', which ` +
              "the scene does not declare. Declare it in coordinateSystems, " +
              "or give the reference the path of the image that declares it.",
          );
        }
        ends.push(key(["scene", reference.name]));
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
      const names = imageSystemNames(image);
      if (!names.includes(reference.name)) {
        throw new Error(
          `${where} names coordinate system '${reference.name}' of image ` +
            `'${reference.path}', which declares ${
              JSON.stringify([...names].sort())
            }.`,
        );
      }
      ends.push(key(["image", reference.path]));
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
  checkScene(scene);
  if (store.startsWith("http://") || store.startsWith("https://")) {
    throw new Error(
      "HTTP/HTTPS URLs are read-only and cannot be used for writing. Use a local file path instead.",
    );
  }

  const fsStore = await localStore(store);
  const root = zarr.root(fsStore);
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
  for (const [path, multiscales] of Object.entries(scene.images)) {
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
    checkScene(scene);
  }
  return scene;
}
