// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The Node.js scene writer; `toOmeZarr` calls it. The model, the checks and
 * the reader live in `scene_common.ts`, which the browser module shares.
 */
import * as zarr from "zarrita";
import type { NgffScene } from "../types/scene.ts";
import { V06_ONDISK_VERSION } from "../types/supported_versions.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import {
  type ConsolidatableStore,
  consolidateMetadata,
  datasetNodePaths,
} from "../utils/consolidate_metadata.ts";
import type { ChunksPerShard } from "../utils/sharding.ts";
import {
  checkImagePath,
  checkScene,
  childStore,
  fieldPaths,
  SCENE_VERSIONS,
  sceneToOmeValue,
} from "./scene_common.ts";
import { toOmeZarr } from "./to_ngff_zarr.ts";

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
  /** Passed to {@link toOmeZarr} for every image. */
  chunksPerShard?: ChunksPerShard;
  /** Passed to {@link toOmeZarr} for every image. */
  codecs?: ZarrCodec[];
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
 * Write a scene and its images to a directory store; `toOmeZarr` calls this.
 *
 * The scene metadata lands in the root group's `ome.scene` and each image is
 * written below its path with {@link toOmeZarr}, at `options.version`. The
 * scene is checked against the spec first ({@link checkScene}); nothing is
 * written when it fails. A transformation that references an array or a
 * field group by `path`, such as a `displacements` field, needs that node
 * written first, below the scene's store, and the scene written with
 * `overwrite: false`, which also keeps the root group's other attributes.
 */
export async function writeScene(
  store: string,
  scene: NgffScene,
  options: WriteSceneOptions = {},
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
