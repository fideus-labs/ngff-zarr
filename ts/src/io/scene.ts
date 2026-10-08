// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The Node.js side of the scene and transformation writers: the store for a
 * directory path, which the browser module has no access to. The writers
 * themselves, `writeSceneToStore` and `writeTransformation`, have no Node.js
 * dependency and are shared with the browser writer; `toOmeZarr` calls them
 * with this store, or with a `MemoryStore`.
 */
import type * as zarr from "zarrita";

/** The file system store at `store`, in Node.js and Deno. */
export async function localStore(store: string): Promise<zarr.Mutable> {
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
