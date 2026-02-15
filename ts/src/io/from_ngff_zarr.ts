// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";

import { Multiscales } from "../types/multiscales.ts";
import { NgffVersion } from "../types/supported_versions.ts";
import {
  fromZarrAttrsV04,
  fromZarrAttrsV05,
} from "../utils/from_zarr_attrs.ts";
import {
  detectVersion,
  extractMethodMetadata,
} from "../utils/parse_metadata.ts";
import { zarrGet } from "../utils/worker_pool.ts";

export type { ChunkCache } from "../utils/worker_pool.ts";

export interface FromNgffZarrOptions {
  /** Enable schema validation of OME-Zarr metadata. */
  validate?: boolean;
  /** Expected OME-Zarr version. */
  version?: "0.4" | "0.5";
  /**
   * Optional decoded-chunk cache passed to `zarrGet` calls.
   *
   * Any object with `get(key)` and `set(key, value)` works — a plain `Map`
   * is the simplest option. For bounded memory use an LRU cache.
   *
   * @see {@link https://github.com/fideus-labs/worker-pool/tree/main/fizarrita#chunk-caching}
   */
  cache?: import("../utils/worker_pool.ts").ChunkCache;
}

export type MemoryStore = Map<string, Uint8Array>;

export async function fromNgffZarr(
  // Also accepts FileSystemStore, ZipFileStore, or any zarrita Readable store
  store: string | MemoryStore | zarr.FetchStore | zarr.Readable,
  options: FromNgffZarrOptions = {},
): Promise<Multiscales> {
  const validate = options.validate ?? false;
  const requestedVersion = options.version;

  try {
    // Determine the appropriate store type based on the path
    let resolvedStore: MemoryStore | zarr.FetchStore | zarr.Readable;
    if (store instanceof Map || store instanceof zarr.FetchStore) {
      resolvedStore = store;
    } else if (
      typeof store === "object" &&
      store !== null &&
      "get" in store &&
      typeof (store as zarr.Readable).get === "function"
    ) {
      // Duck-type check for zarrita Readable stores (e.g. TiffStore)
      resolvedStore = store as zarr.Readable;
    } else if (
      typeof store === "string" &&
      (store.startsWith("http://") || store.startsWith("https://"))
    ) {
      // Use FetchStore for HTTP/HTTPS URLs
      resolvedStore = new zarr.FetchStore(store);
    } else {
      // Only strings (local paths) reach this branch
      const storePath = store as string;

      // For local paths, check if we're in a browser environment
      if (typeof window !== "undefined") {
        throw new Error(
          "Local file paths are not supported in browser environments. Use HTTP/HTTPS URLs instead.",
        );
      }

      // Use dynamic import for FileSystemStore, ZipFileStore in Node.js/Deno environments
      try {
        const { FileSystemStore, ZipFileStore } = await import(
          "@zarrita/storage"
        );
        if (store instanceof FileSystemStore || store instanceof ZipFileStore) {
          resolvedStore = store as unknown as zarr.Readable;
        } else {
          // Normalize the path for cross-platform compatibility
          const normalizedPath = storePath.replace(/^\/([A-Za-z]:)/, "$1");
          resolvedStore = new FileSystemStore(
            normalizedPath,
          ) as unknown as zarr.Readable;
        }
      } catch (error) {
        throw new Error(
          `Failed to load FileSystemStore: ${error}. Use HTTP/HTTPS URLs for browser compatibility.`,
        );
      }
    }

    // Try to use consolidated metadata for better performance
    let optimizedStore: zarr.Readable | zarr.Listable<zarr.Readable>;
    try {
      optimizedStore = await zarr.tryWithConsolidated(
        resolvedStore as zarr.Readable,
      );
    } catch {
      optimizedStore = resolvedStore as zarr.Readable;
    }

    const root = await zarr.open(optimizedStore as zarr.Readable, {
      kind: "group",
    });
    const attrs = root.attrs as unknown;
    const rootAttrs = attrs as Record<string, unknown>;

    // Handle both v0.4 (multiscales at root) and v0.5 (multiscales under "ome")
    const hasOmeWrapper = "ome" in rootAttrs;
    const multiscalesSource = hasOmeWrapper
      ? (rootAttrs.ome as Record<string, unknown>)
      : rootAttrs;

    if (!multiscalesSource.multiscales) {
      throw new Error("No multiscales metadata found in Zarr store");
    }

    // Detect version from attributes
    const detectedVersion = detectVersion(rootAttrs);

    // Validate version if requested
    if (validate && requestedVersion) {
      if (detectedVersion !== requestedVersion) {
        throw new Error(
          `Expected OME-Zarr version ${requestedVersion}, but found ${detectedVersion}`,
        );
      }
    }

    // Parse metadata using version-specific function
    let result;
    if (detectedVersion === NgffVersion.V05) {
      result = await fromZarrAttrsV05(rootAttrs, resolvedStore, validate);
    } else {
      result = await fromZarrAttrsV04(rootAttrs, resolvedStore, validate);
    }

    const { metadata, images } = result;

    // Extract method metadata from the multiscales metadata
    const multiscalesArray = multiscalesSource.multiscales as unknown[];
    const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;
    const { method, methodType, methodMetadata } = extractMethodMetadata(
      multiscalesMetadata,
    );

    // Update metadata with method information
    if (methodType) {
      metadata.type = methodType;
    }
    if (methodMetadata) {
      metadata.metadata = methodMetadata;
    }

    return new Multiscales({
      images,
      metadata,
      scaleFactors: undefined,
      method,
      chunks: undefined,
    });
  } catch (error) {
    throw new Error(
      `Failed to read OME-Zarr: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

export async function readArrayData(
  storePath: string,
  arrayPath: string,
  selection?: (number | null)[],
): Promise<unknown> {
  try {
    const store = new zarr.FetchStore(storePath);
    const root = zarr.root(store);

    // Try to open as zarr v2 first, then v3 if that fails
    let zarrArray;
    try {
      zarrArray = await zarr.open.v2(root.resolve(arrayPath), {
        kind: "array",
      });
    } catch (v2Error) {
      try {
        zarrArray = await zarr.open.v3(root.resolve(arrayPath), {
          kind: "array",
        });
      } catch (v3Error) {
        throw new Error(
          `Failed to open zarr array ${arrayPath} as either v2 or v3 format. v2 error: ${v2Error}. v3 error: ${v3Error}`,
        );
      }
    }

    if (selection) {
      return await zarrGet(zarrArray, selection);
    } else {
      return await zarrGet(zarrArray);
    }
  } catch (error) {
    throw new Error(
      `Failed to read array data: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}
