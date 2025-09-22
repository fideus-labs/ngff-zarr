// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";
import { Multiscales } from "../types/multiscales.ts";
import { NgffImage } from "../types/ngff_image.ts";
import type { Metadata, Omero } from "../types/zarr_metadata.ts";
import { MetadataSchema } from "../schemas/zarr_metadata.ts";
import type { Units } from "../types/units.ts";

export interface FromNgffZarrOptions {
  validate?: boolean;
  version?: "0.4" | "0.5";
}

export type MemoryStore = Map<string, Uint8Array>;

export async function fromNgffZarr(
  // Also accepts FileSystemStore, ZipFileStore objects in Node.js/Deno
  store: string | MemoryStore | zarr.FetchStore,
  options: FromNgffZarrOptions = {},
): Promise<Multiscales> {
  const validate = options.validate ?? false;
  const version = options.version;

  try {
    // Determine the appropriate store type based on the path
    let resolvedStore: MemoryStore | zarr.FetchStore;
    if (store instanceof Map || store instanceof zarr.FetchStore) {
      resolvedStore = store;
    } else if (store.startsWith("http://") || store.startsWith("https://")) {
      // Use FetchStore for HTTP/HTTPS URLs
      resolvedStore = new zarr.FetchStore(store);
    } else {
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
        // @ts-ignore: Node/Deno workaround
        if (store instanceof FileSystemStore || store instanceof ZipFileStore) {
          // @ts-ignore: Node/Deno workaround
          resolvedStore = store;
        } else {
          // Normalize the path for cross-platform compatibility
          const normalizedPath = store.replace(/^\/([A-Za-z]:)/, "$1");
          // @ts-ignore: Node/Deno workaround
          resolvedStore = new FileSystemStore(normalizedPath);
        }
      } catch (error) {
        throw new Error(
          `Failed to load FileSystemStore: ${error}. Use HTTP/HTTPS URLs for browser compatibility.`,
        );
      }
    }

    // Try to use consolidated metadata for better performance
    let optimizedStore;
    try {
      // @ts-ignore: tryWithConsolidated typing
      optimizedStore = await zarr.tryWithConsolidated(resolvedStore);
    } catch {
      optimizedStore = resolvedStore;
    }

    const root = await zarr.open(optimizedStore as zarr.Readable, {
      kind: "group",
    });
    const attrs = root.attrs as unknown;

    if (!attrs || !(attrs as Record<string, unknown>).multiscales) {
      throw new Error("No multiscales metadata found in Zarr store");
    }

    const multiscalesArray = (attrs as Record<string, unknown>)
      .multiscales as unknown[];
    if (!Array.isArray(multiscalesArray) || multiscalesArray.length === 0) {
      throw new Error("No multiscales metadata found in Zarr store");
    }
    const multiscalesMetadata = multiscalesArray[0] as unknown;

    if (validate) {
      const result = MetadataSchema.safeParse(multiscalesMetadata);
      if (!result.success) {
        throw new Error(`Invalid OME-Zarr metadata: ${result.error.message}`);
      }

      // Check version compatibility if specified
      if (version) {
        const metadataWithVersion = multiscalesMetadata as { version?: string };
        if (metadataWithVersion.version !== version) {
          throw new Error(
            `Expected OME-Zarr version ${version}, but found ${
              metadataWithVersion.version || "unknown"
            }`,
          );
        }
      }
    }

    const metadata = multiscalesMetadata as Metadata;

    // Extract omero metadata from root attributes if present
    if ((attrs as Record<string, unknown>).omero) {
      const omeroData = (attrs as Record<string, unknown>).omero as Record<
        string,
        unknown
      >;

      // Handle backward compatibility for OMERO window metadata
      if (omeroData.channels && Array.isArray(omeroData.channels)) {
        for (
          const channel of omeroData.channels as Array<
            Record<string, unknown>
          >
        ) {
          if (channel.window && typeof channel.window === "object") {
            const window = channel.window as Record<string, number | undefined>;

            // Ensure both min/max and start/end are present for compatibility
            if (window.min !== undefined && window.max !== undefined) {
              // If only min/max present, use them for start/end
              if (window.start === undefined) {
                window.start = window.min;
              }
              if (window.end === undefined) {
                window.end = window.max;
              }
            } else if (window.start !== undefined && window.end !== undefined) {
              // If only start/end present, use them for min/max
              if (window.min === undefined) {
                window.min = window.start;
              }
              if (window.max === undefined) {
                window.max = window.end;
              }
            }
          }
        }
      }

      metadata.omero = omeroData as unknown as Omero;
    }

    const images: NgffImage[] = [];

    for (const dataset of metadata.datasets) {
      const arrayPath = dataset.path;

      const zarrArray = (await zarr.open(root.resolve(arrayPath), {
        kind: "array",
      })) as zarr.Array<zarr.DataType, zarr.Readable>;

      // Verify we have an array with the expected properties
      if (
        !zarrArray ||
        !("shape" in zarrArray) ||
        !("dtype" in zarrArray) ||
        !("chunks" in zarrArray)
      ) {
        throw new Error(
          `Invalid zarr array at path ${arrayPath}: missing shape property`,
        );
      }

      const scale: Record<string, number> = {};
      const translation: Record<string, number> = {};

      for (const transform of dataset.coordinateTransformations) {
        if (transform.type === "scale") {
          metadata.axes.forEach((axis, i) => {
            if (i < transform.scale.length) {
              scale[axis.name] = transform.scale[i];
            }
          });
        } else if (transform.type === "translation") {
          metadata.axes.forEach((axis, i) => {
            if (i < transform.translation.length) {
              translation[axis.name] = transform.translation[i];
            }
          });
        }
      }

      const dims = metadata.axes.map((axis) => axis.name);
      const axesUnits = metadata.axes.reduce((acc, axis) => {
        if (axis.unit) {
          acc[axis.name] = axis.unit;
        }
        return acc;
      }, {} as Record<string, Units>);

      const ngffImage = new NgffImage({
        data: zarrArray,
        dims,
        scale,
        translation,
        name: metadata.name,
        axesUnits: Object.keys(axesUnits).length > 0 ? axesUnits : undefined,
        computedCallbacks: undefined,
      });

      images.push(ngffImage);
    }

    return new Multiscales({
      images,
      metadata,
      scaleFactors: undefined,
      method: undefined,
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
      return await zarr.get(zarrArray, selection);
    } else {
      return await zarr.get(zarrArray);
    }
  } catch (error) {
    throw new Error(
      `Failed to read array data: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}
