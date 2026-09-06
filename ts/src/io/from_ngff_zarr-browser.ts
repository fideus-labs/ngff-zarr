// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible version of from_ngff_zarr that doesn't import @zarrita/storage
// (which contains Node.js-specific modules like node:fs, node:buffer, node:path)
import * as zarr from "zarrita";

import { MetadataSchema } from "../schemas/zarr_metadata.ts";
import { NgffMultiscales } from "../types/multiscales.ts";
import { NgffImage } from "../types/ngff_image.ts";
import type { AxisUnit } from "../types/units.ts";
import type { Metadata, Omero } from "../types/zarr_metadata.ts";
import { extractMethodMetadata } from "../utils/parse_metadata.ts";
import { fromZarrAttrsV06 } from "../utils/from_zarr_attrs.ts";
import {
  type ImageVersion,
  isV06Version,
  NgffVersion,
} from "../types/supported_versions.ts";
import {
  distributedSinglescaleTransforms,
  multiscalesEntryFromOme,
} from "../utils/rfc8_multiscale.ts";
import { validateCollection } from "../utils/rfc8_validation.ts";
import { OmeNodeDocumentSchema } from "../schemas/rfc8.ts";

export type { ChunkCache } from "../utils/worker_pool.ts";

export interface FromOmeZarrOptions {
  /** Enable schema validation of OME-Zarr metadata. */
  validate?: boolean;
  /** Expected OME-Zarr version. */
  version?: ImageVersion;
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

/** @deprecated Use {@link FromOmeZarrOptions} instead. */
export type FromNgffZarrOptions = FromOmeZarrOptions;

export type MemoryStore = Map<string, Uint8Array>;

/**
 * Browser-compatible version of fromOmeZarr.
 * Supports HTTP/HTTPS URLs, MemoryStore (Map), FetchStore, and any
 * zarrita-compatible Readable store (e.g. TiffStore from @fideus-labs/fiff).
 * Does NOT support local file paths (use the full version in Node.js/Deno).
 */
export async function fromOmeZarr(
  store: string | MemoryStore | zarr.FetchStore | zarr.Readable,
  options: FromOmeZarrOptions = {},
): Promise<NgffMultiscales> {
  const validate = options.validate ?? false;
  const version = options.version;

  try {
    // Determine the appropriate store type based on the path
    let resolvedStore: MemoryStore | zarr.FetchStore | zarr.Readable;
    if (
      typeof store === "object" &&
      store !== null &&
      "get" in store &&
      typeof (store as zarr.Readable).get === "function"
    ) {
      // Duck-type check for zarrita Readable stores (e.g. TiffStore, FetchStore, MemoryStore)
      resolvedStore = store as zarr.Readable;
    } else if (store instanceof Map || store instanceof zarr.FetchStore) {
      // Defensive fallback for Map/FetchStore (normally caught by duck-type check above)
      resolvedStore = store;
    } else if (
      typeof store === "string" &&
      (store.startsWith("http://") || store.startsWith("https://"))
    ) {
      // Use FetchStore for HTTP/HTTPS URLs
      resolvedStore = new zarr.FetchStore(store);
    } else {
      // Local file paths are not supported in browser environments
      throw new Error(
        "Local file paths are not supported in browser environments. Use HTTP/HTTPS URLs, MemoryStore, or a Readable store instead.",
      );
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

    // OME-Zarr v0.6 (RFC 5) uses a different on-disk shape (coordinateSystems +
    // per-dataset sequence transforms) than the inline v0.4/v0.5 parser below,
    // so delegate to the shared, environment-agnostic v0.6 reader.
    let rootAttrsForVersion = attrs as Record<string, unknown>;
    let omeForVersion = rootAttrsForVersion.ome as
      | Record<string, unknown>
      | undefined;
    let onDiskVersion = omeForVersion &&
        typeof omeForVersion.version === "string"
      ? omeForVersion.version
      : undefined;
    // RFC-8: at 0.9.dev3 the root `ome` value is a typed node document, not
    // a multiscales wrapper. A multiscale node converts to the 0.9.dev1
    // entry shape and reads through the same delegate; any other node type
    // is a collection-model document.
    let isDev2Multiscale = false;
    if (onDiskVersion === NgffVersion.V09dev3 && omeForVersion) {
      if (omeForVersion.type !== "multiscale") {
        throw new Error(
          `The input is an OME-Zarr 0.9.dev3 store whose root is a ` +
            `'${omeForVersion.type}' node. 0.9.dev3 stores the RFC-8 node ` +
            `model in place of the multiscales metadata; use ` +
            `fromCollectionZarr() to read it. fromOmeZarr() reads ` +
            `multiscale image stores.`,
        );
      }
      if (validate) {
        validateCollection(omeForVersion, onDiskVersion);
        OmeNodeDocumentSchema.parse(omeForVersion);
      }
      const singlescale = await distributedSinglescaleTransforms(
        omeForVersion,
        resolvedStore as zarr.Readable,
      );
      const { entry, omero } = multiscalesEntryFromOme(
        omeForVersion,
        singlescale,
      );
      omeForVersion = {
        version: NgffVersion.V09dev1,
        multiscales: [entry],
        ...(omero !== undefined && { omero }),
      };
      onDiskVersion = NgffVersion.V09dev1;
      rootAttrsForVersion = { ...rootAttrsForVersion, ome: omeForVersion };
      isDev2Multiscale = true;
    }

    // 0.9.dev1 carries the same coordinateSystems layout as v0.6, so it reads
    // through the same delegate. Without this the store falls through to the
    // v0.4/v0.5 parser below, which expects a flat `axes` entry and cannot read
    // what this package's own 0.9 writer produced.
    if (
      omeForVersion && onDiskVersion !== undefined &&
      (isV06Version(onDiskVersion) || onDiskVersion === NgffVersion.V09dev1)
    ) {
      // Gate the requested-version mismatch behind `validate`, matching the node
      // reader and the v0.4/v0.5 path below; otherwise behavior diverges by
      // version and environment. The v0.6 family (`0.6` and its pre-release
      // tags) is treated as equivalent.
      const effectiveVersion = isDev2Multiscale
        ? NgffVersion.V09dev3
        : onDiskVersion;
      const versionsMatch = version === undefined ||
        version === effectiveVersion ||
        (isV06Version(effectiveVersion) && isV06Version(version));
      if (validate && !versionsMatch) {
        throw new Error(
          `Expected OME-Zarr version ${version}, but found ${effectiveVersion}`,
        );
      }
      const result = await fromZarrAttrsV06(
        rootAttrsForVersion,
        resolvedStore,
        validate,
      );
      // The v0.6 delegate records `0.6`, which is right for the whole v0.6
      // family but not for a 0.9.dev1 store: that string is the on-disk
      // version, and callers read `metadata.version` to tell them apart.
      if (onDiskVersion === NgffVersion.V09dev1) {
        result.metadata.version = isDev2Multiscale
          ? NgffVersion.V09dev3
          : NgffVersion.V09dev1;
      }
      const entry = (omeForVersion.multiscales as unknown[])[0] as Record<
        string,
        unknown
      >;
      const { method, methodType, methodMetadata } = extractMethodMetadata(
        entry,
      );
      if (methodType) {
        result.metadata.type = methodType;
      }
      if (methodMetadata) {
        result.metadata.metadata = methodMetadata;
      }
      return new NgffMultiscales({
        images: result.images,
        metadata: result.metadata,
        scaleFactors: undefined,
        method,
        chunks: undefined,
      });
    }

    // Handle both 0.4 (multiscales at root) and 0.5 (multiscales under ome) formats
    let multiscalesArray: unknown[];
    let detectedVersion: string | undefined;

    if (attrs && typeof attrs === "object" && "ome" in attrs) {
      // OME-Zarr 0.5 format: metadata is under the 'ome' property
      const omeAttrs = (attrs as Record<string, unknown>).ome as Record<
        string,
        unknown
      >;
      if (!omeAttrs || !omeAttrs.multiscales) {
        throw new Error("No multiscales metadata found in OME-Zarr 0.5 store");
      }
      multiscalesArray = omeAttrs.multiscales as unknown[];
      detectedVersion = omeAttrs.version as string | undefined;
    } else if (attrs && typeof attrs === "object" && "multiscales" in attrs) {
      // OME-Zarr 0.4 format: metadata is at root
      multiscalesArray = (attrs as Record<string, unknown>)
        .multiscales as unknown[];
    } else {
      throw new Error("No multiscales metadata found in Zarr store");
    }

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
        const actualVersion = metadataWithVersion.version || detectedVersion;
        if (actualVersion !== version) {
          throw new Error(
            `Expected OME-Zarr version ${version}, but found ${
              actualVersion || "unknown"
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
      const axesUnits = metadata.axes.reduce(
        (acc, axis) => {
          if (axis.unit) {
            acc[axis.name] = axis.unit;
          }
          return acc;
        },
        {} as Record<string, AxisUnit>,
      );

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

    // Extract method metadata from the multiscales entry (mirrors non-browser version)
    const { method, methodType, methodMetadata } = extractMethodMetadata(
      multiscalesMetadata as Record<string, unknown>,
    );
    if (methodType) {
      metadata.type = methodType;
    }
    if (methodMetadata) {
      metadata.metadata = methodMetadata;
    }

    return new NgffMultiscales({
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

/** @deprecated Use {@link fromOmeZarr} instead. */
export const fromNgffZarr = fromOmeZarr;
