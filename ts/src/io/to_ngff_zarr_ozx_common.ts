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
import {
  buildV06MultiscalesEntry,
  legacyTopLevelTransforms,
} from "../utils/v06_metadata.ts";
import { V06_ONDISK_VERSION } from "../types/supported_versions.ts";
import type { MemoryStore } from "./rfc9_zip.ts";

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

/**
 * Build the root-group attributes for an OME-Zarr store at a given spec version
 * from the version-agnostic in-memory metadata. This is the single source of
 * truth for the on-disk root shape shared by the Node and browser writers
 * ({@link toOmeZarr}) and the in-place metadata rewrite in `upgradeOmeZarr`:
 *
 * - **0.6 (RFC 5):** coordinate systems + per-dataset `sequence` transforms,
 *   wrapped under the `ome` namespace and tagged {@link V06_ONDISK_VERSION}.
 * - **0.5:** axes carried directly on the multiscale entry, wrapped under `ome`.
 * - **0.4:** axes carried directly on the multiscale entry at the root (no
 *   `ome` wrapper).
 *
 * Richer v0.6-only top-level transforms are reduced to the simple
 * scale/translation subset for 0.4/0.5 via {@link legacyTopLevelTransforms}.
 */
export function buildRootAttributes(
  metadata: MetadataInterface,
  version: "0.4" | "0.5" | "0.6",
): Record<string, unknown> {
  // Process axes (orientation included when present).
  const processedAxes = processAxes(metadata.axes);

  if (version === "0.6") {
    const v06Entry = buildV06MultiscalesEntry(metadata, processedAxes);
    return {
      ome: {
        // Tag the store with the pre-release the bundled schemas carry, not
        // with the bare `"0.6"` that was requested; see V06_ONDISK_VERSION.
        version: V06_ONDISK_VERSION,
        multiscales: [v06Entry],
        ...(metadata.omero && { omero: metadata.omero }),
      },
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

  return version === "0.5"
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
}

/**
 * Validate that the NgffMultiscales metadata version is compatible with RFC-9.
 * RFC-9 always uses version 0.5.
 *
 * @param multiscales - NgffMultiscales object to validate
 * @throws Error if metadata.version is defined and not "0.5"
 */
export function validateRfc9Version(multiscales: NgffMultiscales): void {
  const providedVersion = multiscales.metadata.version;
  if (providedVersion !== undefined && providedVersion !== "0.5") {
    throw new Error(
      `Inconsistent NGFF version in NgffMultiscales metadata: expected "0.5" for RFC-9 OZX export, but got "${providedVersion}".`,
    );
  }
}

/**
 * Prepare multiscales metadata for RFC-9 export.
 * Processes axes (anatomical orientation is serialized whenever present) and
 * sets version to 0.5.
 *
 * @param multiscales - Source multiscales data
 * @returns Prepared metadata object
 */
export function prepareRfc9Metadata(
  multiscales: NgffMultiscales,
): Record<string, unknown> {
  const _version = "0.5";

  // Process axes (orientation included when present)
  const processedAxes = processAxes(multiscales.metadata.axes);

  // RFC-9 (.ozx) is always v0.5, which only supports the simple
  // scale/translation subset of top-level transforms. Reduce any v0.6-only
  // transforms (rotation/affine/sequence, or input/output/name on a
  // scale/translation) to that subset so they cannot leak into the store,
  // matching the v0.4/v0.5 branch of the regular writer.
  const legacyTransforms = legacyTopLevelTransforms(
    multiscales.metadata.coordinateTransformations,
  );

  return {
    version: _version,
    name: multiscales.metadata.name,
    axes: processedAxes,
    datasets: multiscales.metadata.datasets,
    ...(legacyTransforms && {
      coordinateTransformations: legacyTransforms,
    }),
    ...(multiscales.metadata.type && {
      type: multiscales.metadata.type,
    }),
    ...(multiscales.metadata.metadata && {
      metadata: multiscales.metadata.metadata,
    }),
  };
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

/**
 * Create root group attributes for RFC-9 export.
 *
 * @param multiscalesMetadata - Prepared multiscales metadata
 * @param multiscales - Original multiscales (for OMERO metadata)
 * @returns Attributes object for root group
 */
export function createRfc9RootAttributes(
  multiscalesMetadata: Record<string, unknown>,
  multiscales: NgffMultiscales,
): Record<string, unknown> {
  const _version = "0.5";

  const attributes: Record<string, unknown> = {
    ome: {
      version: _version,
      multiscales: [multiscalesMetadata],
    },
  };

  // Add OMERO metadata at root level if present
  if (multiscales.metadata.omero) {
    attributes.omero = multiscales.metadata.omero;
  }

  return attributes;
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
 */
export async function writeNgffMultiscalesToMemoryStore(
  store: MemoryStore,
  multiscales: NgffMultiscales,
  writeImage: (
    group: zarr.Group<MemoryStore>,
    image: NgffImage,
    path: string,
    onProgress?:
      | ((completedChunks: number, totalChunks: number) => void)
      | null,
  ) => Promise<void>,
  onProgress?: ((completedChunks: number, totalChunks: number) => void) | null,
): Promise<void> {
  // Validate version
  validateRfc9Version(multiscales);

  // Create root location and group with zarrita API
  const root = zarr.root(store);

  // Prepare metadata
  const multiscalesMetadata = prepareRfc9Metadata(multiscales);

  // Create root attributes
  const attributes = createRfc9RootAttributes(multiscalesMetadata, multiscales);

  const rootGroup = await zarr.create(root, { attributes });

  // Pre-calculate total chunk count across all images for cumulative progress
  let totalChunks = 0;
  if (onProgress) {
    for (const image of multiscales.images) {
      const shape = image.data.shape;
      const chunks = getChunksFromImage(image);
      let imageChunks = 1;
      for (let d = 0; d < shape.length; d++) {
        imageChunks *= Math.ceil(shape[d] / chunks[d]);
      }
      totalChunks += imageChunks;
    }
  }

  // Write each image in the multiscales
  let completedChunks = 0;
  for (let i = 0; i < multiscales.images.length; i++) {
    const image = multiscales.images[i];
    const dataset = multiscales.metadata.datasets[i];

    if (!dataset) {
      throw new Error(`No dataset configuration found for image ${i}`);
    }

    // Create a per-image progress wrapper that reports cumulative progress
    const imageOffset = completedChunks;
    const imageProgress = onProgress
      ? (completed: number, _imageTotal: number) => {
        onProgress(imageOffset + completed, totalChunks);
      }
      : null;

    await writeImage(
      rootGroup as zarr.Group<MemoryStore>,
      image,
      dataset.path,
      imageProgress,
    );

    // Update cumulative offset for next image
    if (onProgress) {
      const shape = image.data.shape;
      const chunks = getChunksFromImage(image);
      let imageChunkCount = 1;
      for (let d = 0; d < shape.length; d++) {
        imageChunkCount *= Math.ceil(shape[d] / chunks[d]);
      }
      completedChunks += imageChunkCount;
    }
  }
}
