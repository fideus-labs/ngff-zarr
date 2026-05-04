// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Shared internal utilities for RFC-9 (.ozx) writing
 * Used by both Node and browser implementations to avoid code duplication
 */

import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import type { NgffImage } from "../types/ngff_image.ts";
import { isRfc4Enabled } from "../types/rfc4.ts";
import type { Axis } from "../types/zarr_metadata.ts";
import type { MemoryStore } from "./rfc9_zip.ts";

/**
 * Process axes for RFC enablement.
 * If RFC 4 is enabled, include orientation metadata.
 * If RFC 4 is not enabled, strip orientation from axes.
 */
export function processAxesForRfcs(
  axes: Axis[],
  enabledRfcs?: number[],
): Record<string, unknown>[] {
  const rfc4Enabled = isRfc4Enabled(enabledRfcs);

  return axes.map((axis) => {
    const result: Record<string, unknown> = {
      name: axis.name,
      type: axis.type,
    };

    // Include unit if present
    if (axis.unit !== undefined) {
      result.unit = axis.unit;
    }

    // Include orientation only if RFC 4 is enabled and orientation exists
    if (rfc4Enabled && axis.orientation) {
      result.orientation = {
        type: axis.orientation.type,
        value: axis.orientation.value,
      };
    }

    return result;
  });
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
 * This includes processing axes for RFC enablement and setting version to 0.5.
 *
 * @param multiscales - Source multiscales data
 * @param enabledRfcs - Optional list of RFC numbers to enable
 * @returns Prepared metadata object
 */
export function prepareRfc9Metadata(
  multiscales: NgffMultiscales,
  enabledRfcs?: number[],
): Record<string, unknown> {
  const _version = "0.5";

  // Process axes - filter orientation based on enabledRfcs
  const processedAxes = processAxesForRfcs(
    multiscales.metadata.axes,
    enabledRfcs,
  );

  return {
    version: _version,
    name: multiscales.metadata.name,
    axes: processedAxes,
    datasets: multiscales.metadata.datasets,
    ...(multiscales.metadata.coordinateTransformations && {
      coordinateTransformations: multiscales.metadata.coordinateTransformations,
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
 * @param enabledRfcs - Optional list of RFC numbers to enable
 * @param writeImage - Function to write individual images
 * @param onProgress - Optional progress callback for cumulative chunk
 *   progress across all scale levels
 */
export async function writeNgffMultiscalesToMemoryStore(
  store: MemoryStore,
  multiscales: NgffMultiscales,
  enabledRfcs: number[] | undefined,
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
  const multiscalesMetadata = prepareRfc9Metadata(multiscales, enabledRfcs);

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
