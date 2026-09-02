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
import type { MemoryStore } from "./rfc9_zip.ts";
import {
  buildV06MultiscalesEntry,
  legacyTopLevelTransforms,
} from "../utils/v06_metadata.ts";
import { multiscaleOmeDict } from "../utils/rfc8_multiscale.ts";
import {
  type ImageVersion,
  NgffVersion,
  V06_ONDISK_VERSION,
} from "../types/supported_versions.ts";
import {
  SpecRule,
  validateAxisCount,
  validateAxisNamesUnique,
  validateAxisOrder,
  validateAxisType,
  validateSpatialAxisOrder,
  ValidationError,
} from "../utils/structural_validation.ts";
import { pyRepr, pyReprOptional } from "../utils/py_format.ts";

/**
 * Every axis list `metadata` will serialize at `version`, paired with its
 * location.
 *
 * Only the v0.6 family writes `coordinateSystems`, and
 * `coordinate_systems.schema` applies `axes.schema` to each one, so every
 * system is returned there. A v0.4/v0.5 target writes a single flat `axes` and
 * drops the systems, so that is the only list to gate: refusing a system the
 * downgrade discards would reject a document the writer can emit, and name a
 * node absent from it. Python reaches the same set by gating after
 * `Metadata.to_version`.
 */
function axisViews(
  metadata: MetadataInterface,
  version: string,
): Array<{ location: string; axes: Axis[] }> {
  const systems = metadata.coordinateSystems ?? [];
  const serializesSystems = version === "0.6" ||
    version === NgffVersion.V09dev1 ||
    version === NgffVersion.V09dev3;
  if (serializesSystems && systems.length > 0) {
    // `buildV06MultiscalesEntry` serializes the first system from
    // `metadata.axes` and the later ones verbatim, and `MetadataInterface`
    // does not tie `axes` to `coordinateSystems[0].axes`. Gate what is
    // written, not what is declared.
    return systems.map((system, index) => ({
      location: `multiscales[0].coordinateSystems[${index}].axes`,
      axes: index === 0 ? metadata.axes : system.axes,
    }));
  }
  return [{ location: "multiscales[0].axes", axes: metadata.axes }];
}

/**
 * Refuse to serialize an axis model the target `version` cannot express.
 *
 * Reuses the axis rules of the structural pass, but this is a second
 * dispatcher, not the same pass as {@link validateStructural}, and it differs
 * from it two ways:
 *
 * 1. It runs only the five axis rules, not the full image cascade.
 * 2. Where the target version serializes coordinate systems, it checks *every*
 *    one of them (see {@link axisViews}), while the structural pass reduces the
 *    metadata to the intrinsic system's axes.
 */
function gateAxisModel(
  metadata: MetadataInterface,
  version: string,
): void {
  const rules = [
    validateAxisCount,
    validateAxisType,
    validateAxisOrder,
    validateSpatialAxisOrder,
    validateAxisNamesUnique,
  ];
  for (const view of axisViews(metadata, version)) {
    for (const rule of rules) {
      try {
        rule({ axes: view.axes }, version);
      } catch (error) {
        if (!(error instanceof ValidationError)) {
          throw error;
        }
        const rendered = view.axes
          .map((ax) => `${pyRepr(ax.name)}(type=${pyReprOptional(ax.type)})`)
          .join(", ");
        if (error.rule === SpecRule.AxisNamesUnique) {
          // Required at every version, 0.9.dev1 included.
          throw new Error(
            `Cannot write OME-Zarr version="${version}": ${error.detail} ` +
              `Axes at ${view.location}: [${rendered}].`,
          );
        }
        throw new Error(
          `Cannot write OME-Zarr version="${version}": this axis model violates ` +
            `that version's [${error.rule}] rule. ${error.detail} ` +
            `Axes at ${view.location}: [${rendered}]. ` +
            `Pass version="${NgffVersion.V09dev1}" or ` +
            `version="${NgffVersion.V09dev3}" to write it: the OME-Zarr 0.9 ` +
            `development series adopts RFC-3 (arbitrary axis count, names, ` +
            `types and ordering).`,
        );
      }
    }
  }
}

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
  version: ImageVersion,
): Record<string, unknown> {
  gateAxisModel(metadata, version);

  // Process axes (orientation included when present).
  const processedAxes = processAxes(metadata.axes);

  if (version === NgffVersion.V09dev3) {
    // RFC-8: the root ome value is a multiscale node document; the datasets
    // become singlescale child nodes and omero rides in the node's
    // attributes.
    const v09Entry = buildV06MultiscalesEntry(metadata, processedAxes);
    return {
      ome: multiscaleOmeDict(v09Entry, NgffVersion.V09dev3, metadata.omero),
    };
  }

  if (version === "0.9.dev1") {
    // "0.9.dev1" is already the on-disk string.
    const v09Entry = buildV06MultiscalesEntry(metadata, processedAxes);
    return {
      ome: {
        version: NgffVersion.V09dev1,
        multiscales: [v09Entry],
        ...(metadata.omero && { omero: metadata.omero }),
      },
    };
  }

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
