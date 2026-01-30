// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Functions for parsing Metadata from zarr store attributes.
 */

import * as zarr from "zarrita";
import type {
  Axis,
  Dataset,
  FromZarrAttrsResult,
  MetadataInterface,
  Omero,
  Scale,
  Transform,
  Translation,
} from "../types/zarr_metadata.ts";
import { SUPPORTED_DIMS } from "../types/zarr_metadata.ts";
import { NgffImage } from "../types/ngff_image.ts";
import type { AxesType, SupportedDims, Units } from "../types/units.ts";
import { parseOmero } from "./parse_metadata.ts";
import type { MemoryStore } from "../io/from_ngff_zarr.ts";
import {
  hasRfc4OrientationMetadata,
  validateRfc4Orientation,
} from "./rfc4_validation.ts";

/**
 * Parse Metadata and NgffImages from OME-Zarr v0.4 root attributes.
 *
 * This mirrors the Python `Metadata._from_zarr_attrs` class method.
 */
export async function fromZarrAttrsV04(
  rootAttrs: Record<string, unknown>,
  store: MemoryStore | zarr.FetchStore | zarr.Readable,
  validate = false,
): Promise<FromZarrAttrsResult> {
  // Extract the multiscales metadata
  const multiscalesArray = rootAttrs.multiscales as unknown[];
  if (!Array.isArray(multiscalesArray) || multiscalesArray.length === 0) {
    throw new Error("No multiscales metadata found in root attributes");
  }

  const multiscalesMetadata = multiscalesArray[0] as Record<string, unknown>;

  // Validate the root attributes against the OME-Zarr v0.4 schema
  if (validate) {
    // Basic structural validation
    if (!("multiscales" in rootAttrs)) {
      throw new Error("Invalid OME-Zarr metadata: missing 'multiscales' key");
    }
    if (
      !Array.isArray(rootAttrs.multiscales) ||
      rootAttrs.multiscales.length === 0
    ) {
      throw new Error(
        "Invalid OME-Zarr metadata: 'multiscales' must be a non-empty array",
      );
    }

    // Validate datasets exist
    const datasets = multiscalesMetadata.datasets;
    if (!Array.isArray(datasets) || datasets.length === 0) {
      throw new Error(
        "Invalid OME-Zarr metadata: 'datasets' must be a non-empty array",
      );
    }

    // RFC 4 validation for anatomical orientation
    if (
      "axes" in multiscalesMetadata &&
      Array.isArray(multiscalesMetadata.axes)
    ) {
      const axesData = multiscalesMetadata.axes as Array<
        Record<string, unknown>
      >;
      // Filter to only dict-style axes for RFC4 validation
      const axesDicts = axesData.filter(
        (axis): axis is Record<string, unknown> =>
          typeof axis === "object" && axis !== null,
      );
      if (axesDicts.length > 0 && hasRfc4OrientationMetadata(axesDicts)) {
        validateRfc4Orientation(axesDicts);
      }
    }
  }

  // Parse OMERO metadata
  const omero: Omero | undefined = parseOmero(
    rootAttrs.omero as Record<string, unknown> | undefined,
  );

  // Handle backwards compatibility for version <= 0.3
  let dims: string[];
  let axes: Axis[];
  let units: Record<string, Units | undefined>;

  if (!("axes" in multiscalesMetadata)) {
    // Version <= 0.3 - use default dims
    dims = [...SUPPORTED_DIMS].reverse();
    axes = [
      { name: "t" as SupportedDims, type: "time" as AxesType, unit: undefined },
      {
        name: "c" as SupportedDims,
        type: "channel" as AxesType,
        unit: undefined,
      },
      {
        name: "z" as SupportedDims,
        type: "space" as AxesType,
        unit: undefined,
      },
      {
        name: "y" as SupportedDims,
        type: "space" as AxesType,
        unit: undefined,
      },
      {
        name: "x" as SupportedDims,
        type: "space" as AxesType,
        unit: undefined,
      },
    ];
    units = Object.fromEntries(dims.map((d) => [d, undefined]));
  } else {
    const axesData = multiscalesMetadata.axes as Array<
      Record<string, unknown> | string
    >;
    dims = axesData.map((a) =>
      typeof a === "object" && "name" in a ? String(a.name) : String(a)
    );

    // Check if axes have names (v0.4+) or are just strings (v0.3)
    if (typeof axesData[0] === "object" && "name" in axesData[0]) {
      axes = axesData.map((axis) => {
        const axisObj = axis as Record<string, unknown>;
        return {
          name: String(axisObj.name) as SupportedDims,
          type: String(axisObj.type) as AxesType,
          unit: axisObj.unit as Units | undefined,
        };
      });
    } else {
      // v0.3 - string axes
      const typeDict: Record<string, AxesType> = {
        t: "time",
        c: "channel",
        z: "space",
        y: "space",
        x: "space",
      };
      axes = axesData.map((axis) => {
        const name = String(axis) as SupportedDims;
        return {
          name,
          type: typeDict[name] ?? "space",
          unit: undefined,
        };
      });
    }

    // Extract units from axes
    units = Object.fromEntries(dims.map((d) => [d, undefined]));
    for (const axis of axesData) {
      if (typeof axis === "object") {
        const axisObj = axis as Record<string, unknown>;
        const name = axisObj.name as string;
        const unit = axisObj.unit as Units | undefined;
        if (name !== undefined && unit !== undefined) {
          units[name] = unit;
        }
      }
    }
  }

  // Parse datasets and create NgffImages
  const images: NgffImage[] = [];
  const datasets: Dataset[] = [];

  const datasetsData = multiscalesMetadata.datasets as Array<
    Record<string, unknown>
  >;

  // Open root group for array access
  let optimizedStore: MemoryStore | zarr.FetchStore | zarr.Readable;
  try {
    const tryWithConsolidated: ((s: unknown) => Promise<unknown>) | undefined =
      (zarr as unknown as {
        tryWithConsolidated?: (s: unknown) => Promise<unknown>;
      })
        .tryWithConsolidated;
    if (tryWithConsolidated) {
      optimizedStore = (await tryWithConsolidated(store)) as
        | MemoryStore
        | zarr.FetchStore
        | zarr.Readable;
    } else {
      optimizedStore = store;
    }
  } catch {
    optimizedStore = store;
  }

  const root = await zarr.open(optimizedStore as zarr.Readable, {
    kind: "group",
  });

  for (const dataset of datasetsData) {
    const path = String(dataset.path);

    // Open the zarr array
    const zarrArray = (await zarr.open(root.resolve(path), {
      kind: "array",
    })) as zarr.Array<zarr.DataType, zarr.Readable>;

    // Parse coordinate transformations
    const scale: Record<string, number> = Object.fromEntries(
      dims.map((d) => [d, 1.0]),
    );
    const translation: Record<string, number> = Object.fromEntries(
      dims.map((d) => [d, 0.0]),
    );
    const coordinateTransformations: Transform[] = [];

    if ("coordinateTransformations" in dataset) {
      const transforms = dataset.coordinateTransformations as Array<
        Record<string, unknown>
      >;
      for (const transformation of transforms) {
        if ("scale" in transformation) {
          const scaleValues = transformation.scale as number[];
          dims.forEach((dim, i) => {
            if (i < scaleValues.length) {
              scale[dim] = scaleValues[i];
            }
          });
          coordinateTransformations.push({
            type: "scale",
            scale: scaleValues,
          } as Scale);
        } else if ("translation" in transformation) {
          const translationValues = transformation.translation as number[];
          dims.forEach((dim, i) => {
            if (i < translationValues.length) {
              translation[dim] = translationValues[i];
            }
          });
          coordinateTransformations.push({
            type: "translation",
            translation: translationValues,
          } as Translation);
        }
      }
    }

    datasets.push({
      path,
      coordinateTransformations,
    });

    const filteredUnits: Record<string, Units> = {};
    for (const [axis, unit] of Object.entries(units)) {
      if (unit !== undefined && unit !== null) {
        filteredUnits[axis] = unit;
      }
    }

    const ngffImage = new NgffImage({
      data: zarrArray,
      dims,
      scale,
      translation,
      name: (multiscalesMetadata.name as string) ?? "image",
      axesUnits: Object.keys(filteredUnits).length > 0
        ? filteredUnits
        : undefined,
      computedCallbacks: undefined,
    });

    images.push(ngffImage);
  }

  // Build the metadata object - use spread to only include optional properties if defined
  const metadata: MetadataInterface = {
    axes,
    datasets,
    name: (multiscalesMetadata.name as string) ?? "image",
    version: (multiscalesMetadata.version as string) ?? "0.4",
    omero,
    coordinateTransformations:
      (multiscalesMetadata.coordinateTransformations as Transform[]) ??
        undefined,
    ...(multiscalesMetadata.type !== undefined &&
        multiscalesMetadata.type !== null
      ? { type: String(multiscalesMetadata.type) }
      : {}),
    ...(multiscalesMetadata.metadata !== undefined &&
        multiscalesMetadata.metadata !== null
      ? {
        metadata: multiscalesMetadata.metadata as {
          description: string;
          method: string;
          version: string;
        },
      }
      : {}),
  };

  return { metadata, images };
}

/**
 * Parse Metadata and NgffImages from OME-Zarr v0.5 root attributes.
 *
 * The v0.5 format typically wraps multiscales under the "ome" key,
 * but for compatibility we also handle the case where multiscales
 * is at the root level (as written by toNgffZarr).
 */
export async function fromZarrAttrsV05(
  rootAttrs: Record<string, unknown>,
  store: MemoryStore | zarr.FetchStore | zarr.Readable,
  validate = false,
): Promise<FromZarrAttrsResult> {
  // v0.5 may wrap everything under "ome" key, or may have multiscales at root
  const omeData = rootAttrs.ome as Record<string, unknown> | undefined;

  let v04Attrs: Record<string, unknown>;
  if (omeData && "multiscales" in omeData) {
    // Standard v0.5 format with "ome" wrapper
    v04Attrs = {
      multiscales: omeData.multiscales,
      omero: omeData.omero ?? rootAttrs.omero,
    };
  } else if ("multiscales" in rootAttrs) {
    // Compatibility mode: multiscales at root level (as written by toNgffZarr)
    v04Attrs = {
      multiscales: rootAttrs.multiscales,
      omero: rootAttrs.omero,
    };
  } else {
    throw new Error(
      "No multiscales metadata found in root attributes for v0.5 format",
    );
  }

  const result = await fromZarrAttrsV04(v04Attrs, store, validate);

  // Update version to 0.5
  result.metadata.version = "0.5";

  return result;
}
