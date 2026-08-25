// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Functions for parsing Metadata from zarr store attributes.
 */

import * as zarr from "zarrita";
import type {
  Axis,
  CoordinateSystem,
  Dataset,
  FromZarrAttrsResult,
  MetadataInterface,
  Omero,
  Scale,
  Transform,
  Translation,
  V06Transform,
} from "../types/zarr_metadata.ts";
import { SUPPORTED_DIMS } from "../types/zarr_metadata.ts";
import { extractScaleTranslation, parseV06Transforms } from "./v06_metadata.ts";
import { NgffImage } from "../types/ngff_image.ts";
import type { AxesType, AxisUnit, SupportedDims } from "../types/units.ts";
import { parseOmero } from "./parse_metadata.ts";
import type { MemoryStore } from "../io/from_ngff_zarr.ts";
import {
  hasRfc4OrientationMetadata,
  validateRfc4Orientation,
} from "./rfc4_validation.ts";
import {
  validateStructural,
  ValidationLevel,
} from "./structural_validation.ts";
import type { AnatomicalOrientation } from "../types/rfc4.ts";
import { AnatomicalOrientationValues } from "../types/rfc4.ts";

/**
 * Map string orientation values to enum values
 */
const ORIENTATION_VALUE_MAP: Record<string, AnatomicalOrientationValues> = {
  "left-to-right": AnatomicalOrientationValues.LeftToRight,
  "right-to-left": AnatomicalOrientationValues.RightToLeft,
  "anterior-to-posterior": AnatomicalOrientationValues.AnteriorToPosterior,
  "posterior-to-anterior": AnatomicalOrientationValues.PosteriorToAnterior,
  "inferior-to-superior": AnatomicalOrientationValues.InferiorToSuperior,
  "superior-to-inferior": AnatomicalOrientationValues.SuperiorToInferior,
  "dorsal-to-ventral": AnatomicalOrientationValues.DorsalToVentral,
  "ventral-to-dorsal": AnatomicalOrientationValues.VentralToDorsal,
  "dorsal-to-palmar": AnatomicalOrientationValues.DorsalToPalmar,
  "palmar-to-dorsal": AnatomicalOrientationValues.PalmarToDorsal,
  "dorsal-to-plantar": AnatomicalOrientationValues.DorsalToPlantar,
  "plantar-to-dorsal": AnatomicalOrientationValues.PlantarToDorsal,
  "rostral-to-caudal": AnatomicalOrientationValues.RostralToCaudal,
  "caudal-to-rostral": AnatomicalOrientationValues.CaudalToRostral,
  "cranial-to-caudal": AnatomicalOrientationValues.CranialToCaudal,
  "caudal-to-cranial": AnatomicalOrientationValues.CaudalToCranial,
  "proximal-to-distal": AnatomicalOrientationValues.ProximalToDistal,
  "distal-to-proximal": AnatomicalOrientationValues.DistalToProximal,
  "superficial-to-deep": AnatomicalOrientationValues.SuperficialToDeep,
  "deep-to-superficial": AnatomicalOrientationValues.DeepToSuperficial,
  "apical-to-basal": AnatomicalOrientationValues.ApicalToBasal,
  "basal-to-apical": AnatomicalOrientationValues.BasalToApical,
  "apex-to-base": AnatomicalOrientationValues.ApexToBase,
  "base-to-apex": AnatomicalOrientationValues.BaseToApex,
};

/**
 * Extract anatomical orientations from axes metadata.
 * Returns a record mapping axis names to their orientations, or undefined if no orientations.
 */
function extractOrientationsFromAxes(
  axesData: Array<Record<string, unknown> | string>,
): Record<string, AnatomicalOrientation> | undefined {
  const orientations: Record<string, AnatomicalOrientation> = {};
  let hasOrientation = false;

  for (const axis of axesData) {
    if (typeof axis === "object" && axis !== null) {
      const axisObj = axis as Record<string, unknown>;
      const name = axisObj.name as string;
      const orientation = axisObj.orientation as
        | { type: string; value: string }
        | undefined;

      if (
        orientation &&
        typeof orientation === "object" &&
        orientation.type === "anatomical"
      ) {
        const enumValue = ORIENTATION_VALUE_MAP[orientation.value];
        if (enumValue) {
          orientations[name] = {
            type: "anatomical" as const,
            value: enumValue,
          };
          hasOrientation = true;
        }
      }
    }
  }

  return hasOrientation ? orientations : undefined;
}

/**
 * Return whether an OME-Zarr spec version string is `0.4` or newer.
 *
 * The dotted version is compared component-wise as integers, so multi-digit
 * minor versions order correctly: `"0.10"` is newer than `"0.4"`, not older.
 * A plain `Number.parseFloat("0.10")` would yield `0.1` and wrongly drop v0.10+
 * metadata from structural validation. This mirrors the Python port's
 * `packaging.version.parse(version) >= packaging.version.parse("0.4")` gate.
 *
 * Returns `false` for an unparsable version (leading component not a number),
 * matching the prior `Number.isNaN` guard that skipped validation in that case.
 */
function isSpecVersionAtLeastV04(version: string): boolean {
  const [major, minor = 0] = version
    .split(".")
    .map((part) => Number.parseInt(part, 10));
  if (Number.isNaN(major)) {
    return false;
  }
  // A NaN minor (e.g. "0.x") fails `minor >= 4`, so it needs no separate guard.
  return major > 0 || (major === 0 && minor >= 4);
}

/**
 * Recognized keys of a single `multiscales[]` entry. Any other key present in an
 * entry is a leak -- a group-level `ome` / `multiscales` wrapper, a
 * `zarr_format` byte that belongs on the enclosing Zarr v3 group, and so on --
 * and is routed to `Metadata.extra` for the v0.5 namespacing rules to inspect.
 * `omero` is absent on purpose: it is a sibling of `multiscales`, parsed
 * separately, never a field of the entry. `@type` *is* recognized: the Python
 * writer stamps the JSON-LD `@type` (`"ngff:Image"`) onto every entry, so it is
 * a legitimate library-written key, not a leak.
 */
const MULTISCALE_ENTRY_FIELDS = new Set<string>([
  "@type",
  "version",
  "name",
  "axes",
  "datasets",
  "coordinateTransformations",
  "type",
  "metadata",
]);

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
  let units: Record<string, AxisUnit | undefined>;

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
          // RFC-3 permits any axis name, so the parsed value stays a string
          // rather than being narrowed to the closed `SupportedDims` union.
          name: String(axisObj.name),
          // `type` is optional; keep undefined rather than the "undefined" string
          type: axisObj.type === undefined || axisObj.type === null
            ? undefined
            : String(axisObj.type) as AxesType,
          unit: axisObj.unit as AxisUnit | undefined,
          // Preserve RFC 4 orientation on the parsed axis so the strict
          // structural pass below can run validateAxisOrientation; dropping it
          // here makes that rule a no-op. This mirrors the Python port, whose
          // _filter_axis_dict keeps the raw orientation field on each Axis.
          ...(axisObj.orientation !== undefined && axisObj.orientation !== null
            ? { orientation: axisObj.orientation as Axis["orientation"] }
            : {}),
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
        const unit = axisObj.unit as AxisUnit | undefined;
        if (name !== undefined && unit !== undefined) {
          units[name] = unit;
        }
      }
    }
  }

  // Extract anatomical orientations from axes (RFC 4)
  let axesOrientations: Record<string, AnatomicalOrientation> | undefined;
  if (
    "axes" in multiscalesMetadata && Array.isArray(multiscalesMetadata.axes)
  ) {
    const axesData = multiscalesMetadata.axes as Array<
      Record<string, unknown> | string
    >;
    axesOrientations = extractOrientationsFromAxes(axesData);
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

    const filteredUnits: Record<string, AxisUnit> = {};
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
      axesOrientations,
      computedCallbacks: undefined,
    });

    images.push(ngffImage);
  }

  // Keys that a malformed or double-namespaced document leaked into the
  // multiscale entry -- e.g. a group-level `ome` / `multiscales` wrapper, or a
  // `zarr_format` byte that belongs on the enclosing Zarr v3 group, never on the
  // entry. Captured verbatim so the v0.5 namespacing rules can flag them.
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(multiscalesMetadata)) {
    if (!MULTISCALE_ENTRY_FIELDS.has(key)) {
      extra[key] = value;
    }
  }

  // Build the metadata object - use spread to only include optional properties if defined
  const metadata: MetadataInterface = {
    axes,
    datasets,
    name: (multiscalesMetadata.name as string) ?? "image",
    // OME-Zarr v0.5 hoists the spec `version` to the group-level `ome`
    // namespace; v0.4 carries it on each multiscale entry. Prefer the
    // group-level value (which fromZarrAttrsV05 forwards as a top-level
    // `version`) so the structural pass sees the true spec version -- which the
    // v0.5 namespacing rules gate on. v0.4 has no top-level version, so this
    // falls back to the entry's.
    version: (rootAttrs.version as string | undefined) ??
      (multiscalesMetadata.version as string) ?? "0.4",
    omero,
    extra,
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

  // Strict structural validation of the parsed metadata, layered after the
  // schema pass and RFC 4 dict check above. The structural rules encode the
  // v0.4+ metadata model (typed axes and per-dataset coordinate
  // transformations); the legacy v0.1-0.3 layouts predate it, so the rules are
  // scoped to v0.4 and newer.
  if (validate) {
    if (isSpecVersionAtLeastV04(metadata.version)) {
      validateStructural(
        metadata,
        { level: ValidationLevel.Strict },
        metadata.version,
      );
    }
  }

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
    // Standard v0.5 format with "ome" wrapper. Forward the hoisted
    // `ome.version` as a top-level `version` so fromZarrAttrsV04 records the
    // true spec version (v0.5), which the v0.5 namespacing rules gate on. This
    // entry point definitionally handles v0.5, so default to "0.5" when
    // `ome.version` is absent -- otherwise a malformed v0.5 store missing its
    // version would fall back to "0.4" and skip the v0.5 namespacing rules
    // during the structural pass (which runs before the version is corrected
    // below).
    v04Attrs = {
      version: omeData.version ?? "0.5",
      multiscales: omeData.multiscales,
      omero: omeData.omero ?? rootAttrs.omero,
    };
  } else if ("multiscales" in rootAttrs) {
    // Compatibility mode: multiscales at root level (as written by toNgffZarr).
    // Same v0.5 floor as the wrapped branch above.
    v04Attrs = {
      version: rootAttrs.version ?? "0.5",
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

/**
 * Parse Metadata and NgffImages from OME-Zarr v0.6 (RFC 5) root attributes.
 *
 * Unlike v0.4/v0.5, a v0.6 multiscale entry carries `coordinateSystems` instead
 * of a top-level `axes` list, and maps each dataset's array index space into the
 * implicit "intrinsic" system through a `sequence` of scale + translation. This
 * reader recovers the version-agnostic in-memory model (axes + per-dataset scale
 * and translation) while preserving the coordinate systems and any top-level
 * transformations for round-tripping. Mirrors the Python
 * `v06.zarr_metadata.Metadata._from_zarr_attrs`.
 */
export async function fromZarrAttrsV06(
  rootAttrs: Record<string, unknown>,
  store: MemoryStore | zarr.FetchStore | zarr.Readable,
  validate = false,
): Promise<FromZarrAttrsResult> {
  // v0.6 wraps the multiscales under the "ome" key; tolerate a root-level
  // layout for symmetry with the v0.5 reader.
  const omeData = rootAttrs.ome as Record<string, unknown> | undefined;
  // v0.6 and 0.9.dev1 share this reader; the group-level `version` says which.
  const declaredVersion = (omeData?.version as string | undefined) ??
    (rootAttrs.version as string | undefined) ?? "0.6";
  let entry: Record<string, unknown>;
  let omeroRaw: unknown;
  if (omeData && "multiscales" in omeData) {
    const multiscales = omeData.multiscales as unknown[];
    if (!Array.isArray(multiscales) || multiscales.length === 0) {
      throw new Error("No multiscales metadata found in root attributes");
    }
    entry = multiscales[0] as Record<string, unknown>;
    omeroRaw = omeData.omero ?? rootAttrs.omero;
  } else if ("multiscales" in rootAttrs) {
    const multiscales = rootAttrs.multiscales as unknown[];
    if (!Array.isArray(multiscales) || multiscales.length === 0) {
      throw new Error("No multiscales metadata found in root attributes");
    }
    entry = multiscales[0] as Record<string, unknown>;
    omeroRaw = rootAttrs.omero;
  } else {
    throw new Error(
      "No multiscales metadata found in root attributes for v0.6 format",
    );
  }

  const coordinateSystemsRaw = entry.coordinateSystems as
    | Array<Record<string, unknown>>
    | undefined;
  if (
    !Array.isArray(coordinateSystemsRaw) || coordinateSystemsRaw.length === 0
  ) {
    throw new Error(
      "Invalid OME-Zarr v0.6 metadata: missing 'coordinateSystems' in " +
        "multiscales metadata.",
    );
  }

  const coordinateSystems: CoordinateSystem[] = coordinateSystemsRaw.map(
    (cs, csIndex) => {
      // Guard the untrusted-input boundary: a missing/non-array `axes` would
      // otherwise surface as a generic `.map` TypeError rather than a clear
      // metadata error. (Per-axis name/type are still coerced, mirroring the
      // v0.4 reader's tolerance.)
      if (!Array.isArray(cs.axes)) {
        throw new Error(
          "Invalid OME-Zarr v0.6 metadata: " +
            `coordinateSystems[${csIndex}].axes must be an array.`,
        );
      }
      return {
        name: String(cs.name),
        axes: (cs.axes as Array<Record<string, unknown>>).map((axis) => ({
          // RFC-3 permits any axis name, so the parsed value stays a string
          // rather than being narrowed to the closed `SupportedDims` union.
          name: String(axis.name),
          // `type` is optional; keep undefined rather than the "undefined" string
          type: axis.type === undefined || axis.type === null
            ? undefined
            : String(axis.type) as AxesType,
          unit: axis.unit as AxisUnit | undefined,
          ...(axis.orientation !== undefined && axis.orientation !== null
            ? { orientation: axis.orientation as Axis["orientation"] }
            : {}),
          ...(typeof axis.discrete === "boolean"
            ? { discrete: axis.discrete }
            : {}),
        })),
      };
    },
  );
  const coordinateSystemNames = coordinateSystems.map((cs) => cs.name);

  // The intrinsic coordinate system (the first one) backs the multiscale image
  // dimensions, units, and anatomical orientations.
  const intrinsic = coordinateSystems[0];
  const intrinsicRawAxes = coordinateSystemsRaw[0].axes as Array<
    Record<string, unknown> | string
  >;
  const dims = intrinsic.axes.map((axis) => String(axis.name));

  const units: Record<string, AxisUnit> = {};
  for (const axis of intrinsic.axes) {
    if (axis.unit !== undefined && axis.unit !== null) {
      units[axis.name] = axis.unit;
    }
  }
  const axesOrientations = extractOrientationsFromAxes(intrinsicRawAxes);

  // RFC 4 anatomical-orientation validation, mirroring the v0.4 reader.
  if (validate) {
    const axesDicts = intrinsicRawAxes.filter(
      (axis): axis is Record<string, unknown> =>
        typeof axis === "object" && axis !== null,
    );
    if (axesDicts.length > 0 && hasRfc4OrientationMetadata(axesDicts)) {
      validateRfc4Orientation(axesDicts);
    }
  }

  // Open root group for array access (reuse consolidated metadata if present).
  let optimizedStore: MemoryStore | zarr.FetchStore | zarr.Readable;
  try {
    const tryWithConsolidated: ((s: unknown) => Promise<unknown>) | undefined =
      (zarr as unknown as {
        tryWithConsolidated?: (s: unknown) => Promise<unknown>;
      }).tryWithConsolidated;
    optimizedStore = tryWithConsolidated
      ? (await tryWithConsolidated(store)) as
        | MemoryStore
        | zarr.FetchStore
        | zarr.Readable
      : store;
  } catch {
    optimizedStore = store;
  }
  const root = await zarr.open(optimizedStore as zarr.Readable, {
    kind: "group",
  });

  const datasetsData = entry.datasets as Array<Record<string, unknown>>;
  if (!Array.isArray(datasetsData) || datasetsData.length === 0) {
    throw new Error(
      "Invalid OME-Zarr metadata: 'datasets' must be a non-empty array",
    );
  }

  const images: NgffImage[] = [];
  const datasets: Dataset[] = [];

  for (const dataset of datasetsData) {
    const path = String(dataset.path);
    const zarrArray = (await zarr.open(root.resolve(path), {
      kind: "array",
    })) as zarr.Array<zarr.DataType, zarr.Readable>;

    let scaleValues = dims.map(() => 1.0);
    let translationValues = dims.map(() => 0.0);
    if ("coordinateTransformations" in dataset) {
      if (!Array.isArray(dataset.coordinateTransformations)) {
        throw new Error(
          "Invalid OME-Zarr v0.6 metadata: " +
            `dataset '${path}' coordinateTransformations must be an array.`,
        );
      }
      const parsed = parseV06Transforms(
        dataset.coordinateTransformations as Array<Record<string, unknown>>,
        coordinateSystemNames,
        coordinateSystems,
      );
      const extracted = extractScaleTranslation(parsed, dims);
      scaleValues = extracted.scale;
      translationValues = extracted.translation;
    }

    const scale: Record<string, number> = {};
    const translation: Record<string, number> = {};
    dims.forEach((dim, i) => {
      scale[dim] = i < scaleValues.length ? scaleValues[i] : 1.0;
      translation[dim] = i < translationValues.length
        ? translationValues[i]
        : 0.0;
    });

    datasets.push({
      path,
      coordinateTransformations: [
        { type: "scale", scale: scaleValues } as Scale,
        { type: "translation", translation: translationValues } as Translation,
      ],
    });

    images.push(
      new NgffImage({
        data: zarrArray,
        dims,
        scale,
        translation,
        name: (entry.name as string) ?? "image",
        axesUnits: Object.keys(units).length > 0 ? units : undefined,
        axesOrientations,
        computedCallbacks: undefined,
      }),
    );
  }

  let coordinateTransformations: V06Transform[] | undefined;
  if ("coordinateTransformations" in entry) {
    if (!Array.isArray(entry.coordinateTransformations)) {
      throw new Error(
        "Invalid OME-Zarr v0.6 metadata: multiscales " +
          "coordinateTransformations must be an array.",
      );
    }
    coordinateTransformations = parseV06Transforms(
      entry.coordinateTransformations as Array<Record<string, unknown>>,
      coordinateSystemNames,
      coordinateSystems,
    );
  }

  const omero: Omero | undefined = parseOmero(
    omeroRaw as Record<string, unknown> | undefined,
  );

  const metadata: MetadataInterface = {
    axes: intrinsic.axes,
    datasets,
    coordinateSystems,
    coordinateTransformations,
    name: (entry.name as string) ?? "image",
    version: "0.6",
    omero,
    ...(entry.type !== undefined && entry.type !== null
      ? { type: String(entry.type) }
      : {}),
    ...(entry.metadata !== undefined && entry.metadata !== null
      ? {
        metadata: entry.metadata as {
          description: string;
          method: string;
          version: string;
        },
      }
      : {}),
  };

  // Run the structural pass against the normalized metadata (axes + per-dataset
  // scale/translation), matching the v0.4/v0.5 readers. The metadata is
  // reconstructed into the same shape, so the same rules apply. Full RFC-5 wire
  // validation (coordinate-system graphs, arbitrary transform chains) remains
  // deferred.
  //
  // The declared version is what decides whether the axis rules apply: it
  // selects the RFC-3 axis model at 0.9.dev1 and the `array` coordinate-system
  // arm of the v0.6 axes schema. Validating without it enforces the v0.4 caps
  // on every store, so a document this port's own writer emits is refused on
  // the way back in.
  if (validate) {
    validateStructural(
      metadata,
      { level: ValidationLevel.Strict },
      declaredVersion,
    );
  }

  return { metadata, images };
}
