// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Declare a field image as an RFC-5 `displacements` or `coordinates`
 * transform.
 *
 * A field store that only types its component axis is labelled, not
 * applicable: a reader sees what the channels are, but no transformation says
 * between which coordinate systems the field maps, so nothing can apply it.
 * The declaration is the `displacements` (or `coordinates`) entry on the
 * multiscales' `coordinateTransformations`; building it by hand means reaching
 * into the version metadata model and validating axis order, component count
 * and system references oneself. This function is that composition, validated,
 * in one call.
 *
 * Two layouts use it. A standalone field store -- the artifact a registration
 * emits -- declares the transform on its own multiscales, mapping a spatial
 * coordinate system onto itself through its own level-0 array; the default
 * arguments produce exactly that. A field written beside the image it
 * displaces declares the transform on the image's multiscales instead, with
 * `path` naming the field's array and `inputSystem`/`outputSystem`
 * referencing the image's own coordinate systems.
 *
 * The port of `declare_field_transform` in the Python package.
 */

import { NgffMultiscales } from "../types/multiscales.ts";
import type {
  Axis,
  Coordinates,
  CoordinateSystem,
  Displacements,
} from "../types/zarr_metadata.ts";

/** The axis type a transform of each kind calls for on the component axis. */
const COMPONENT_TYPES = {
  displacements: "displacement",
  coordinates: "coordinate",
} as const;

/** The axis type `toMultiscales` gives a dimension that declares none. */
const DEFAULT_AXIS_TYPES: Record<string, string> = {
  x: "space",
  y: "space",
  z: "space",
  c: "channel",
  t: "time",
};

export interface DeclareFieldTransformOptions {
  /** `displacements` (the field holds offsets) or `coordinates` (absolute output positions). */
  transformType?: keyof typeof COMPONENT_TYPES;
  /**
   * The zarr path of the field's array. Default: the multiscales' own finest
   * dataset -- the standalone store. When given, the field is elsewhere and
   * its shape cannot be checked here; the references still are.
   */
  path?: string;
  /**
   * Name of the declared coordinate system the transform maps from. Pass both
   * `inputSystem` and `outputSystem`, or neither: with neither, a spatial
   * system named `coordinateSystem` is declared from the field's own axes and
   * the transform maps it onto itself, which is the standalone total field.
   */
  inputSystem?: string;
  /** Name of the declared coordinate system the transform maps onto. See `inputSystem`. */
  outputSystem?: string;
  /** Name of the spatial system to declare when the two above are not given. */
  coordinateSystem?: string;
  /** How a reader interpolates the field between grid points. */
  interpolation?: string;
}

/**
 * Return `multiscales` with the field transform declared on it.
 *
 * Functional and additive: the argument is not mutated, and the entry is
 * appended to any `coordinateTransformations` already declared. The result
 * must be written at OME-Zarr 0.6 or later; earlier versions cannot carry
 * multiscale-level transformations.
 *
 * @throws If the field's axes are not one component axis of the type the
 * transform calls for followed by the spatial axes, if it holds a number of
 * components other than the number of spatial axes, or if a named system is
 * not declared.
 */
export function declareFieldTransform(
  multiscales: NgffMultiscales,
  options: DeclareFieldTransformOptions = {},
): NgffMultiscales {
  const {
    transformType = "displacements",
    path,
    inputSystem,
    outputSystem,
    coordinateSystem = "physical",
    interpolation = "linear",
  } = options;
  // `in` reaches the prototype chain, so "toString" would pass and then read
  // a function out of the table; a JS caller is not held to the type above.
  if (
    typeof transformType !== "string" ||
    !Object.hasOwn(COMPONENT_TYPES, transformType)
  ) {
    throw new Error(
      `transformType must be one of ${
        Object.keys(COMPONENT_TYPES).join(", ")
      }, got '${transformType}'`,
    );
  }
  const metadata = multiscales.metadata;
  let spatial: string[] = [];
  let entryPath = path;
  if (entryPath === undefined) {
    spatial = checkFieldImage(multiscales, COMPONENT_TYPES[transformType]);
    entryPath = metadata.datasets[0].path;
  } else if (inputSystem === undefined) {
    throw new Error(
      "with path, pass inputSystem and outputSystem: the field is elsewhere, " +
        "so there are no axes here to derive a system from",
    );
  }

  let systems = [...(metadata.coordinateSystems ?? [])];
  let input: string;
  let output: string;
  // Both or neither, as a narrowing rather than a guard on the side: the two
  // names are what the entry references, and TypeScript is told so here.
  if (inputSystem === undefined && outputSystem === undefined) {
    systems = withSpatialSystem(
      multiscales,
      systems,
      coordinateSystem,
      spatial,
    );
    input = output = coordinateSystem;
  } else if (inputSystem !== undefined && outputSystem !== undefined) {
    const declared = new Map(
      systems.map((system) => [system.name, system.axes.length]),
    );
    for (const name of [inputSystem, outputSystem]) {
      if (!declared.has(name)) {
        throw new Error(
          `'${name}' names no declared coordinate system (declared: ${
            [...declared.keys()].sort().join(", ")
          })`,
        );
      }
    }
    if (declared.get(inputSystem) !== declared.get(outputSystem)) {
      throw new Error(
        `'${inputSystem}' names ${declared.get(inputSystem)} axes and ` +
          `'${outputSystem}' names ${declared.get(outputSystem)}; a field ` +
          `transform maps a point onto one of the same dimension, which is ` +
          `what the field holds per point`,
      );
    }
    input = inputSystem;
    output = outputSystem;
  } else {
    throw new Error("pass both inputSystem and outputSystem, or neither");
  }

  const entry: Displacements | Coordinates = {
    type: transformType,
    path: entryPath,
    interpolation,
    input: { name: input },
    output: { name: output },
  };
  return new NgffMultiscales({
    images: multiscales.images,
    metadata: {
      ...metadata,
      coordinateSystems: systems,
      coordinateTransformations: [
        ...(metadata.coordinateTransformations ?? []),
        entry,
      ],
    },
    scaleFactors: multiscales.scaleFactors,
    method: multiscales.method,
    chunks: multiscales.chunks,
  });
}

/**
 * The standalone checks -- the multiscales is the field, so its shape is here
 * to check: one component axis of the right type first, then the spatial axes,
 * one component per spatial axis. Returns the spatial dims.
 */
function checkFieldImage(
  multiscales: NgffMultiscales,
  componentType: string,
): string[] {
  const image = multiscales.images[0];
  const types = image.axesTypes ?? {};
  const componentDims = Object.entries(types)
    .filter(([, axisType]) => axisType === componentType)
    .map(([dim]) => dim);
  if (componentDims.length !== 1) {
    throw new Error(
      `the field image must have exactly one axis of type '${componentType}' ` +
        `(axesTypes on the NgffImage); got ${
          componentDims.length ? componentDims.join(", ") : "none"
        } on dims ${image.dims.join(", ")}`,
    );
  }
  const component = componentDims[0];
  const other = image.dims.filter((dim) =>
    dim !== component &&
    (types[dim] ?? DEFAULT_AXIS_TYPES[dim] ?? "space") !== "space"
  );
  if (other.length) {
    throw new Error(
      `a field transform is defined on spatial axes only; dims ${
        image.dims.join(", ")
      } name ${
        other.join(", ")
      }, which a coordinate system cannot carry as space`,
    );
  }
  if (image.dims[0] !== component) {
    throw new Error(
      `the field's dims are ${
        image.dims.join(", ")
      }; the component axis '${component}' must come first, then the spatial ` +
        "axes in order",
    );
  }
  const spatial = image.dims.filter((dim) => dim !== component);
  const components = image.data.shape[0];
  if (components !== spatial.length) {
    throw new Error(
      `the field holds ${components} components per point over spatial axes ${
        spatial.join(", ")
      }, which name ${spatial.length} axes`,
    );
  }
  return spatial;
}

/**
 * The systems with a spatial one named `name`, declared from the field's own
 * axes if absent; an existing one must already carry those axes in order, or
 * the declaration would reference a system the field does not match.
 */
function withSpatialSystem(
  multiscales: NgffMultiscales,
  systems: CoordinateSystem[],
  name: string,
  spatial: string[],
): CoordinateSystem[] {
  const existing = systems.find((system) => system.name === name);
  if (existing) {
    const axes = existing.axes.map((axis) => axis.name);
    if (
      axes.length !== spatial.length || axes.some((a, i) => a !== spatial[i])
    ) {
      throw new Error(
        `coordinate system '${name}' is already declared with axes ${
          axes.join(", ")
        }, but the field's spatial axes are ${spatial.join(", ")}`,
      );
    }
    const other = existing.axes.filter((axis) => axis.type !== "space");
    if (other.length > 0) {
      throw new Error(
        `coordinate system '${name}' carries ${
          other.map((axis) => `${axis.name} (${axis.type})`).join(", ")
        } where a field transform needs space; it is defined on spatial axes ` +
          `only`,
      );
    }
    return systems;
  }
  const units = multiscales.images[0].axesUnits ?? {};
  const axes: Axis[] = spatial.map((dim) => ({
    name: dim,
    type: "space",
    unit: units[dim],
  }));
  return [...systems, { name, axes }];
}
