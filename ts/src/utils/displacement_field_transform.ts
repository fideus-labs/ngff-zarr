// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Bridge ITK displacement fields and RFC-5 `displacements` transformations.
 *
 * A displacement field is a transformation and an array at once. ITK keeps the
 * array inside the transform, as a vector image with its own grid; RFC-5 keeps
 * it in the store, as a multiscale image the `displacements` entry points at
 * by `path`. So the two conversions here are the only ones in this package
 * whose result has two parts, and they stay pure: nothing is read from or
 * written to a store. The caller writes the field next to its image, and
 * loads it back, with `toOmeZarr` and `fromOmeZarr`.
 *
 * The same two conventions as the affine conversion apply, plus one.
 *
 * Axis order. ITK orders a vector's components fastest-axis-first, x then y
 * then z, and lays the field out as `[z][y][x][component]`. RFC-5 says the
 * `i`-th component of the field refers to the `i`-th axis of the output
 * coordinate system, so components follow `dims`; the same permutation as for
 * an affine's matrix is applied to the component axis.
 *
 * Frames. An ITK vector is a difference of two physical points. An RFC-5
 * displacement is a difference of two intrinsic points, `d = q' - q`. With
 * `phi(q) = D (q - o) + o` relating each image's intrinsic system to ITK
 * physical space (`D` from RFC-4 orientation, `o` the translation), a vector
 * `v` sampled at the input grid point `q` becomes
 *
 *     d(q) = D_out^-1 (v + D_in (q - o_in) + o_in - o_out) + o_out - q
 *
 * which collapses to `d = D^-1 v` when both images share a frame. The field's
 * own grid follows `phi_in^-1`, so `translation = D_in^-1 (o_f - o_in) + o_in`
 * and the spacing is unchanged.
 *
 * Grid direction. RFC-5 maps the field's array coordinates to the input system
 * with the field's own `coordinateTransformations`, which this package writes
 * as a scale and a translation. That can only express a grid oriented like the
 * input image, so a field whose direction matrix differs from the fixed
 * image's (the identity when no frames are given) is refused rather than
 * written with a mapping a reader would misread. Registrations sample the
 * field on the fixed grid, so the common case is exact.
 */

import * as zarr from "zarrita";
import type { Image, Transform, TransformList } from "itk-wasm";
import { NgffImage } from "../types/ngff_image.ts";
import type { NgffMultiscales } from "../types/multiscales.ts";
import type { Coordinates, Displacements } from "../types/zarr_metadata.ts";
import { toNgffImage } from "../io/to_ngff_image.ts";
import {
  changeOfFrame,
  directionRows,
  type FrameGeometry,
  itkDirection,
  optionalFrameGeometry,
  transposed,
} from "./itk_direction.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

/** The name given to the component axis of a converted field. */
const COMPONENT_DIM = "c";

type Vectors = Float32Array | Float64Array;

/** A field in ITK conventions: `[z][y][x][component]` on an ITK-order grid. */
interface ItkField {
  vectors: Vectors;
  size: number[];
  origin: number[];
  spacing: number[];
  direction: number[][];
}

/** The fixed and moving images a converted field relates. */
export interface FieldFrames {
  /** The image whose grid the field is sampled on and maps from. */
  fixed?: NgffImage;
  /** The image the field maps into. */
  moving?: NgffImage;
}

export interface ItkDisplacementFieldOptions extends FieldFrames {
  /** Where the field will be written, relative to the image's group. */
  path: string;
}

/** The two parts a converted displacement field is made of. */
export interface NgffDisplacementField {
  transform: Displacements;
  field: NgffImage;
}

function checkDims(dims: string[]): void {
  if (dims.length === 0 || dims.some((dim) => !SPATIAL_DIMS.includes(dim))) {
    throw new Error(
      `a displacement field is defined on spatial axes only; got dims ` +
        `[${dims.join(", ")}]. ITK has no notion of a time or channel axis, ` +
        "and RFC-5 requires one field dimension per input axis.",
    );
  }
  if (new Set(dims).size !== dims.length) {
    throw new Error(`dims [${dims.join(", ")}] name an axis twice`);
  }
}

function itkAxisOrder(dims: string[]): string[] {
  return SPATIAL_DIMS.filter((dim) => dims.includes(dim));
}

function identity(dimension: number): number[][] {
  return Array.from(
    { length: dimension },
    (_, row) => Array.from({ length: dimension }, (_, col) => +(row === col)),
  );
}

function allClose(left: number[][], right: number[][]): boolean {
  return left.every((row, i) =>
    row.every((value, j) => Math.abs(value - right[i][j]) <= 1e-9)
  );
}

function matvec(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) =>
    row.reduce((sum, value, col) => sum + value * vector[col], 0)
  );
}

/**
 * The frame pair of two images with no orientation and no translation.
 *
 * With it the terms below all vanish, so the conversion that changes frames
 * and the one that does not are the same arithmetic rather than two branches.
 */
function unorientedFrames(dimension: number): FrameGeometry {
  const zero = new Array<number>(dimension).fill(0);
  return {
    directionFixed: identity(dimension),
    directionMoving: identity(dimension),
    originFixed: zero,
    originMoving: zero,
  };
}

/** `D_out`, `M - I` and `b`: the per-point terms of the two conversions. */
interface FrameTerms {
  directionOut: number[][];
  shiftMatrix: number[][];
  shiftVector: number[];
  /** Whether either `q` term is non-zero; a shared frame makes both vanish. */
  shifts: boolean;
}

/**
 * The per-point terms relating ITK vectors and RFC-5 displacements.
 *
 * An ITK vector is a difference of two *physical* points; an RFC-5
 * displacement is a difference of two *intrinsic* points. Writing
 * `phi_out^-1 . phi_in` as `q -> M q + b` -- the change of frame the affine
 * conversion applies, given the identity as its mapping -- the defining
 * identity `phi_out(q + d(q)) = phi_in(q) + v(q)` rearranges to
 *
 *     d(q) = D_out^-1 v + (M - I) q + b
 *     v(q) = D_out (d - (M - I) q - b)
 *
 * so one derivation serves both directions. The two `q` terms vanish when the
 * images share a frame, leaving `d = D_out^-1 v`.
 */
function frameTerms(frame: FrameGeometry): FrameTerms {
  const dimension = frame.originFixed.length;
  const { matrix, offset } = changeOfFrame(
    identity(dimension),
    new Array<number>(dimension).fill(0),
    frame.directionFixed,
    frame.directionMoving,
    frame.originFixed,
    frame.originMoving,
  );
  const shiftMatrix = matrix.map((row, i) =>
    row.map((value, j) => value - (i === j ? 1 : 0))
  );
  return {
    directionOut: frame.directionMoving,
    shiftMatrix,
    shiftVector: offset,
    shifts: shiftMatrix.some((row) => row.some((value) => value !== 0)) ||
      offset.some((value) => value !== 0),
  };
}

/**
 * The per-voxel value conversion both field directions share.
 *
 * `toIntrinsic` turns an ITK vector into an RFC-5 displacement,
 * `d(q) = D_out^-1 v + (M - I) q + b`, and `toPhysical` is its inverse. Every
 * term is a function of the voxel's position and the frames alone, which is
 * what lets a block be converted where it sits: see `convertFieldBlock`. A
 * `coordinates` field holds `q + d` rather than `d`, so `absolute` folds the
 * grid point itself into the positional term, in one pass over the grid.
 */
function fieldValueConverters(frame: FrameGeometry, absolute: boolean): {
  toIntrinsic: (vector: number[], point: number[]) => number[];
  toPhysical: (displacement: number[], point: number[]) => number[];
} {
  const terms = frameTerms(frame);
  const inverseOut = transposed(terms.directionOut);
  return {
    toIntrinsic: (vector: number[], point: number[]): number[] => {
      const rotated = matvec(inverseOut, vector);
      if (!terms.shifts) return rotated;
      const shift = matvec(terms.shiftMatrix, point);
      return rotated.map((value, i) => value + shift[i] + terms.shiftVector[i]);
    },
    toPhysical: (displacement: number[], point: number[]): number[] => {
      if (!terms.shifts && !absolute) {
        return matvec(terms.directionOut, displacement);
      }
      const shift = matvec(terms.shiftMatrix, point);
      return matvec(
        terms.directionOut,
        displacement.map((value, i) =>
          value - shift[i] - terms.shiftVector[i] - (absolute ? point[i] : 0)
        ),
      );
    },
  };
}

/** The field a vector `Image` or a `DisplacementField` transform holds. */
function decodeField(input: Transform | TransformList | Image): ItkField {
  if (Array.isArray(input)) {
    if (input.length !== 1) {
      throw new Error(
        `expected a single displacement field, got a transform list of ` +
          `${input.length} entries. A registration that chains an affine ` +
          "and a field is not converted as one transform.",
      );
    }
    return decodeField(input[0]);
  }
  if ("imageType" in input) return decodeImage(input);
  if ("transformType" in input) return decodeTransform(input);
  throw new Error(
    "unsupported displacement field input. Expected an ITK-Wasm vector " +
      "Image or an ITK-Wasm 'DisplacementField' transform.",
  );
}

function decodeImage(image: Image): ItkField {
  const dimension = image.imageType.dimension;
  const components = image.imageType.components;
  if (components !== dimension) {
    throw new Error(
      `a displacement field over ${dimension} dimensions needs ` +
        `${dimension} components per pixel; got ${components}`,
    );
  }
  if (image.data === null) throw new Error("the field image holds no data");
  const size = [...image.size];
  const expected = size.reduce((a, b) => a * b, 1) * dimension;
  if (image.data.length !== expected) {
    throw new Error(
      `a field of size [${size.join(", ")}] with ${dimension} components ` +
        `holds ${expected} values; got ${image.data.length}`,
    );
  }
  return {
    vectors: asVectors(image.data as ArrayLike<number>),
    size,
    origin: [...image.origin],
    spacing: [...image.spacing],
    direction: directionRows(
      Float64Array.from(image.direction as ArrayLike<number>),
      dimension,
    ),
  };
}

function decodeTransform(transform: Transform): ItkField {
  const parameterization = String(
    transform.transformType.transformParameterization,
  );
  if (parameterization !== "DisplacementField") {
    throw new Error(
      `expected an ITK-Wasm 'DisplacementField' transform, got ` +
        `'${parameterization}'. Linear transforms are converted by ` +
        "itkTransformToNgffTransform.",
    );
  }
  const dimension = transform.transformType.inputDimension;
  const fixed = Array.from(transform.fixedParameters as ArrayLike<number>);
  const expected = 3 * dimension + dimension * dimension;
  if (fixed.length !== expected) {
    throw new Error(
      `an ITK-Wasm 'DisplacementField' transform of dimension ${dimension} ` +
        `packs size, origin, spacing and direction into ${expected} fixed ` +
        `parameters; got ${fixed.length}`,
    );
  }
  const size = fixed.slice(0, dimension).map((value) => Math.round(value));
  const count = size.reduce((a, b) => a * b, 1) * dimension;
  const parameters = transform.parameters as ArrayLike<number>;
  if (parameters.length !== count) {
    throw new Error(
      `an ITK-Wasm 'DisplacementField' transform over a grid of size ` +
        `[${size.join(", ")}] holds ${count} parameters; got ` +
        `${parameters.length}`,
    );
  }
  return {
    vectors: asVectors(parameters),
    size,
    origin: fixed.slice(dimension, 2 * dimension),
    spacing: fixed.slice(2 * dimension, 3 * dimension),
    direction: directionRows(
      Float64Array.from(fixed.slice(3 * dimension)),
      dimension,
    ),
  };
}

function asVectors(values: ArrayLike<number>): Vectors {
  return values instanceof Float32Array ? values : Float64Array.from(values);
}

/** Multipliers turning an ITK-order index into a flat `[z][y][x]` offset. */
function strides(size: number[]): number[] {
  const result = new Array<number>(size.length);
  let step = 1;
  for (let axis = 0; axis < size.length; axis++) {
    result[axis] = step;
    step *= size[axis];
  }
  return result;
}

/**
 * Convert an ITK displacement field to an RFC-5 `displacements` transform.
 *
 * The result has two parts, because the field is an array: the transform
 * metadata, which names `path`, and the field itself as an image to be written
 * at that path. Write the field first, into a subgroup of the store the image
 * goes to, then the image.
 *
 * @param input An ITK-Wasm `DisplacementField` transform, a one-entry list of
 *   it, or a vector ITK-Wasm `Image` holding the field directly, the form a
 *   warp comes in from most registration tools. The field maps *fixed* points
 *   into *moving* space.
 * @param dims The spatial axis names of the input coordinate system, in RFC-5
 *   (Zarr) order. The field must be defined on these axes and no others.
 * @param options `path`, where the field will be written relative to the
 *   image's group, and the optional `fixed` and `moving` images. Passing both
 *   re-expresses the vectors on the images' intrinsic coordinate systems,
 *   including the direction matrix derived from RFC-4 anatomical orientation,
 *   and gives the field the fixed image's orientation and units. Omitting
 *   them is exact only when neither image carries an anatomical orientation.
 * @returns The `displacements` transform, with `interpolation` set to
 *   `linear` (ITK's interpolator for a field), and the field as an
 *   `NgffImage` whose first axis carries the components, `type:
 *   "displacement"`, followed by `dims`.
 * @throws If the field's grid is not oriented like the fixed image (the
 *   identity without frames), which the field's scale and translation could
 *   not express; resample the field onto the fixed grid first. Also for a
 *   field whose dimensionality does not match `dims`.
 */
export async function itkDisplacementFieldToNgffTransform(
  input: Transform | TransformList | Image,
  dims: string[],
  options: ItkDisplacementFieldOptions,
): Promise<NgffDisplacementField> {
  checkDims(dims);
  const { vectors, size, origin, spacing, direction } = decodeField(input);
  const dimension = size.length;
  if (dimension !== dims.length) {
    throw new Error(
      `the field has ${dimension} components over ${dimension} dimensions, ` +
        `but dims [${dims.join(", ")}] name ${dims.length} axes`,
    );
  }
  const itkDims = itkAxisOrder(dims);
  let frame = optionalFrameGeometry(
    options.fixed,
    options.moving,
    itkDims,
  );
  if (frame === undefined) {
    if (!allClose(direction, identity(dimension))) {
      throw new Error(
        "the field's grid has a non-identity direction matrix, which its " +
          "scale and translation cannot express. Pass the fixed and moving " +
          "images so it is read against the fixed image's orientation, or " +
          "resample the field onto the fixed grid.",
      );
    }
    frame = unorientedFrames(dimension);
  } else if (!allClose(direction, frame.directionFixed)) {
    throw new Error(
      "the field's grid is not oriented like the fixed image: its " +
        `direction is ${JSON.stringify(direction)} where the fixed image ` +
        `gives ${JSON.stringify(frame.directionFixed)}. Resample the field ` +
        "onto the fixed grid first.",
    );
  }

  // The field's grid follows phi_in^-1, so its origin moves and its spacing
  // does not.
  const gridOrigin = matvec(
    transposed(frame.directionFixed),
    origin.map((value, i) => value - frame.originFixed[i]),
  ).map((value, i) => value + frame.originFixed[i]);

  // d(q) = D_out^-1 v + (M - I) q + b -- see fieldValueConverters.
  const { toIntrinsic } = fieldValueConverters(frame, false);

  // (c, *dims) with components in dims order, C-contiguous.
  const shape = dims.map((dim) => size[itkDims.indexOf(dim)]);
  const voxels = size.reduce((a, b) => a * b, 1);
  const out = vectors instanceof Float32Array
    ? new Float32Array(voxels * dimension)
    : new Float64Array(voxels * dimension);
  const itkStrides = strides(size);
  const dimsStrides = new Array<number>(dimension);
  let step = 1;
  for (let axis = dimension - 1; axis >= 0; axis--) {
    dimsStrides[axis] = step;
    step *= shape[axis];
  }
  const componentOf = dims.map((dim) => itkDims.indexOf(dim));
  const index = new Array<number>(dimension);
  const vector = new Array<number>(dimension);
  const point = new Array<number>(dimension);
  for (let voxel = 0; voxel < voxels; voxel++) {
    let offset = 0;
    for (let axis = 0; axis < dimension; axis++) {
      index[axis] = Math.floor(voxel / itkStrides[axis]) % size[axis];
      point[axis] = gridOrigin[axis] + spacing[axis] * index[axis];
      vector[axis] = vectors[voxel * dimension + axis];
    }
    const displacement = toIntrinsic(vector, point);
    for (let axis = 0; axis < dimension; axis++) {
      offset += index[componentOf[axis]] * dimsStrides[axis];
    }
    for (let component = 0; component < dimension; component++) {
      out[component * voxels + offset] = displacement[componentOf[component]];
    }
  }

  const scale: Record<string, number> = { [COMPONENT_DIM]: 1.0 };
  const translation: Record<string, number> = { [COMPONENT_DIM]: 0.0 };
  for (const dim of dims) {
    scale[dim] = spacing[itkDims.indexOf(dim)];
    translation[dim] = gridOrigin[itkDims.indexOf(dim)];
  }
  const name = options.path.replace(/\/+$/, "").split("/").pop() ||
    "displacement_field";
  const image = await toNgffImage(out, {
    dims: [COMPONENT_DIM, ...dims],
    shape: [dimension, ...shape],
    scale,
    translation,
    name,
    axesTypes: { [COMPONENT_DIM]: "displacement" },
  });

  const fixed = options.fixed;
  const orientations = fixed?.axesOrientations
    ? Object.fromEntries(
      dims
        .filter((dim) => fixed.axesOrientations![dim] !== undefined)
        .map((dim) => [dim, fixed.axesOrientations![dim]]),
    )
    : undefined;
  const units = fixed?.axesUnits
    ? Object.fromEntries(
      dims
        .filter((dim) => fixed.axesUnits![dim] !== undefined)
        .map((dim) => [dim, fixed.axesUnits![dim]]),
    )
    : undefined;
  const field = new NgffImage({
    data: image.data,
    dims: image.dims,
    scale: image.scale,
    translation: image.translation,
    name: image.name,
    axesUnits: units && Object.keys(units).length > 0 ? units : undefined,
    axesOrientations: orientations && Object.keys(orientations).length > 0
      ? orientations
      : undefined,
    axesTypes: image.axesTypes,
    computedCallbacks: undefined,
  });
  return {
    transform: {
      type: "displacements",
      path: options.path,
      interpolation: "linear",
    },
    field,
  };
}

/**
 * Refuse a field whose grid cannot be placed without the two images.
 *
 * An ITK transform lives in physical space, and an anatomical orientation only
 * says where a grid sits once the images it relates are known.
 *
 * @param image The field image.
 * @param dims The spatial axis names, in RFC-5 order.
 * @throws If the field carries an anatomical orientation.
 */
export interface ConvertFieldBlockOptions {
  /** `displacements` (the block holds offsets) or `coordinates` (absolute positions). */
  transformType?: "displacements" | "coordinates";
  /** The fixed and moving images the field relates; pass both or neither. */
  fixed?: NgffImage;
  /** See `fixed`. */
  moving?: NgffImage;
  /** Convert the store's values back to ITK's convention. */
  inverse?: boolean;
}

/**
 * Convert one block of a field between ITK's convention and RFC-5's.
 *
 * The block-level face of the two whole-field converters, for a field that is
 * never assembled: a producer computing an ITK-convention field region by
 * region converts each block on its way into a store, and `inverse` is the way
 * back. Every term of the conversion is a function of the voxel's position and
 * the frames alone (`fieldValueConverters`), so converting a block with the
 * block's own translation equals cutting that block from the converted whole.
 *
 * @param values The block, channel-first: `shape[0]` components, then the
 * spatial axes in `dims` order. Forward, the components are ITK's (a vector's
 * x, then y, then z); with `inverse` they follow `dims`, as the store holds
 * them.
 * @param shape The block's shape, `[components, ...spatial in dims order]`.
 * @param dims The spatial axis names of the input coordinate system, in RFC-5
 * (Zarr) order.
 * @param translation Where the block starts on the field's grid, per axis: the
 * field's own translation advanced by the block's voxel offset times `scale`.
 * This is the RFC-5 value, in the input coordinate system's frame -- with
 * `fixed`/`moving` it is NOT the block's ITK physical origin.
 * @param scale The field's scale per axis.
 * @returns The converted block, channel-first, components in `dims` order
 * (ITK's with `inverse`).
 */
export function convertFieldBlock(
  values: Float32Array | Float64Array,
  shape: number[],
  dims: string[],
  translation: Record<string, number>,
  scale: Record<string, number>,
  options: ConvertFieldBlockOptions = {},
): Float32Array | Float64Array {
  checkDims(dims);
  const dimension = dims.length;
  const { transformType = "displacements", inverse = false } = options;
  if (transformType !== "displacements" && transformType !== "coordinates") {
    throw new Error(
      `transformType must be 'displacements' or 'coordinates', got '${transformType}'`,
    );
  }
  if (shape.length !== dimension + 1) {
    throw new Error(
      `the block has shape ${shape.join(", ")}; dims ${
        dims.join(", ")
      } need one component axis and ${dimension} spatial axes`,
    );
  }
  if (shape[0] !== dimension) {
    throw new Error(
      `the block holds ${shape[0]} components per point, but dims ${
        dims.join(", ")
      } name ${dimension} axes`,
    );
  }
  for (const dim of dims) {
    if (translation[dim] === undefined || scale[dim] === undefined) {
      throw new Error(
        `translation and scale must cover every axis dims ${
          dims.join(", ")
        } names; '${dim}' is missing`,
      );
    }
  }

  const itkDims = itkAxisOrder(dims);
  const frame = optionalFrameGeometry(options.fixed, options.moving, itkDims) ??
    unorientedFrames(dimension);
  const absolute = transformType === "coordinates";
  const { toIntrinsic, toPhysical } = fieldValueConverters(frame, absolute);
  // The component axis of the block, per ITK axis and back.
  const componentOf = dims.map((dim) => itkDims.indexOf(dim));

  const spatial = shape.slice(1);
  const voxels = spatial.reduce((a, b) => a * b, 1);
  const out = values instanceof Float32Array
    ? new Float32Array(values.length)
    : new Float64Array(values.length);
  // C-contiguous strides over the block's spatial axes, which are in dims order.
  const dimsStrides = new Array<number>(dimension);
  let step = 1;
  for (let axis = dimension - 1; axis >= 0; axis--) {
    dimsStrides[axis] = step;
    step *= spatial[axis];
  }
  const point = new Array<number>(dimension);
  const held = new Array<number>(dimension);
  for (let voxel = 0; voxel < voxels; voxel++) {
    for (let axis = 0; axis < dimension; axis++) {
      const index = Math.floor(voxel / dimsStrides[axis]) % spatial[axis];
      // The point in ITK axis order, which is what the converters index.
      point[itkDims.indexOf(dims[axis])] = translation[dims[axis]] +
        scale[dims[axis]] * index;
    }
    for (let axis = 0; axis < dimension; axis++) {
      // Forward the components are ITK's already; inverse they follow dims.
      const source = inverse ? componentOf[axis] : axis;
      held[source] = values[axis * voxels + voxel];
    }
    const converted = inverse
      ? toPhysical(held, point)
      : toIntrinsic(held, point);
    for (let axis = 0; axis < dimension; axis++) {
      const target = inverse ? axis : componentOf[axis];
      out[axis * voxels + voxel] = converted[target];
    }
  }
  return out;
}

export function checkUnorientedField(image: NgffImage, dims: string[]): void {
  const itkDims = itkAxisOrder(dims);
  const direction = directionRows(itkDirection(image, itkDims), dims.length);
  if (allClose(direction, identity(dims.length))) return;
  throw new Error(
    "the field carries an anatomical orientation; pass the fixed and " +
      "moving images so its grid is placed in their frame",
  );
}

/**
 * The field image a field transform names, checked against `dims`.
 *
 * Takes the `NgffImage` or the `NgffMultiscales` a caller passes in `fields`,
 * and returns the single-scale image, after checking that its axes are the
 * component axis followed by `dims` in order and that it holds one component
 * per input axis.
 *
 * @param transform The `displacements` or `coordinates` transform.
 * @param field The field image or multiscales, as `fields` holds it.
 * @param dims The spatial axis names, in RFC-5 (Zarr) order.
 * @returns The field as a single-scale image; the finest level of a
 *   multiscales.
 * @throws If the field has no single component axis of the type the transform
 *   calls for, if its axes are not that axis followed by `dims`, or if it
 *   holds a number of components other than `dims.length`.
 */
export function fieldImage(
  transform: Displacements | Coordinates,
  field: NgffImage | NgffMultiscales,
  dims: string[],
): NgffImage {
  const componentType = transform.type === "coordinates"
    ? "coordinate"
    : "displacement";
  let componentDims: string[];
  let image: NgffImage;
  if ("images" in field && "metadata" in field) {
    // A read multiscales keeps the axis types in its metadata, not on the
    // image: the component axis is the one typed there.
    componentDims = field.metadata.axes
      .filter((axis) => axis.type === componentType)
      .map((axis) => axis.name);
    image = field.images[0];
  } else {
    image = field;
    componentDims = Object.entries(image.axesTypes ?? {})
      .filter(([, type]) => type === componentType)
      .map(([dim]) => dim);
  }
  if (componentDims.length !== 1) {
    throw new Error(
      `the field image must have exactly one axis of type '${componentType}' ` +
        "(axesTypes on an NgffImage, the axes metadata of a multiscales); " +
        `got [${componentDims.join(", ")}] on dims [${image.dims.join(", ")}]`,
    );
  }
  const expectedDims = [componentDims[0], ...dims];
  if (
    image.dims.length !== expectedDims.length ||
    image.dims.some((dim, i) => dim !== expectedDims[i])
  ) {
    throw new Error(
      `the field's dims are [${image.dims.join(", ")}]; a ${transform.type} ` +
        `transform over dims [${dims.join(", ")}] needs ` +
        `[${expectedDims.join(", ")}]: the component axis first, then the ` +
        "input axes in order",
    );
  }
  if (image.data.shape[0] !== dims.length) {
    throw new Error(
      `the field holds ${image.data.shape[0]} components per point, but dims ` +
        `[${dims.join(", ")}] name ${dims.length} axes`,
    );
  }
  return image;
}

/**
 * The field indices a grid of `shape` at `translation` reads.
 *
 * The field is evaluated at the grid's own points, so the window is that
 * grid's extent expressed in field indices, whatever the displacement is:
 * what a displacement sizes is the *moving* read, which
 * {@link fieldDisplacementRange} answers.
 *
 * @param image The field image.
 * @param dims The spatial axis names, in RFC-5 order.
 * @param translation The grid's translation, keyed by dimension.
 * @param scale The grid's scale, keyed by dimension.
 * @param shape The grid's extent, in `dims` order.
 * @param margin Lattice points kept beyond the bracketing pair, so that linear
 *   interpolation at a point on the boundary reads the same values it reads
 *   from the whole field.
 * @returns `[start, stop]` per axis, in `dims` order, clamped to the field,
 *   and whether the grid also has points beyond the field, which ITK
 *   displaces by nothing. {@link fieldDisplacementRange} takes the second as
 *   its `outside`.
 */
export function fieldWindow(
  image: NgffImage,
  dims: string[],
  translation: Record<string, number>,
  scale: Record<string, number>,
  shape: number[],
  margin = 1,
): { window: [number, number][]; outside: boolean } {
  let outside = false;
  const window = dims.map((dim, axis) => {
    const extent = image.data.shape[1 + axis];
    if (shape[axis] === 0) return [0, 0] as [number, number];
    const corners = [0, shape[axis] - 1].map((index) =>
      (translation[dim] + scale[dim] * index - image.translation[dim]) /
      image.scale[dim]
    );
    if (Math.min(...corners) < 0 || Math.max(...corners) > extent - 1) {
      outside = true;
    }
    const low = Math.floor(Math.min(...corners)) - margin;
    const high = Math.ceil(Math.max(...corners)) + margin + 1;
    return [
      Math.max(0, Math.min(low, extent)),
      Math.max(0, Math.min(high, extent)),
    ] as [number, number];
  });
  return { window, outside };
}

/**
 * The range of displacement a window of the field can produce, per component.
 *
 * A field is read through a kernel that is non-negative and sums to one, so a
 * displacement anywhere is a convex combination of the values around it and
 * lies between their smallest and largest. That makes the range a bound on
 * every interpolated displacement, not a sample of one: walking the boundary
 * of a region instead misses a bump the region encloses.
 *
 * The window is read a chunk at a time and only a number per component is
 * kept, so a field larger than memory is bounded without being held.
 *
 * @param transform The `displacements` or `coordinates` transform.
 * @param image The field image, as {@link fieldImage} returns it.
 * @param dims The spatial axis names, in RFC-5 order.
 * @param window `[start, stop]` per axis, as {@link fieldWindow} returns it.
 * @param outside Whether the grid the window came from also has points beyond
 *   the field. ITK displaces those by nothing, so zero belongs in the range as
 *   much as the values do; leaving it out lets a field that displaces every
 *   point it covers one way carry the region away from the points it does not
 *   cover.
 * @returns The smallest and largest displacement per component, ordered like
 *   `dims`. Zero for an empty window, where the field displaces nothing.
 */
export async function fieldDisplacementRange(
  transform: Displacements | Coordinates,
  image: NgffImage,
  dims: string[],
  window: [number, number][],
  outside = false,
): Promise<{ low: number[]; high: number[] }> {
  const rank = dims.length;
  const low = new Array(rank).fill(Number.POSITIVE_INFINITY);
  const high = new Array(rank).fill(Number.NEGATIVE_INFINITY);
  const zero = { low: new Array(rank).fill(0), high: new Array(rank).fill(0) };
  if (window.some(([start, stop]) => stop <= start)) return zero;

  const absolute = transform.type === "coordinates";
  const chunkShape = image.data.chunks ?? image.data.shape;
  const starts: number[][] = window.map(([begin, end], axis) => {
    const size = chunkShape[1 + axis];
    const first = Math.floor(begin / size) * size;
    const positions: number[] = [];
    for (let position = first; position < end; position += size) {
      positions.push(position);
    }
    return positions;
  });

  for (const origin of gridPositions(starts)) {
    const selection: (zarr.Slice | null)[] = [null];
    const sizes: number[] = [];
    origin.forEach((start, axis) => {
      const stop = Math.min(
        start + chunkShape[1 + axis],
        image.data.shape[1 + axis],
      );
      selection.push(zarr.slice(start, stop));
      sizes.push(stop - start);
    });
    const chunk = await zarr.get(image.data, selection);
    const values = chunk.data as ArrayLike<number>;
    const count = sizes.reduce((a, b) => a * b, 1);
    const strides = new Array(rank).fill(1);
    for (let axis = rank - 2; axis >= 0; axis--) {
      strides[axis] = strides[axis + 1] * sizes[axis + 1];
    }
    for (let component = 0; component < rank; component++) {
      const offset = component * count;
      const origin_ = image.translation[dims[component]];
      const step = image.scale[dims[component]];
      for (let index = 0; index < count; index++) {
        let value = values[offset + index];
        if (absolute) {
          // A coordinates field holds the output position of each grid point
          // rather than the offset from it, so the grid point comes off first.
          const along = Math.floor(index / strides[component]) %
            sizes[component];
          value -= origin_ + step * (origin[component] + along);
        }
        if (value < low[component]) low[component] = value;
        if (value > high[component]) high[component] = value;
      }
    }
  }
  // A window that reached no value displaces nothing.
  if (!Number.isFinite(low[0])) return zero;
  if (outside) {
    for (let component = 0; component < rank; component++) {
      low[component] = Math.min(low[component], 0);
      high[component] = Math.max(high[component], 0);
    }
  }
  return { low, high };
}

function* gridPositions(starts: number[][]): Generator<number[]> {
  if (starts.length === 0) {
    yield [];
    return;
  }
  const [head, ...rest] = starts;
  for (const value of head) {
    for (const tail of gridPositions(rest)) yield [value, ...tail];
  }
}

/**
 * Convert an RFC-5 `displacements` or `coordinates` transform to ITK.
 *
 * The counterpart of {@link itkDisplacementFieldToNgffTransform}. The field
 * is the image stored at `transform.path`; load it with
 * `fromOmeZarr(`${store}/${transform.path}`)`.
 *
 * A `coordinates` field holds the absolute output position of each grid point
 * where a `displacements` field holds the offset from it. The two differ by
 * the position of the grid point itself, so both reach ITK as one
 * `DisplacementField`: ITK has no absolute-coordinate transform.
 *
 * @param transform The `displacements` or `coordinates` transform.
 * @param field The field image: an `NgffImage` whose component axis is the one
 *   with `axesTypes` `displacement` (`coordinate` for a `coordinates`
 *   transform), followed by `dims` in order; or an `NgffMultiscales`, whose
 *   finest level is used and whose metadata names the component axis.
 * @param dims The spatial axis names of the input coordinate system, in RFC-5
 *   (Zarr) order.
 * @param frames The fixed and moving images the field relates; see
 *   {@link itkDisplacementFieldToNgffTransform}. The field's own orientation,
 *   if any, must be the fixed image's.
 * @returns A single-entry ITK-Wasm `TransformList` of parameterization
 *   `DisplacementField`.
 * @throws If the field's axes are not the component axis followed by `dims`,
 *   or if its orientation is not the fixed image's.
 */
export async function ngffDisplacementFieldToItkTransform(
  transform: Displacements | Coordinates,
  field: NgffImage | NgffMultiscales,
  dims: string[],
  frames_: FieldFrames = {},
): Promise<TransformList> {
  checkDims(dims);
  // A coordinates field holds the absolute output position of each grid point
  // where a displacements field holds the offset from it, so the two differ by
  // the position of the grid point itself. ITK has no absolute-coordinate
  // transform, so both reach it as one DisplacementField.
  const absolute = transform.type === "coordinates";
  const image = fieldImage(transform, field, dims);

  const chunk = await zarr.get(image.data, null);
  const data = chunk.data as ArrayLike<number>;
  const shape = chunk.shape;
  const dimension = dims.length;

  const itkDims = itkAxisOrder(dims);
  const size = itkDims.map((dim) => shape[1 + dims.indexOf(dim)]);
  const spacing = itkDims.map((dim) => image.scale[dim]);
  const translation = itkDims.map((dim) => image.translation[dim]);
  const ownDirection = directionRows(itkDirection(image, itkDims), dimension);

  let frame = optionalFrameGeometry(frames_.fixed, frames_.moving, itkDims);
  if (frame === undefined) {
    checkUnorientedField(image, dims);
    frame = unorientedFrames(dimension);
  } else if (
    !allClose(ownDirection, identity(dimension)) &&
    !allClose(ownDirection, frame.directionFixed)
  ) {
    throw new Error(
      "the field's orientation is not the fixed image's: it gives " +
        `${JSON.stringify(ownDirection)} where the fixed image gives ` +
        `${JSON.stringify(frame.directionFixed)}`,
    );
  }

  const direction = frame.directionFixed;
  const origin = matvec(
    direction,
    translation.map((value, i) => value - frame.originFixed[i]),
  ).map((value, i) => value + frame.originFixed[i]);

  // v(q) = D_out (d - (M - I) q - b) -- see fieldValueConverters.
  const { toPhysical } = fieldValueConverters(frame, absolute);

  if (
    transform.interpolation !== undefined &&
    transform.interpolation !== "linear"
  ) {
    console.warn(
      `the ${transform.type} transform asks for '${transform.interpolation}' ` +
        "interpolation; ITK interpolates a displacement field linearly. " +
        "RFC-5 leaves the choice to the consumer.",
    );
  }

  const voxels = size.reduce((a, b) => a * b, 1);
  const float32 = data instanceof Float32Array;
  const parameters = float32
    ? new Float32Array(voxels * dimension)
    : new Float64Array(voxels * dimension);
  const itkStrides = strides(size);
  const dimsShape = dims.map((dim) => size[itkDims.indexOf(dim)]);
  const dimsStrides = new Array<number>(dimension);
  let step = 1;
  for (let axis = dimension - 1; axis >= 0; axis--) {
    dimsStrides[axis] = step;
    step *= dimsShape[axis];
  }
  const componentOf = dims.map((dim) => itkDims.indexOf(dim));
  const index = new Array<number>(dimension);
  const point = new Array<number>(dimension);
  const displacement = new Array<number>(dimension);
  for (let voxel = 0; voxel < voxels; voxel++) {
    let offset = 0;
    for (let axis = 0; axis < dimension; axis++) {
      index[axis] = Math.floor(voxel / itkStrides[axis]) % size[axis];
      point[axis] = translation[axis] + spacing[axis] * index[axis];
    }
    for (let axis = 0; axis < dimension; axis++) {
      offset += index[componentOf[axis]] * dimsStrides[axis];
    }
    // displacement[j] is the component along ITK axis j: dims order -> ITK.
    for (let axis = 0; axis < dimension; axis++) {
      displacement[componentOf[axis]] = data[axis * voxels + offset];
    }
    const vector = toPhysical(displacement, point);
    for (let axis = 0; axis < dimension; axis++) {
      parameters[voxel * dimension + axis] = vector[axis];
    }
  }

  const fixedParameters = new Float64Array(
    3 * dimension + dimension * dimension,
  );
  fixedParameters.set(size, 0);
  fixedParameters.set(origin, dimension);
  fixedParameters.set(spacing, 2 * dimension);
  fixedParameters.set(direction.flat(), 3 * dimension);
  const itkTransform: Transform = {
    transformType: {
      transformParameterization: "DisplacementField",
      parametersValueType: float32 ? "float32" : "float64",
      inputDimension: dimension,
      outputDimension: dimension,
    },
    name: "DisplacementFieldTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: fixedParameters.length,
    numberOfParameters: parameters.length,
    fixedParameters,
    parameters,
    metadata: new Map(),
  } as unknown as Transform;
  return [itkTransform];
}
