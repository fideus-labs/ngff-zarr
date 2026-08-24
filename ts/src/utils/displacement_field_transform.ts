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
import type { Displacements } from "../types/zarr_metadata.ts";
import { toNgffImage } from "../io/to_ngff_image.ts";
import { directionRows, frameGeometry, itkDirection } from "./itk_direction.ts";

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

interface Frames {
  directionIn: number[][];
  directionOut: number[][];
  originIn: number[];
  originOut: number[];
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

function transposed(matrix: number[][]): number[][] {
  return matrix[0].map((_, col) => matrix.map((row) => row[col]));
}

function matvec(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) =>
    row.reduce((sum, value, col) => sum + value * vector[col], 0)
  );
}

function frames(
  fixed: NgffImage | undefined,
  moving: NgffImage | undefined,
  dims: string[],
): Frames | undefined {
  if ((fixed === undefined) !== (moving === undefined)) {
    throw new Error("pass both fixed and moving, or neither");
  }
  if (fixed === undefined || moving === undefined) return undefined;
  const geometry = frameGeometry(fixed, moving, itkAxisOrder(dims));
  return {
    directionIn: geometry.directionFixed,
    directionOut: geometry.directionMoving,
    originIn: geometry.originFixed,
    originOut: geometry.originMoving,
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
  const frame = frames(options.fixed, options.moving, dims);

  let gridOrigin: number[];
  let toIntrinsic: (vector: number[], point: number[]) => number[];
  if (frame === undefined) {
    if (!allClose(direction, identity(dimension))) {
      throw new Error(
        "the field's grid has a non-identity direction matrix, which its " +
          "scale and translation cannot express. Pass the fixed and moving " +
          "images so it is read against the fixed image's orientation, or " +
          "resample the field onto the fixed grid.",
      );
    }
    gridOrigin = origin;
    toIntrinsic = (vector) => vector;
  } else {
    if (!allClose(direction, frame.directionIn)) {
      throw new Error(
        "the field's grid is not oriented like the fixed image: its " +
          `direction is ${JSON.stringify(direction)} where the fixed image ` +
          `gives ${JSON.stringify(frame.directionIn)}. Resample the field ` +
          "onto the fixed grid first.",
      );
    }
    const inverseIn = transposed(frame.directionIn);
    const inverseOut = transposed(frame.directionOut);
    gridOrigin = matvec(
      inverseIn,
      origin.map((value, i) => value - frame.originIn[i]),
    ).map((value, i) => value + frame.originIn[i]);
    const shared = allClose(frame.directionIn, frame.directionOut) &&
      frame.originIn.every((value, i) =>
        Math.abs(value - frame.originOut[i]) <= 1e-9
      );
    toIntrinsic = shared
      ? (vector) => matvec(inverseOut, vector)
      : (vector, point) => {
        const rotated = matvec(
          frame.directionIn,
          point.map((value, i) => value - frame.originIn[i]),
        );
        const shifted = vector.map(
          (value, i) =>
            value + rotated[i] + frame.originIn[i] - frame.originOut[i],
        );
        return matvec(inverseOut, shifted).map(
          (value, i) => value + frame.originOut[i] - point[i],
        );
      };
  }

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
 * Convert an RFC-5 `displacements` transform and its field to ITK.
 *
 * The counterpart of {@link itkDisplacementFieldToNgffTransform}. The field
 * is the image stored at `transform.path`; load it with
 * `fromOmeZarr(`${store}/${transform.path}`)`.
 *
 * @param transform The `displacements` transform.
 * @param field The field image: an `NgffImage` whose component axis is the one
 *   with `axesTypes` `displacement`, followed by `dims` in order; or an
 *   `NgffMultiscales`, whose finest level is used and whose metadata names the
 *   component axis.
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
  transform: Displacements,
  field: NgffImage | NgffMultiscales,
  dims: string[],
  frames_: FieldFrames = {},
): Promise<TransformList> {
  checkDims(dims);
  let componentDims: string[];
  let image: NgffImage;
  if ("images" in field && "metadata" in field) {
    // A read multiscales keeps the axis types in its metadata, not on the
    // image: the component axis is the one typed "displacement" there.
    componentDims = field.metadata.axes
      .filter((axis) => axis.type === "displacement")
      .map((axis) => axis.name);
    image = field.images[0];
  } else {
    image = field;
    componentDims = Object.entries(image.axesTypes ?? {})
      .filter(([, type]) => type === "displacement")
      .map(([dim]) => dim);
  }
  if (componentDims.length !== 1) {
    throw new Error(
      "the field image must have exactly one axis of type 'displacement' " +
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
      `the field's dims are [${image.dims.join(", ")}]; a displacements ` +
        `transform over dims [${dims.join(", ")}] needs ` +
        `[${expectedDims.join(", ")}]: the component axis first, then the ` +
        "input axes in order",
    );
  }

  const chunk = await zarr.get(image.data, null);
  const data = chunk.data as ArrayLike<number>;
  const shape = chunk.shape;
  const dimension = dims.length;
  if (shape[0] !== dimension) {
    throw new Error(
      `the field holds ${shape[0]} components per point, but dims ` +
        `[${dims.join(", ")}] name ${dimension} axes`,
    );
  }

  const itkDims = itkAxisOrder(dims);
  const size = itkDims.map((dim) => shape[1 + dims.indexOf(dim)]);
  const spacing = itkDims.map((dim) => image.scale[dim]);
  const translation = itkDims.map((dim) => image.translation[dim]);
  const ownDirection = directionRows(itkDirection(image, itkDims), dimension);

  const frame = frames(frames_.fixed, frames_.moving, dims);
  let origin: number[];
  let direction: number[][];
  let toPhysical: (displacement: number[], point: number[]) => number[];
  if (frame === undefined) {
    if (!allClose(ownDirection, identity(dimension))) {
      throw new Error(
        "the field carries an anatomical orientation; pass the fixed and " +
          "moving images so its grid is placed in their frame",
      );
    }
    origin = translation;
    direction = identity(dimension);
    toPhysical = (displacement) => displacement;
  } else {
    if (
      !allClose(ownDirection, identity(dimension)) &&
      !allClose(ownDirection, frame.directionIn)
    ) {
      throw new Error(
        "the field's orientation is not the fixed image's: it gives " +
          `${JSON.stringify(ownDirection)} where the fixed image gives ` +
          `${JSON.stringify(frame.directionIn)}`,
      );
    }
    origin = matvec(
      frame.directionIn,
      translation.map((value, i) => value - frame.originIn[i]),
    ).map((value, i) => value + frame.originIn[i]);
    direction = frame.directionIn;
    const shared = allClose(frame.directionIn, frame.directionOut) &&
      frame.originIn.every((value, i) =>
        Math.abs(value - frame.originOut[i]) <= 1e-9
      );
    toPhysical = shared
      ? (displacement) => matvec(frame.directionOut, displacement)
      : (displacement, point) => {
        const moved = matvec(
          frame.directionOut,
          displacement.map((value, i) => value + point[i] - frame.originOut[i]),
        ).map((value, i) => value + frame.originOut[i]);
        const rotated = matvec(
          frame.directionIn,
          point.map((value, i) => value - frame.originIn[i]),
        );
        return moved.map((value, i) => value - rotated[i] - frame.originIn[i]);
      };
  }

  if (
    transform.interpolation !== undefined &&
    transform.interpolation !== "linear"
  ) {
    console.warn(
      `the displacements transform asks for '${transform.interpolation}' ` +
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
