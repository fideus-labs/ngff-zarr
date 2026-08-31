// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Find the region of a moving image needed to resample a fixed image grid.
 *
 * Environment-agnostic core. The node and browser modules supply the
 * `resampleBoundingBox` pipeline; everything else lives here.
 */

import * as zarr from "zarrita";
import type { Image, TransformList } from "itk-wasm";
import { NgffImage } from "../types/ngff_image.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import type { NgffMultiscales } from "../types/multiscales.ts";
import { identityDirection, itkDirection } from "../utils/itk_direction.ts";
export { itkDirection };
import { ngffTransformToItkTransform } from "../utils/ngff_transform_to_itk_transform.ts";
import {
  checkUnorientedField,
  fieldDisplacementRange,
  fieldImage,
  fieldWindow,
} from "../utils/displacement_field_transform.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

/** The shape of the JSON the `resample-bounding-box` pipeline emits. */
interface RawBoundingBox {
  paddedStartIndex: number[];
  paddedSize: number[];
  paddedCorners: { min: number[]; max: number[] };
  corners: { min: number[]; max: number[] };
}

/** Options for {@link resampleBoundingBox}. */
export interface ResampleBoundingBoxOptions {
  /**
   * Pixels of padding added per side. The default of 1 covers linear
   * interpolation, which reads one neighbor beyond the continuous index
   * bound. Use 0 for the tight region, or more for wider kernels.
   */
  padding?: number;
  /**
   * The field images an RFC-5 `displacements` or `coordinates` transformation
   * points at, keyed by its `path`. Required for those two, ignored
   * otherwise. A field carrying an anatomical orientation is refused here,
   * since this branch works on the intrinsic systems where none applies:
   * convert it with `ngffDisplacementFieldToItkTransform`, passing `fixed`
   * and `moving`, and pass the transform list that returns.
   */
  fields?: Record<string, NgffImage | NgffMultiscales>;
}

/**
 * The region of a moving image needed to resample a fixed image grid.
 *
 * Every record is keyed by dimension name, so callers never have to track
 * whether a given array is in Zarr or ITK axis order.
 */
export class ResampleBoundingBox {
  /** Spatial dimension names, in the moving image's (Zarr) order. */
  readonly dims: string[];
  /**
   * Start of the region in the moving image's index space. May be negative
   * when the transformed grid extends past the moving image origin.
   */
  readonly startIndex: Record<string, number>;
  /** Size of the region in moving-image pixels, before clamping. */
  readonly size: Record<string, number>;
  /**
   * Tight extent of the transformed fixed grid, in moving-image physical
   * coordinates. Independent of `padding`.
   */
  readonly cornersMin: Record<string, number>;
  readonly cornersMax: Record<string, number>;
  /** Physical coordinates of the padded region's first and last pixel. */
  readonly paddedCornersMin: Record<string, number>;
  readonly paddedCornersMax: Record<string, number>;
  /** Shape of the moving image, used to clamp the region into bounds. */
  readonly movingShape: Record<string, number>;

  constructor(options: {
    dims: string[];
    startIndex: Record<string, number>;
    size: Record<string, number>;
    cornersMin: Record<string, number>;
    cornersMax: Record<string, number>;
    paddedCornersMin: Record<string, number>;
    paddedCornersMax: Record<string, number>;
    movingShape: Record<string, number>;
  }) {
    this.dims = [...options.dims];
    this.startIndex = { ...options.startIndex };
    this.size = { ...options.size };
    this.cornersMin = { ...options.cornersMin };
    this.cornersMax = { ...options.cornersMax };
    this.paddedCornersMin = { ...options.paddedCornersMin };
    this.paddedCornersMax = { ...options.paddedCornersMax };
    this.movingShape = { ...options.movingShape };
  }

  /**
   * The region intersected with the moving image bounds.
   *
   * @returns `{dim: [start, stop]}` with `0 <= start <= stop <= shape`.
   */
  clamped(): Record<string, [number, number]> {
    const bounds: Record<string, [number, number]> = {};
    for (const dim of this.dims) {
      const extent = this.movingShape[dim];
      const start = Math.max(0, Math.min(this.startIndex[dim], extent));
      const stop = Math.max(
        start,
        Math.min(this.startIndex[dim] + this.size[dim], extent),
      );
      bounds[dim] = [start, stop];
    }
    return bounds;
  }

  /** Whether the region does not overlap the moving image at all. */
  get isEmpty(): boolean {
    return Object.values(this.clamped()).some(([start, stop]) => stop <= start);
  }

  /**
   * A zarrita selection for the region, clamped to the moving image bounds.
   *
   * Negative start indices are clamped rather than passed through, since a
   * negative bound would silently wrap around instead of raising.
   *
   * @param dims Dimension order of the array to be read. Defaults to the
   *   spatial dims. Dimensions outside the region (`t`, `c`) select everything.
   */
  selection(dims?: string[]): (zarr.Slice | null)[] {
    const order = dims ?? this.dims;
    const bounds = this.clamped();
    return order.map((dim) =>
      dim in bounds ? zarr.slice(bounds[dim][0], bounds[dim][1]) : null
    );
  }

  /**
   * The translation the cropped region would have, per dimension.
   *
   * Reading `selection()` yields an array whose origin has moved; this is the
   * matching `translation` for an {@link NgffImage} built from it.
   */
  croppedTranslation(moving: NgffImage): Record<string, number> {
    const translation = { ...moving.translation };
    for (const [dim, [start]] of Object.entries(this.clamped())) {
      translation[dim] = moving.translation[dim] + start * moving.scale[dim];
    }
    return translation;
  }
}

function spatialDims(image: NgffImage): string[] {
  return image.dims.filter((dim) => SPATIAL_DIMS.includes(dim));
}

/**
 * Require a finite scale and translation for every spatial axis.
 *
 * A missing or non-finite entry would otherwise reach the pipeline and come
 * back as a plausible-looking region computed from the wrong geometry.
 */
function checkGeometry(
  label: string,
  image: NgffImage,
  spatial: string[],
): void {
  for (
    const [name, values] of [
      ["scale", image.scale],
      ["translation", image.translation],
    ] as const
  ) {
    for (const dim of spatial) {
      const value = values[dim];
      if (value === undefined) {
        throw new Error(
          `${label} image ${name} has no entry for dimension '${dim}'`,
        );
      }
      if (!Number.isFinite(value)) {
        throw new Error(
          `${label} image ${name} for dimension '${dim}' is ${value}; ` +
            `it must be finite`,
        );
      }
      if (name === "scale" && value === 0) {
        throw new Error(
          `${label} image scale for dimension '${dim}' is zero`,
        );
      }
    }
  }
}

/**
 * Reject a region that does not contain the points it was derived from.
 *
 * The pipeline computes the integer region in 32-bit index space while
 * reporting the corners as doubles, so a grid whose extent exceeds that range
 * comes back as a wrapped -- and typically empty -- region. Padding is
 * non-negative, so a correct region always spans its own tight corners;
 * checking that catches the wrap without assuming an axis direction.
 */
// The returned records are built from these two arrays alone, so a short or
// non-finite pipeline result would otherwise become an undefined bound that
// only surfaces once a caller slices with it.
function checkIndexArrays(raw: RawBoundingBox, itkDims: string[]): void {
  const arrays: [string, number[]][] = [
    ["paddedStartIndex", raw.paddedStartIndex],
    ["paddedSize", raw.paddedSize],
  ];
  for (const [name, values] of arrays) {
    if (values?.length !== itkDims.length) {
      throw new Error(
        `the pipeline reported ${values?.length ?? 0} ${name} values for ` +
          `${itkDims.length} dimensions`,
      );
    }
    itkDims.forEach((dim, axis) => {
      if (!Number.isFinite(values[axis])) {
        throw new Error(
          `the ${name} for dimension '${dim}' is not finite; check the ` +
            `transform and the image scale and translation`,
        );
      }
    });
  }
}

function checkRegionContainsCorners(
  raw: RawBoundingBox,
  itkDims: string[],
): void {
  itkDims.forEach((dim, axis) => {
    const values = [
      raw.corners.min[axis],
      raw.corners.max[axis],
      raw.paddedCorners.min[axis],
      raw.paddedCorners.max[axis],
    ];
    if (values.some((value) => value === null || !Number.isFinite(value))) {
      throw new Error(
        `the resample region for dimension '${dim}' is not finite; check ` +
          `the transform and the image scale and translation`,
      );
    }
    // paddedCorners hold the start and end index positions, which invert when
    // an axis direction is negative, so compare against the span.
    const low = Math.min(
      raw.paddedCorners.min[axis],
      raw.paddedCorners.max[axis],
    );
    const high = Math.max(
      raw.paddedCorners.min[axis],
      raw.paddedCorners.max[axis],
    );
    const tightLow = Math.min(raw.corners.min[axis], raw.corners.max[axis]);
    const tightHigh = Math.max(raw.corners.min[axis], raw.corners.max[axis]);
    if (tightLow < low || tightHigh > high) {
      throw new Error(
        `the resample region for dimension '${dim}' does not contain the ` +
          `transformed grid (${tightLow} to ${tightHigh} lies outside ` +
          `${low} to ${high}). The region spans more than the index range ` +
          `the pipeline can represent; check for a scale mismatch between ` +
          `the fixed and moving images.`,
      );
    }
  });
}

/**
 * Build an ITK-Wasm image carrying geometry only, with an empty buffer.
 *
 * The zarr array is never read -- only its `shape` is consulted. This is what
 * lets the region be computed against images whose pixels are remote or
 * enormous.
 */
export function metadataOnlyItkImage(
  image: NgffImage,
  itkDims: string[],
  direction: Float64Array,
): Image {
  const itkImage: Image = {
    imageType: {
      dimension: itkDims.length,
      componentType: "uint8",
      pixelType: "Scalar",
      components: 1,
    },
    name: image.name,
    origin: itkDims.map((dim) => image.translation[dim]),
    spacing: itkDims.map((dim) => image.scale[dim]),
    direction,
    size: itkDims.map((dim) => image.data.shape[image.dims.indexOf(dim)]),
    metadata: new Map(),
    data: new Uint8Array(0),
  };
  return itkImage;
}

/** The field `fields` holds for a field transform, with a message. */
function fieldFor(
  transform: { type: string; path: string },
  fields: Record<string, NgffImage | NgffMultiscales> | undefined,
): NgffImage | NgffMultiscales {
  const field = fields?.[transform.path];
  if (field === undefined) {
    const available = Object.keys(fields ?? {}).sort().join(", ");
    throw new Error(
      `the ${transform.type} transform points at '${transform.path}', but no ` +
        `field was passed for it (fields given: [${available}]). Load it ` +
        `with fromOmeZarr(\`\${store}/${transform.path}\`) and pass ` +
        `{ fields: { "${transform.path}": field } }.`,
    );
  }
  return field;
}

function isV06Transform(value: unknown): value is V06Transform {
  return typeof value === "object" && value !== null && "type" in value &&
    typeof (value as { type: unknown }).type === "string";
}

/**
 * Compute the moving-image region needed to resample a fixed image grid.
 *
 * @param pipeline The environment's `resampleBoundingBox` implementation.
 */
export async function resampleBoundingBoxShared(
  pipeline: (
    transform: TransformList,
    fixed: Image,
    moving: Image,
    options: { padding?: number },
  ) => Promise<{ boundingBox: unknown }>,
  transform: V06Transform | TransformList,
  fixed: NgffImage,
  moving: NgffImage,
  options: ResampleBoundingBoxOptions = {},
): Promise<ResampleBoundingBox> {
  const padding = options.padding ?? 1;
  if (!Number.isInteger(padding) || padding < 0) {
    throw new Error(`padding must be a non-negative integer, got ${padding}`);
  }

  const fixedSpatial = spatialDims(fixed);
  const movingSpatial = spatialDims(moving);
  if (fixedSpatial.join(",") !== movingSpatial.join(",")) {
    throw new Error(
      `fixed image has spatial dims [${fixedSpatial.join(", ")}] but moving ` +
        `image has [${movingSpatial.join(", ")}]; they must match`,
    );
  }
  if (fixedSpatial.length < 2) {
    throw new Error(
      `images have ${fixedSpatial.length} spatial dims ` +
        `[${fixedSpatial.join(", ")}]; only 2 and 3 are supported`,
    );
  }

  checkGeometry("fixed", fixed, fixedSpatial);
  checkGeometry("moving", moving, fixedSpatial);

  // ITK orders points fastest-axis-first by name: x, then y, then z.
  // Reversing the dims is only right for the canonical (z, y, x).
  const itkDims = SPATIAL_DIMS.filter((dim) => fixedSpatial.includes(dim));

  const movingShape: Record<string, number> = {};
  for (const dim of movingSpatial) {
    movingShape[dim] = moving.data.shape[moving.dims.indexOf(dim)];
  }
  const fixedDegenerate = fixedSpatial.some(
    (dim) => fixed.data.shape[fixed.dims.indexOf(dim)] === 0,
  );
  if (fixedDegenerate) {
    // A grid with a zero-length axis has no samples to resample, so there is
    // nothing to fetch.
    const zeros: Record<string, number> = {};
    for (const dim of fixedSpatial) zeros[dim] = 0;
    return new ResampleBoundingBox({
      dims: fixedSpatial,
      startIndex: { ...zeros },
      size: { ...zeros },
      cornersMin: { ...zeros },
      cornersMax: { ...zeros },
      paddedCornersMin: { ...zeros },
      paddedCornersMax: { ...zeros },
      movingShape,
    });
  }

  let transformList: TransformList;
  let fixedDirection: Float64Array;
  let movingDirection: Float64Array;
  let fieldRange: { low: number[]; high: number[] } | undefined;

  if (Array.isArray(transform)) {
    // An ITK transform list acts on ITK physical space, so the geometry is
    // built the way ngffImageToItkImage builds it, direction included.
    transformList = transform;
    fixedDirection = itkDirection(fixed, itkDims);
    movingDirection = itkDirection(moving, itkDims);
  } else if (isV06Transform(transform)) {
    if (
      transform.type === "displacements" || transform.type === "coordinates"
    ) {
      // A field transform is the identity plus a displacement. The identity
      // is what the pipeline measures; the displacement is read off the field
      // a chunk at a time and widens the region afterwards, so the field is
      // never held whole and a bump inside the grid is not walked past.
      const image = fieldImage(
        transform,
        fieldFor(transform, options.fields),
        fixedSpatial,
      );
      checkUnorientedField(image, fixedSpatial);
      const { window, outside } = fieldWindow(
        image,
        fixedSpatial,
        fixed.translation,
        fixed.scale,
        fixedSpatial.map((dim) => fixed.data.shape[fixed.dims.indexOf(dim)]),
      );
      fieldRange = await fieldDisplacementRange(
        transform,
        image,
        fixedSpatial,
        window,
        outside,
      );
      transformList = ngffTransformToItkTransform(
        { type: "identity" },
        fixed.dims,
      );
    } else {
      transformList = ngffTransformToItkTransform(transform, fixed.dims);
    }
    // An RFC-5 transformation is defined on the intrinsic coordinate system,
    // which carries no direction matrix.
    fixedDirection = identityDirection(itkDims.length);
    movingDirection = identityDirection(itkDims.length);
  } else {
    throw new Error(
      `unsupported transform type. Expected an RFC-5 coordinate ` +
        `transformation or an ITK-Wasm TransformList.`,
    );
  }

  const { boundingBox } = await pipeline(
    transformList,
    metadataOnlyItkImage(fixed, itkDims, fixedDirection),
    metadataOnlyItkImage(moving, itkDims, movingDirection),
    { padding },
  );
  const raw = boundingBox as RawBoundingBox;
  checkIndexArrays(raw, itkDims);
  checkRegionContainsCorners(raw, itkDims);

  // The pipeline reports arrays fastest-axis-first; key them by dimension
  // name so no caller has to remember that, and order the records like `dims`
  // so a serialized result reads in Zarr order.
  const byDim = (values: number[]): Record<string, number> => {
    const indexed: Record<string, number> = {};
    itkDims.forEach((dim, index) => {
      indexed[dim] = values[index];
    });
    const record: Record<string, number> = {};
    for (const dim of fixedSpatial) record[dim] = indexed[dim];
    return record;
  };

  const region = new ResampleBoundingBox({
    dims: fixedSpatial,
    startIndex: byDim(raw.paddedStartIndex),
    size: byDim(raw.paddedSize),
    cornersMin: byDim(raw.corners.min),
    cornersMax: byDim(raw.corners.max),
    paddedCornersMin: byDim(raw.paddedCorners.min),
    paddedCornersMax: byDim(raw.paddedCorners.max),
    movingShape,
  });
  return fieldRange === undefined
    ? region
    : grown(region, fieldRange, fixedSpatial, moving, movingShape);
}

/**
 * `region` moved and widened by a per-axis displacement range.
 *
 * The pipeline walks the boundary of the transformed grid, which reports
 * where a *linear* map sends the grid exactly and misses whatever a
 * displacement does strictly inside it. A point `p` of the region reaches
 * `p + d` with `d` in the range, so the region's image lies between the two
 * ends of that range: a field that shifts every point the same way moves the
 * region, and only what varies widens it.
 */
function grown(
  region: ResampleBoundingBox,
  range: { low: number[]; high: number[] },
  spatial: string[],
  moving: NgffImage,
  movingShape: Record<string, number>,
): ResampleBoundingBox {
  const startIndex = { ...region.startIndex };
  const size = { ...region.size };
  const cornersMin = { ...region.cornersMin };
  const cornersMax = { ...region.cornersMax };
  const paddedCornersMin = { ...region.paddedCornersMin };
  const paddedCornersMax = { ...region.paddedCornersMax };
  spatial.forEach((dim, axis) => {
    const low = range.low[axis];
    const high = range.high[axis];
    if (low === 0 && high === 0) return;
    const scale = moving.scale[dim];
    const first = Math.floor(Math.min(low / scale, high / scale));
    const last = Math.ceil(Math.max(low / scale, high / scale));
    startIndex[dim] += first;
    size[dim] += last - first;
    // The reported corners hold a first and a last index position, which swap
    // when an axis direction is negative; move the span either way.
    for (
      const [begin, stop] of [
        [cornersMin, cornersMax],
        [paddedCornersMin, paddedCornersMax],
      ] as Record<string, number>[][]
    ) {
      if (begin[dim] <= stop[dim]) {
        begin[dim] += low;
        stop[dim] += high;
      } else {
        begin[dim] += high;
        stop[dim] += low;
      }
    }
  });
  return new ResampleBoundingBox({
    dims: region.dims,
    startIndex,
    size,
    cornersMin,
    cornersMax,
    paddedCornersMin,
    paddedCornersMax,
    movingShape,
  });
}
