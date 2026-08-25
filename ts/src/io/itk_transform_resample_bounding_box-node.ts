// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/** Node/Deno implementation: native WASM, no web worker. */

import { resampleBoundingBoxNode } from "@itk-wasm/downsample";
import type { TransformList } from "itk-wasm";
import type { NgffImage } from "../types/ngff_image.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import {
  type ItkTransformResampleBoundingBoxOptions,
  type ResampleBoundingBox,
  resampleBoundingBoxShared,
} from "./itk_transform_resample_bounding_box-shared.ts";

/**
 * Compute the moving-image region needed to resample a fixed image grid.
 *
 * The region is derived from image geometry alone -- the pixel buffers of
 * `fixed` and `moving` are never read. That is the point: describe two images
 * and a transform with a few numbers, learn exactly which block of the moving
 * image a resample will touch, and only then move pixels.
 *
 * Two kinds of transform are accepted, and they are interpreted in different
 * coordinate spaces:
 *
 * - An **RFC-5 coordinate transformation** acts on the intrinsic coordinate
 *   system, where a point is `translation + scale * index`. Its parameters are
 *   in Zarr axis order and no direction matrix applies.
 * - An **ITK-Wasm `TransformList`** acts on ITK physical space, so the
 *   geometry is built the way {@link ngffImageToItkImage} builds it, including
 *   the direction matrix derived from RFC-4 anatomical orientation.
 *
 * In both cases the transform maps *fixed* points into *moving* space. An ITK
 * transform need not be linear; an RFC-5 transformation is converted first, so
 * it must be a linear mapping or a `displacements` transformation whose field
 * is passed in `options.fields`.
 *
 * @param transform An RFC-5 coordinate transformation or an ITK-Wasm
 *   `TransformList`.
 * @param fixed The image whose grid is resampled. Geometry only.
 * @param moving The image to be sampled. Geometry only.
 * @param options `padding`, and `fields` for a `displacements` transformation.
 * @returns The region, keyed by dimension name in Zarr order.
 */
export function itkTransformResampleBoundingBox(
  transform: V06Transform | TransformList,
  fixed: NgffImage,
  moving: NgffImage,
  options: ItkTransformResampleBoundingBoxOptions = {},
): Promise<ResampleBoundingBox> {
  return resampleBoundingBoxShared(
    resampleBoundingBoxNode,
    transform,
    fixed,
    moving,
    options,
  );
}
