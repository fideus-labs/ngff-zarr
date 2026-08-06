// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/** Node/Deno implementation: native WASM, no web worker. */

import { resampleBoundingBoxNode } from "@itk-wasm/downsample";
import type { TransformList } from "itk-wasm";
import type { NgffImage } from "../types/ngff_image.ts";
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
 * The transform acts on ITK physical space, so the geometry is built the way
 * {@link ngffImageToItkImage} builds it, including the direction matrix derived
 * from RFC-4 anatomical orientation. It maps *fixed* points into *moving*
 * space.
 *
 * @param transform An ITK-Wasm `TransformList`.
 * @param fixed The image whose grid is resampled. Geometry only.
 * @param moving The image to be sampled. Geometry only.
 * @param options Padding options.
 * @returns The region, keyed by dimension name in Zarr order.
 */
export function itkTransformResampleBoundingBox(
  transform: TransformList,
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
