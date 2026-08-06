// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/** Browser implementation: dispatches through the package's web worker pool. */

import { resampleBoundingBox } from "@itk-wasm/downsample";
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
 * Browser counterpart of the Node implementation; see
 * `itk_transform_resample_bounding_box-node.ts` for the full description.
 */
export function itkTransformResampleBoundingBox(
  transform: TransformList,
  fixed: NgffImage,
  moving: NgffImage,
  options: ItkTransformResampleBoundingBoxOptions = {},
): Promise<ResampleBoundingBox> {
  return resampleBoundingBoxShared(
    resampleBoundingBox,
    transform,
    fixed,
    moving,
    options,
  );
}
