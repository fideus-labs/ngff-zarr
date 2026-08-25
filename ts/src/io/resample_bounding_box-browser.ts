// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/** Browser implementation: dispatches through the package's web worker pool. */

import { resampleBoundingBox as resampleBoundingBoxWasm } from "@itk-wasm/downsample";
import type { TransformList } from "itk-wasm";
import type { NgffImage } from "../types/ngff_image.ts";
import type { V06Transform } from "../types/zarr_metadata.ts";
import {
  type ResampleBoundingBox,
  type ResampleBoundingBoxOptions,
  resampleBoundingBoxShared,
} from "./resample_bounding_box-shared.ts";

/**
 * Compute the moving-image region needed to resample a fixed image grid.
 *
 * Browser counterpart of the Node implementation; see
 * `resample_bounding_box-node.ts` for the full description.
 */
export function resampleBoundingBox(
  transform: V06Transform | TransformList,
  fixed: NgffImage,
  moving: NgffImage,
  options: ResampleBoundingBoxOptions = {},
): Promise<ResampleBoundingBox> {
  return resampleBoundingBoxShared(
    resampleBoundingBoxWasm,
    transform,
    fixed,
    moving,
    options,
  );
}
