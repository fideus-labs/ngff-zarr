// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Resample bounding box support.
 *
 * This module provides conditional exports for browser and Node environments.
 * The actual implementation is delegated to environment-specific modules:
 * - resample_bounding_box-browser.ts: WebWorker-based functions
 * - resample_bounding_box-node.ts: native WASM for Node/Deno
 *
 * For Deno runtime, we default to the node implementation.
 * For browser bundlers, they should use conditional exports in package.json
 * to resolve to the browser implementation.
 */

export { resampleBoundingBox } from "./resample_bounding_box-node.ts";
export {
  ResampleBoundingBox,
  type ResampleBoundingBoxOptions,
} from "./resample_bounding_box-shared.ts";
