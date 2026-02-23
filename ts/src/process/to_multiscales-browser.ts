// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Browser-compatible toMultiscales implementation
 * Uses WebWorker-based ITK-Wasm downsampling via @itk-wasm/downsample browser exports
 */

import { NgffImage } from "../types/ngff_image.ts";
import { Multiscales } from "../types/multiscales.ts";
import { downsampleItkWasm } from "../methods/itkwasm-browser.ts";
import {
  toMultiscalesCore,
  type ToMultiscalesOptions,
} from "./to_multiscales-shared.ts";

// Re-export types for convenience
// Note: toNgffImage is NOT re-exported from browser version because it depends on I/O modules
// that have Node.js-specific imports. Use toNgffImage from the main mod.ts for Node.js/Deno.
export type { ToMultiscalesOptions } from "./to_multiscales-shared.ts";

/**
 * Generate multiple resolution scales for an NgffImage (browser version)
 *
 * @param image - Input NgffImage
 * @param options - Configuration options
 * @returns Multiscales object
 */
export async function toMultiscales(
  image: NgffImage,
  options: ToMultiscalesOptions = {},
): Promise<Multiscales> {
  return await toMultiscalesCore(image, options, downsampleItkWasm);
}
