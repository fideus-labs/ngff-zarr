// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * ITK-Wasm downsampling support for multiscale generation
 *
 * This module provides conditional exports for browser and Node environments.
 * The actual implementation is delegated to environment-specific modules:
 * - itkwasm-browser.ts: Uses WebWorker-based functions for browser environments
 * - itkwasm-node.ts: Uses native WASM functions for Node environments
 *
 * For browser bundlers, they should use conditional exports in package.json
 * to resolve to the browser implementation.
 */

// Default to the Node implementation
// Browser bundlers should use conditional exports to get itkwasm-browser.ts
export { downsampleItkWasm } from "./itkwasm-node.ts";
