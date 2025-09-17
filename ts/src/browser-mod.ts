// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible module exports
// This module excludes I/O functionality (from_ngff_zarr, to_ngff_zarr)
// because those modules depend on Node.js/Deno-specific filesystem APIs
// that are not available in browser environments.
//
// For browser use cases, the schemas, types, and validation utilities
// are the most commonly needed functionality.
export * from "./types/units.ts";
export * from "./types/methods.ts";
export * from "./types/array_interface.ts";
export * from "./types/zarr_metadata.ts";
export * from "./types/ngff_image.ts";
export * from "./types/multiscales.ts";

export * from "./schemas/units.ts";
export * from "./schemas/methods.ts";
export * from "./schemas/zarr_metadata.ts";
export * from "./schemas/ngff_image.ts";
export * from "./schemas/multiscales.ts";

export {
  isValidDimension,
  isValidUnit,
  validateMetadata,
} from "./utils/validation.ts";
export {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
  createNgffImage,
} from "./utils/factory.ts";

// Note: Excluding I/O modules for browser compatibility
