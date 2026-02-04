// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible module exports
// This module includes browser versions of fromNgffZarr and toNgffZarr.
//
// The browser versions support MemoryStore (Map) and HTTP/HTTPS URLs (read-only),
// but not local file paths (which require Node.js/Deno filesystem APIs).
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
export { getMethodMetadata } from "./utils/method_metadata.ts";
export {
  computeOmeroFromMultiscales,
  computeOmeroFromNgffImage,
  getDefaultColors,
  GLASBEY_COLORS,
} from "./utils/compute_omero.ts";
export type {
  ComputeOmeroFromMultiscalesOptions,
  ComputeOmeroOptions,
} from "./utils/compute_omero.ts";

// Browser-compatible I/O modules
// Note: Uses browser-specific versions that don't import @zarrita/storage
// (which contains Node.js-specific modules like node:fs, node:buffer, node:path)
export {
  fromNgffZarr,
  type FromNgffZarrOptions,
  type MemoryStore,
} from "./io/from_ngff_zarr-browser.ts";
export {
  isOzxPath,
  toNgffZarr,
  type ToNgffZarrOptions,
  toNgffZarrOzx,
  type ToNgffZarrOzxOptions,
} from "./io/to_ngff_zarr-browser.ts";

// RFC-9 exports
export {
  getZipFileCompressionMethod,
  getZipFileList,
  memoryStoreToZip,
  orderFilesForRfc9,
  readOzxVersion,
} from "./io/rfc9_zip.ts";
export type { MemoryStoreToZipOptions } from "./io/rfc9_zip.ts";

// Browser-compatible processing modules
export * from "./process/to_multiscales-browser.ts";

// ITK-Wasm conversion utilities
export {
  itkImageToNgffImage,
  type ItkImageToNgffImageOptions,
} from "./io/itk_image_to_ngff_image.ts";
