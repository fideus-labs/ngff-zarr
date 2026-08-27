// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible module exports
// This module includes browser versions of fromNgffZarr and toNgffZarr.
//
// The browser versions support MemoryStore (Map) and HTTP/HTTPS URLs (read-only),
// but not local file paths (which require Node.js/Deno filesystem APIs).

export { config, setWorkerPoolSize } from "./config.ts";
// Browser-compatible I/O modules
// Note: Uses browser-specific versions that don't import @zarrita/storage
// (which contains Node.js-specific modules like node:fs, node:buffer, node:path)
export {
  type ChunkCache,
  fromNgffZarr,
  type FromNgffZarrOptions,
  fromOmeZarr,
  type FromOmeZarrOptions,
  type MemoryStore,
} from "./io/from_ngff_zarr-browser.ts";
// ITK-Wasm conversion utilities
export {
  itkImageToNgffImage,
  type ItkImageToNgffImageOptions,
} from "./io/itk_image_to_ngff_image.ts";
export { resampleBoundingBox } from "./io/resample_bounding_box-browser.ts";
export {
  ResampleBoundingBox,
  type ResampleBoundingBoxOptions,
} from "./io/resample_bounding_box-shared.ts";
export {
  itkTransformToNgffMatrix,
  itkTransformToNgffTransform,
  type NgffMatrixAndOffset,
} from "./utils/itk_transform_to_ngff_transform.ts";
export { ngffTransformToItkTransform } from "./utils/ngff_transform_to_itk_transform.ts";
export {
  type FieldFrames,
  type ItkDisplacementFieldOptions,
  itkDisplacementFieldToNgffTransform,
  type NgffDisplacementField,
  ngffDisplacementFieldToItkTransform,
} from "./utils/displacement_field_transform.ts";
export {
  dataTypeToComponentType,
  ngffImageToItkImage,
  type NgffImageToItkImageOptions,
} from "./io/ngff_image_to_itk_image.ts";
export type { MemoryStoreToZipOptions } from "./io/rfc9_zip.ts";
// RFC-9 exports
export {
  getZipFileCompressionMethod,
  getZipFileList,
  memoryStoreToZip,
  orderFilesForRfc9,
  readOzxVersion,
} from "./io/rfc9_zip.ts";
export {
  isOzxPath,
  toNgffZarr,
  type ToNgffZarrOptions,
  toNgffZarrOzx,
  type ToNgffZarrOzxOptions,
  toOmeZarr,
  type ToOmeZarrOptions,
  toOmeZarrOzx,
  type ToOmeZarrOzxOptions,
} from "./io/to_ngff_zarr-browser.ts";
// upgradeOmeZarr: spec-version upgrade (in-place metadata rewrite or
// write-to-new-store). In the browser only MemoryStore inputs/outputs are
// supported; in-place upgrade of a local *path* store is Node/Deno-only.
export {
  type UpgradeInput,
  upgradeOmeZarr,
  type UpgradeOmeZarrOptions,
} from "./io/upgrade_ome_zarr-browser.ts";
// Browser-compatible processing modules
export * from "./process/to_multiscales-browser.ts";
export * from "./schemas/methods.ts";
export * from "./schemas/multiscales.ts";
export * from "./schemas/ngff_image.ts";
export * from "./schemas/units.ts";
export * from "./schemas/zarr_metadata.ts";
export * from "./types/array_interface.ts";
export * from "./types/methods.ts";
export * from "./types/multiscales.ts";
export * from "./types/ngff_image.ts";
export * from "./types/units.ts";
export * from "./types/zarr_metadata.ts";
export type { CodecName, ZarrCodec } from "./utils/codecs.ts";
export {
  AVAILABLE_CODECS,
  bytesOnlyCodecs,
  codecFromName,
  defaultCodecs,
} from "./utils/codecs.ts";
export type {
  ComputeOmeroFromMultiscalesOptions,
  ComputeOmeroOptions,
} from "./utils/compute_omero.ts";
// OMERO computation with WorkerPool
export {
  computeOmeroFromMultiscales,
  computeOmeroFromNgffImage,
  getDefaultColors,
  GLASBEY_COLORS,
  terminateOmeroWorkerPool,
} from "./utils/compute_omero.ts";
export type { ChannelStatisticsAccumulator } from "./utils/compute_omero-shared.ts";
// Shared OMERO computation functions for benchmarking and direct use
export {
  buildOmeroFromAccumulators,
  computeChannelStatistics,
  createAccumulator,
  extractChannel,
  finalizeStatistics,
  QUANTILE_SAMPLE_SIZE,
  updateAccumulator,
  validateQuantiles,
} from "./utils/compute_omero-shared.ts";
export {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
  createNgffImage,
  createNgffMultiscales,
} from "./utils/factory.ts";
export { getMethodMetadata } from "./utils/method_metadata.ts";
export {
  SpecRule,
  type ValidateOptions,
  validatePlate,
  validateStructural,
  validateWell,
  ValidationError,
  ValidationLevel,
} from "./utils/structural_validation.ts";
export {
  isValidDimension,
  isValidUnit,
  validateMetadata,
} from "./utils/validation.ts";
export { terminateWorkerPool, zarrGet, zarrSet } from "./utils/worker_pool.ts";
