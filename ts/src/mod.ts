// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

export { config, setWorkerPoolSize } from "./config.ts";
export * from "./io/from_ngff_zarr.ts";
export * from "./io/hcs.ts";
export * from "./io/itk_image_to_ngff_image.ts";
export * from "./io/itk_transform_resample_bounding_box.ts";
export * from "./io/ngff_image_to_itk_image.ts";
export type { MemoryStoreToZipOptions } from "./io/rfc9_zip.ts";
// RFC-9 exports
export {
  getZipFileCompressionMethod,
  getZipFileList,
  isOzxPath,
  memoryStoreToZip,
  orderFilesForRfc9,
  readOzxJsonFirst,
  readOzxVersion,
} from "./io/rfc9_zip.ts";
export * from "./io/to_ngff_zarr.ts";
// upgradeOmeZarr: spec-version upgrade (in-place metadata rewrite or
// write-to-new-store). In-place upgrade of a local *path* store is Node/Deno-
// only (needs a filesystem store); MemoryStore/FetchStore inputs work
// everywhere, consistent with the other I/O functions.
export {
  type UpgradeInput,
  upgradeOmeZarr,
  type UpgradeOmeZarrOptions,
} from "./io/upgrade_ome_zarr.ts";
export * from "./process/to_multiscales-node.ts";
export * from "./schemas/methods.ts";
export * from "./schemas/multiscales.ts";
export * from "./schemas/ngff_image.ts";
export * from "./schemas/units.ts";
export * from "./schemas/zarr_metadata.ts";
export * from "./types/array_interface.ts";
export * from "./types/hcs.ts";
export * from "./types/methods.ts";
export * from "./types/multiscales.ts";
export * from "./types/ngff_image.ts";
export * from "./types/rfc4.ts";
export * from "./types/supported_versions.ts";
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
export {
  computeOmeroFromMultiscales,
  computeOmeroFromNgffImage,
  getDefaultColors,
  GLASBEY_COLORS,
  terminateOmeroWorkerPool,
} from "./utils/compute_omero.ts";
export {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
  createNgffImage,
  createNgffMultiscales,
} from "./utils/factory.ts";
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
  fromZarrAttrsV04,
  fromZarrAttrsV05,
  fromZarrAttrsV06,
} from "./utils/from_zarr_attrs.ts";
export { getMethodMetadata } from "./utils/method_metadata.ts";
export {
  detectVersion,
  extractMethodMetadata,
  parseOmero,
} from "./utils/parse_metadata.ts";
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
