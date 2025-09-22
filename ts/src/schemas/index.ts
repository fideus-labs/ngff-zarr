// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Re-export all schemas with explicit naming to avoid conflicts
export * from "./units.ts";
export * from "./methods.ts";

// RFC4 schemas
export {
  type AnatomicalOrientation,
  AnatomicalOrientationSchema,
  // Type exports
  type AnatomicalOrientationValues,
  AnatomicalOrientationValuesSchema,
  type Axes as RFC4Axes,
  type AxesNames,
  AxesNamesSchema,
  AxesSchema as RFC4AxesSchema,
  type Axis as RFC4Axis,
  AxisSchema as RFC4AxisSchema,
  type AxisType,
  AxisTypeSchema,
  type BaseAxis,
  BaseAxisSchema,
  type ChannelAxis,
  ChannelAxisSchema,
  type Orientation,
  OrientationSchema,
  type SpaceAxesNames,
  SpaceAxesNamesSchema,
  type SpaceAxis,
  SpaceAxisSchema,
  type SpaceUnit,
  SpaceUnitSchema,
  type TimeAxis,
  TimeAxisSchema,
  type TimeUnit,
  TimeUnitSchema,
} from "./rfc4.ts";

// Coordinate systems (RFC5)
export * from "./coordinate_systems.ts";

// Zarr metadata schemas (with RFC4 integration)
export {
  AxisSchema,
  DatasetSchema,
  IdentitySchema,
  MetadataSchema,
  OmeroChannelSchema,
  OmeroSchema,
  OmeroWindowSchema,
  ScaleSchema,
  TransformSchema,
  TranslationSchema,
} from "./zarr_metadata.ts";

// OME-Zarr specification schemas
export {
  // Axis schemas
  AxisSchemaV05,
  type AxisV05,
  type CoordinateTransformation as OmeZarrCoordinateTransformation,
  // Dataset schemas
  DatasetSchemaV01,
  DatasetSchemaV05,
  type DatasetV01,
  type DatasetV05,
  type Image,
  ImageSchema,
  // Image schemas
  ImageSchemaV01,
  ImageSchemaV05,
  type ImageV01,
  type ImageV05,
  type LabelColor,
  // Label schemas
  LabelColorSchema,
  type LabelProperty,
  LabelPropertySchema,
  LabelSchemaV04,
  type LabelV04,
  type Multiscales,
  MultiscalesSchema,
  // Multiscales schemas
  MultiscalesSchemaV01,
  MultiscalesSchemaV05,
  type MultiscalesV01,
  type MultiscalesV05,
  type OmeMetadata,
  OmeMetadataSchema,
  type Omero as OmeZarrOmero,
  type OmeroChannel as OmeZarrOmeroChannel,
  OmeroChannelSchema as OmeZarrOmeroChannelSchema,
  OmeroSchema as OmeZarrOmeroSchema,
  type OmeroWindow as OmeZarrOmeroWindow,
  // Core schemas
  OmeroWindowSchema as OmeZarrOmeroWindowSchema,
  type OmeSeries,
  // OME metadata schemas
  OmeSeriesSchema,
  OmeZarrCoordinateTransformationSchema,
  type OmeZarrMetadata,
  // Comprehensive schema
  OmeZarrMetadataSchema,
  // Type exports
  type OmeZarrVersion,
  OmeZarrVersion01Schema,
  OmeZarrVersion02Schema,
  OmeZarrVersion03Schema,
  OmeZarrVersion04Schema,
  OmeZarrVersion05Schema,
  // Version schemas
  OmeZarrVersionSchema,
  type PlateAcquisition,
  // Plate schemas
  PlateAcquisitionSchema,
  type PlateColumn,
  PlateColumnSchema,
  type PlateRow,
  PlateRowSchema,
  PlateSchemaV05,
  type PlateV05,
  type PlateWell,
  PlateWellSchema,
  type WellImage,
  // Well schemas
  WellImageSchema,
  WellSchemaV05,
  type WellV05,
} from "./ome_zarr.ts";

// NGFF Image and Multiscales
export * from "./ngff_image.ts";
export * from "./multiscales.ts";
