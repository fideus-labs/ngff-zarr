// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";

// Version schemas for different OME-Zarr specification versions
export const OmeZarrVersion01Schema = z.literal("0.1");
export const OmeZarrVersion02Schema = z.literal("0.2");
export const OmeZarrVersion03Schema = z.literal("0.3");
export const OmeZarrVersion04Schema = z.literal("0.4");
export const OmeZarrVersion05Schema = z.literal("0.5");

export const OmeZarrVersionSchema = z.union([
  OmeZarrVersion01Schema,
  OmeZarrVersion02Schema,
  OmeZarrVersion03Schema,
  OmeZarrVersion04Schema,
  OmeZarrVersion05Schema,
]);

// Common schemas used across versions
export const OmeroWindowSchema = z.object({
  start: z.number(),
  min: z.number(),
  end: z.number(),
  max: z.number(),
});

export const OmeroChannelSchema = z.object({
  window: OmeroWindowSchema,
  label: z.string().optional(),
  family: z.string().optional(),
  color: z.string(),
  active: z.boolean().optional(),
});

export const OmeroChannelSchemaV01 = z.object({
  window: OmeroWindowSchema,
  label: z.string().optional(),
  family: z.string().optional(),
  color: z.string(),
  active: z.boolean().optional(),
});

export const OmeroSchema = z.object({
  channels: z.array(OmeroChannelSchema),
});

// Dataset schemas for different versions
export const DatasetSchemaV01 = z.object({
  path: z.string(),
});

export const OmeZarrCoordinateTransformationSchema = z.union([
  z.object({
    type: z.literal("scale"),
    scale: z.array(z.number()).min(2),
  }),
  z.object({
    type: z.literal("translation"),
    translation: z.array(z.number()).min(2),
  }),
]);

export const DatasetSchemaV05 = z.object({
  path: z.string(),
  coordinateTransformations: z
    .array(OmeZarrCoordinateTransformationSchema)
    .min(1),
});

// Axis schemas for different versions
export const AxisSchemaV05 = z.union([
  z.object({
    name: z.string(),
    type: z.enum(["channel", "time", "space"]),
    unit: z.string().optional(),
  }),
  z.object({
    name: z.string(),
    type: z
      .string()
      .refine((val) => !["space", "time", "channel"].includes(val)),
  }),
]);

// Multiscales schemas for different versions
export const MultiscalesSchemaV01 = z.object({
  name: z.string().optional(),
  datasets: z.array(DatasetSchemaV01).min(1),
  version: OmeZarrVersion01Schema.optional(),
  metadata: z
    .object({
      method: z.string().optional(),
      version: z.string().optional(),
    })
    .optional(),
});

export const MultiscalesSchemaV05 = z.object({
  name: z.string().optional(),
  datasets: z.array(DatasetSchemaV05).min(1),
  axes: z
    .array(AxisSchemaV05)
    .min(2)
    .max(5)
    .refine(
      (axes) => {
        const spaceAxes = axes.filter(
          (axis) => typeof axis.type === "string" && axis.type === "space",
        );
        return spaceAxes.length >= 2 && spaceAxes.length <= 3;
      },
      {
        message: "Must contain 2-3 space axes",
      },
    ),
  coordinateTransformations: z
    .array(OmeZarrCoordinateTransformationSchema)
    .optional(),
});

// Image schemas for different versions
export const ImageSchemaV01 = z.object({
  multiscales: z.array(MultiscalesSchemaV01).min(1),
  omero: OmeroSchema.optional(),
});

export const ImageSchemaV05 = z.object({
  ome: z.object({
    multiscales: z.array(MultiscalesSchemaV05).min(1),
    omero: OmeroSchema.optional(),
    version: OmeZarrVersion05Schema,
  }),
});

// Label schemas
export const LabelColorSchema = z.object({
  "label-value": z.number(),
  rgba: z.array(z.number().min(0).max(255)).length(4).optional(),
});

export const LabelPropertySchema = z.object({
  "label-value": z.number(),
});

export const LabelSchemaV04 = z.object({
  "image-label": z
    .object({
      colors: z.array(LabelColorSchema).min(1).optional(),
      properties: z.array(LabelPropertySchema).min(1).optional(),
      source: z
        .object({
          image: z.string().optional(),
        })
        .optional(),
      version: OmeZarrVersion04Schema.optional(),
    })
    .optional(),
});

// Plate schemas
export const PlateAcquisitionSchema = z.object({
  id: z.number().min(0),
  maximumfieldcount: z.number().positive().optional(),
  name: z.string().optional(),
  description: z.string().optional(),
  starttime: z.number().min(0).optional(),
  endtime: z.number().min(0).optional(),
});

export const PlateColumnSchema = z.object({
  name: z.string().regex(/^[A-Za-z0-9]+$/),
});

export const PlateRowSchema = z.object({
  name: z.string().regex(/^[A-Za-z0-9]+$/),
});

export const PlateWellSchema = z.object({
  path: z.string().regex(/^[A-Za-z0-9]+\/[A-Za-z0-9]+$/),
  rowIndex: z.number().min(0),
  columnIndex: z.number().min(0),
});

export const PlateSchemaV05 = z.object({
  ome: z.object({
    plate: z.object({
      acquisitions: z.array(PlateAcquisitionSchema).optional(),
      field_count: z.number().positive().optional(),
      name: z.string().optional(),
      columns: z.array(PlateColumnSchema).min(1),
      rows: z.array(PlateRowSchema).min(1),
      wells: z.array(PlateWellSchema).min(1),
    }),
    version: OmeZarrVersion05Schema,
  }),
});

// Well schemas
export const WellImageSchema = z.object({
  acquisition: z.number().optional(),
  path: z.string().regex(/^[A-Za-z0-9]+$/),
});

export const WellSchemaV05 = z.object({
  ome: z.object({
    well: z.object({
      images: z.array(WellImageSchema).min(1),
    }),
    version: OmeZarrVersion05Schema,
  }),
});

// OME metadata schema (for ome.schema)
export const OmeSeriesSchema = z.object({
  image: z.string(),
});

export const OmeMetadataSchema = z.object({
  ome: z.object({
    series: z.array(OmeSeriesSchema),
    version: OmeZarrVersionSchema,
  }),
});

// Union schemas for version-agnostic usage
export const ImageSchema = z.union([ImageSchemaV01, ImageSchemaV05]);

export const MultiscalesSchema = z.union([
  MultiscalesSchemaV01,
  MultiscalesSchemaV05,
]);

// Comprehensive OME-Zarr metadata schema that can handle any version
export const OmeZarrMetadataSchema = z.union([
  ImageSchemaV01,
  ImageSchemaV05,
  LabelSchemaV04,
  PlateSchemaV05,
  WellSchemaV05,
  OmeMetadataSchema,
]);

// Type exports
export type OmeZarrVersion = z.infer<typeof OmeZarrVersionSchema>;
export type OmeroWindow = z.infer<typeof OmeroWindowSchema>;
export type OmeroChannel = z.infer<typeof OmeroChannelSchema>;
export type Omero = z.infer<typeof OmeroSchema>;
export type CoordinateTransformation = z.infer<
  typeof OmeZarrCoordinateTransformationSchema
>;
export type DatasetV01 = z.infer<typeof DatasetSchemaV01>;
export type DatasetV05 = z.infer<typeof DatasetSchemaV05>;
export type AxisV05 = z.infer<typeof AxisSchemaV05>;
export type MultiscalesV01 = z.infer<typeof MultiscalesSchemaV01>;
export type MultiscalesV05 = z.infer<typeof MultiscalesSchemaV05>;
export type ImageV01 = z.infer<typeof ImageSchemaV01>;
export type ImageV05 = z.infer<typeof ImageSchemaV05>;
export type LabelColor = z.infer<typeof LabelColorSchema>;
export type LabelProperty = z.infer<typeof LabelPropertySchema>;
export type LabelV04 = z.infer<typeof LabelSchemaV04>;
export type PlateAcquisition = z.infer<typeof PlateAcquisitionSchema>;
export type PlateColumn = z.infer<typeof PlateColumnSchema>;
export type PlateRow = z.infer<typeof PlateRowSchema>;
export type PlateWell = z.infer<typeof PlateWellSchema>;
export type PlateV05 = z.infer<typeof PlateSchemaV05>;
export type WellImage = z.infer<typeof WellImageSchema>;
export type WellV05 = z.infer<typeof WellSchemaV05>;
export type OmeSeries = z.infer<typeof OmeSeriesSchema>;
export type OmeMetadata = z.infer<typeof OmeMetadataSchema>;
export type Image = z.infer<typeof ImageSchema>;
export type Multiscales = z.infer<typeof MultiscalesSchema>;
export type OmeZarrMetadata = z.infer<typeof OmeZarrMetadataSchema>;
