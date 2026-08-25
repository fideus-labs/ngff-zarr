// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";
import { AxisSchema } from "./rfc4.ts";

// RFC5: Coordinate Systems and Transformations

// Coordinate System schema
export const CoordinateSystemSchema = z.object({
  name: z.string().min(1), // MUST be non-empty and unique
  axes: z.array(AxisSchema), // Array of axes that define the coordinate system
});

// Identity transformation
export const IdentityTransformationSchema: z.ZodType<{
  type: "identity";
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("identity"),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Map Axis transformation: an axis permutation stored as a transpose vector
// of zero-based integer indices, each appearing exactly once.
export const MapAxisTransformationSchema: z.ZodType<{
  type: "mapAxis";
  mapAxis: number[];
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("mapAxis"),
  mapAxis: z
    .array(z.number().int().nonnegative())
    .min(2)
    .max(5)
    .refine(
      (indices) =>
        [...indices].sort((a, b) => a - b).every(
          (value, position) => value === position,
        ),
      {
        message:
          "mapAxis must be a permutation holding every zero-based input " +
          "axis index exactly once",
      },
    ),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Project Axis transformation: drops input axes and inserts zero-valued output
// axes. At least one of droppedInputs and createdOutputs is given, and the
// indices in each are unique.
export const ProjectAxisTransformationSchema: z.ZodType<{
  type: "projectAxis";
  droppedInputs?: number[] | undefined;
  createdOutputs?: number[] | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z
  .object({
    type: z.literal("projectAxis"),
    droppedInputs: z
      .array(z.number().int().nonnegative())
      .min(1)
      .max(3)
      .refine((indices) => new Set(indices).size === indices.length, {
        message: "droppedInputs indices must be unique",
      })
      .optional(),
    createdOutputs: z
      .array(z.number().int().nonnegative())
      .min(1)
      .max(3)
      .refine((indices) => new Set(indices).size === indices.length, {
        message: "createdOutputs indices must be unique",
      })
      .optional(),
    input: z.union([z.string(), z.array(z.string())]).optional(),
    output: z.union([z.string(), z.array(z.string())]).optional(),
    name: z.string().optional(),
  })
  .refine(
    (transform) =>
      transform.droppedInputs !== undefined ||
      transform.createdOutputs !== undefined,
    {
      message:
        "projectAxis must declare droppedInputs, createdOutputs, or both",
    },
  );

// Translation transformation
export const TranslationTransformationSchema: z.ZodType<{
  type: "translation";
  translation?: number[] | undefined;
  path?: string | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z
  .object({
    type: z.literal("translation"),
    translation: z.array(z.number()).optional(),
    path: z.string().optional(), // For binary data
    input: z.union([z.string(), z.array(z.string())]).optional(),
    output: z.union([z.string(), z.array(z.string())]).optional(),
    name: z.string().optional(),
  })
  .refine((data) => data.translation !== undefined || data.path !== undefined, {
    message: "Either translation array or path must be provided",
  });

// Scale transformation
export const ScaleTransformationSchema: z.ZodType<{
  type: "scale";
  scale?: number[] | undefined;
  path?: string | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z
  .object({
    type: z.literal("scale"),
    scale: z.array(z.number()).optional(),
    path: z.string().optional(), // For binary data
    input: z.union([z.string(), z.array(z.string())]).optional(),
    output: z.union([z.string(), z.array(z.string())]).optional(),
    name: z.string().optional(),
  })
  .refine((data) => data.scale !== undefined || data.path !== undefined, {
    message: "Either scale array or path must be provided",
  });

// Affine transformation
export const AffineTransformationSchema: z.ZodType<{
  type: "affine";
  affine?: number[][] | undefined;
  path?: string | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z
  .object({
    type: z.literal("affine"),
    affine: z.array(z.array(z.number())).optional(), // 2D array for matrix
    path: z.string().optional(), // For binary data
    input: z.union([z.string(), z.array(z.string())]).optional(),
    output: z.union([z.string(), z.array(z.string())]).optional(),
    name: z.string().optional(),
  })
  .refine((data) => data.affine !== undefined || data.path !== undefined, {
    message: "Either affine matrix or path must be provided",
  });

// Rotation transformation
export const RotationTransformationSchema: z.ZodType<{
  type: "rotation";
  rotation?: number[] | undefined;
  path?: string | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z
  .object({
    type: z.literal("rotation"),
    rotation: z.array(z.number()).optional(),
    path: z.string().optional(), // For binary data
    input: z.union([z.string(), z.array(z.string())]).optional(),
    output: z.union([z.string(), z.array(z.string())]).optional(),
    name: z.string().optional(),
  })
  .refine((data) => data.rotation !== undefined || data.path !== undefined, {
    message: "Either rotation array or path must be provided",
  });

// Coordinates transformation referencing a coordinate field array
export const CoordinatesTransformationSchema: z.ZodType<{
  type: "coordinates";
  path: string;
  interpolation?: string | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("coordinates"),
  path: z.string(),
  interpolation: z.string().optional(),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Displacements transformation referencing a displacement field array
export const DisplacementsTransformationSchema: z.ZodType<{
  type: "displacements";
  path: string;
  interpolation?: string | undefined;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("displacements"),
  path: z.string(),
  interpolation: z.string().optional(),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

type TransformationCommon = {
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
};

/**
 * The full RFC-5 transformation union as a recursive type: wrapper
 * transformations (sequence, bijection, byDimension) nest any
 * transformation, matching the bundled JSON schema's recursive
 * coordinateTransformation reference.
 */
export type CoordinateTransformation =
  | ({ type: "identity" } & TransformationCommon)
  | ({ type: "mapAxis"; mapAxis: number[] } & TransformationCommon)
  | (
    & {
      type: "projectAxis";
      droppedInputs?: number[] | undefined;
      createdOutputs?: number[] | undefined;
    }
    & TransformationCommon
  )
  | (
    & {
      type: "translation";
      translation?: number[] | undefined;
      path?: string | undefined;
    }
    & TransformationCommon
  )
  | (
    & {
      type: "scale";
      scale?: number[] | undefined;
      path?: string | undefined;
    }
    & TransformationCommon
  )
  | (
    & {
      type: "affine";
      affine?: number[][] | undefined;
      path?: string | undefined;
    }
    & TransformationCommon
  )
  | (
    & {
      type: "rotation";
      rotation?: number[] | undefined;
      path?: string | undefined;
    }
    & TransformationCommon
  )
  | (
    & { type: "coordinates"; path: string; interpolation?: string | undefined }
    & TransformationCommon
  )
  | (
    & {
      type: "displacements";
      path: string;
      interpolation?: string | undefined;
    }
    & TransformationCommon
  )
  | (
    & { type: "sequence"; transformations: CoordinateTransformation[] }
    & TransformationCommon
  )
  | (
    & {
      type: "bijection";
      forward: CoordinateTransformation;
      inverse: CoordinateTransformation;
    }
    & TransformationCommon
  )
  | (
    & {
      type: "byDimension";
      transformations: Array<{
        transformation: CoordinateTransformation;
        inputAxes: number[];
        outputAxes: number[];
      }>;
    }
    & TransformationCommon
  );

// Complete coordinate transformation schema (union of all types). Lazy so the
// wrapper schemas below can nest it recursively.
export const CoordinateTransformationSchema: z.ZodType<
  CoordinateTransformation
> = z.lazy(() =>
  z.union([
    IdentityTransformationSchema,
    MapAxisTransformationSchema,
    ProjectAxisTransformationSchema,
    TranslationTransformationSchema,
    ScaleTransformationSchema,
    AffineTransformationSchema,
    RotationTransformationSchema,
    CoordinatesTransformationSchema,
    DisplacementsTransformationSchema,
    SequenceTransformationSchema,
    BijectionTransformationSchema,
    ByDimensionTransformationSchema,
  ])
);

// Sequence transformation (for chaining transformations)
export const SequenceTransformationSchema: z.ZodType<
  Extract<CoordinateTransformation, { type: "sequence" }>
> = z.object({
  type: z.literal("sequence"),
  transformations: z.array(z.lazy(() => CoordinateTransformationSchema)),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Bijection transformation (forward and inverse)
export const BijectionTransformationSchema: z.ZodType<
  Extract<CoordinateTransformation, { type: "bijection" }>
> = z.object({
  type: z.literal("bijection"),
  forward: z.lazy(() => CoordinateTransformationSchema),
  inverse: z.lazy(() => CoordinateTransformationSchema),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// One wrapped item of a byDimension transformation. The axis arrays hold
// zero-based indices into the parent's input and output coordinate systems.
export const ByDimensionItemSchema: z.ZodType<{
  transformation: CoordinateTransformation;
  inputAxes: number[];
  outputAxes: number[];
}> = z.object({
  transformation: z.lazy(() => CoordinateTransformationSchema),
  inputAxes: z.array(z.number().int().nonnegative()),
  outputAxes: z.array(z.number().int().nonnegative()),
});

// By dimension transformation: a high dimensional transformation built from
// lower dimensional transformations on subsets of dimensions.
export const ByDimensionTransformationSchema: z.ZodType<
  Extract<CoordinateTransformation, { type: "byDimension" }>
> = z.object({
  type: z.literal("byDimension"),
  transformations: z.array(ByDimensionItemSchema),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Array coordinate system schema
export const ArrayCoordinateSystemSchema = z.object({
  name: z.string().min(1),
  axes: z.array(
    z.object({
      name: z.string(),
      type: z.literal("array"),
    }),
  ),
});

// Export type definitions
export type CoordinateSystem = z.infer<typeof CoordinateSystemSchema>;
export type IdentityTransformation = z.infer<
  typeof IdentityTransformationSchema
>;
export type MapAxisTransformation = z.infer<typeof MapAxisTransformationSchema>;
export type TranslationTransformation = z.infer<
  typeof TranslationTransformationSchema
>;
export type ScaleTransformation = z.infer<typeof ScaleTransformationSchema>;
export type AffineTransformation = z.infer<typeof AffineTransformationSchema>;
export type RotationTransformation = z.infer<
  typeof RotationTransformationSchema
>;
export type SequenceTransformation = z.infer<
  typeof SequenceTransformationSchema
>;
export type BijectionTransformation = z.infer<
  typeof BijectionTransformationSchema
>;
export type ByDimensionTransformationItem = z.infer<
  typeof ByDimensionItemSchema
>;
export type ByDimensionTransformation = z.infer<
  typeof ByDimensionTransformationSchema
>;
export type ArrayCoordinateSystem = z.infer<typeof ArrayCoordinateSystemSchema>;
