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
    .refine((indices) => new Set(indices).size === indices.length, {
      message: "mapAxis indices must be unique",
    }),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

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

// Forward declaration for recursive types
type BaseCoordinateTransformation = {
  type:
    | "identity"
    | "mapAxis"
    | "translation"
    | "scale"
    | "affine"
    | "rotation";
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
  mapAxis?: number[] | undefined;
  translation?: number[] | undefined;
  path?: string | undefined;
  scale?: number[] | undefined;
  affine?: number[][] | undefined;
  rotation?: number[] | undefined;
};

const BaseCoordinateTransformationSchema: z.ZodType<
  BaseCoordinateTransformation
> = z.union([
  IdentityTransformationSchema,
  MapAxisTransformationSchema,
  TranslationTransformationSchema,
  ScaleTransformationSchema,
  AffineTransformationSchema,
  RotationTransformationSchema,
]);

// Sequence transformation (for chaining transformations)
export const SequenceTransformationSchema: z.ZodType<{
  type: "sequence";
  transformations: BaseCoordinateTransformation[];
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("sequence"),
  transformations: z.array(BaseCoordinateTransformationSchema),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Inverse transformation
export const InverseTransformationSchema: z.ZodType<{
  type: "inverseOf";
  transformation: BaseCoordinateTransformation;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("inverseOf"),
  transformation: BaseCoordinateTransformationSchema,
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Bijection transformation (forward and inverse)
export const BijectionTransformationSchema: z.ZodType<{
  type: "bijection";
  forward: BaseCoordinateTransformation;
  inverse: BaseCoordinateTransformation;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("bijection"),
  forward: BaseCoordinateTransformationSchema,
  inverse: BaseCoordinateTransformationSchema,
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// One wrapped item of a byDimension transformation. The axis arrays hold
// zero-based indices into the parent's input and output coordinate systems.
export const ByDimensionItemSchema: z.ZodType<{
  transformation: BaseCoordinateTransformation;
  input_axes: number[];
  output_axes: number[];
}> = z.object({
  transformation: BaseCoordinateTransformationSchema,
  input_axes: z.array(z.number().int().nonnegative()),
  output_axes: z.array(z.number().int().nonnegative()),
});

// By dimension transformation: a high dimensional transformation built from
// lower dimensional transformations on subsets of dimensions.
export const ByDimensionTransformationSchema: z.ZodType<{
  type: "byDimension";
  transformations: Array<{
    transformation: BaseCoordinateTransformation;
    input_axes: number[];
    output_axes: number[];
  }>;
  input?: string | string[] | undefined;
  output?: string | string[] | undefined;
  name?: string | undefined;
}> = z.object({
  type: z.literal("byDimension"),
  transformations: z.array(ByDimensionItemSchema),
  input: z.union([z.string(), z.array(z.string())]).optional(),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  name: z.string().optional(),
});

// Complete coordinate transformation schema (union of all types)
export const CoordinateTransformationSchema: z.ZodType<
  | {
    type: "identity";
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "mapAxis";
    mapAxis: number[];
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "translation";
    translation?: number[] | undefined;
    path?: string | undefined;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "scale";
    scale?: number[] | undefined;
    path?: string | undefined;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "affine";
    affine?: number[][] | undefined;
    path?: string | undefined;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "rotation";
    rotation?: number[] | undefined;
    path?: string | undefined;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "sequence";
    transformations: BaseCoordinateTransformation[];
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "inverseOf";
    transformation: BaseCoordinateTransformation;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "bijection";
    forward: BaseCoordinateTransformation;
    inverse: BaseCoordinateTransformation;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
  | {
    type: "byDimension";
    transformations: Array<{
      transformation: BaseCoordinateTransformation;
      input_axes: number[];
      output_axes: number[];
    }>;
    input?: string | string[] | undefined;
    output?: string | string[] | undefined;
    name?: string | undefined;
  }
> = z.union([
  IdentityTransformationSchema,
  MapAxisTransformationSchema,
  TranslationTransformationSchema,
  ScaleTransformationSchema,
  AffineTransformationSchema,
  RotationTransformationSchema,
  SequenceTransformationSchema,
  InverseTransformationSchema,
  BijectionTransformationSchema,
  ByDimensionTransformationSchema,
]);

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
export type InverseTransformation = z.infer<typeof InverseTransformationSchema>;
export type BijectionTransformation = z.infer<
  typeof BijectionTransformationSchema
>;
export type ByDimensionTransformationItem = z.infer<
  typeof ByDimensionItemSchema
>;
export type ByDimensionTransformation = z.infer<
  typeof ByDimensionTransformationSchema
>;
export type CoordinateTransformation = z.infer<
  typeof CoordinateTransformationSchema
>;
export type ArrayCoordinateSystem = z.infer<typeof ArrayCoordinateSystemSchema>;
