// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";

// Anatomical Orientation Values based on RFC4
export const AnatomicalOrientationValuesSchema = z.enum([
  "left-to-right",
  "right-to-left",
  "anterior-to-posterior",
  "posterior-to-anterior",
  "inferior-to-superior",
  "superior-to-inferior",
  "dorsal-to-ventral",
  "ventral-to-dorsal",
  "dorsal-to-palmar",
  "palmar-to-dorsal",
  "dorsal-to-plantar",
  "plantar-to-dorsal",
  "rostral-to-caudal",
  "caudal-to-rostral",
  "cranial-to-caudal",
  "caudal-to-cranial",
  "proximal-to-distal",
  "distal-to-proximal",
]);

// AnatomicalOrientation object schema
export const AnatomicalOrientationSchema = z.object({
  type: z.literal("anatomical"),
  value: AnatomicalOrientationValuesSchema,
});

// Base Orientation schema (extensible for future orientation types)
export const OrientationTypeSchema = z.enum(["anatomical"]);
export const OrientationSchema = z.object({
  type: OrientationTypeSchema,
  value: AnatomicalOrientationValuesSchema,
});

// Axes Names enum
export const AxesNamesSchema = z.enum(["t", "c", "z", "y", "x"]);

// Space Axes Names enum (subset of AxesNames for spatial dimensions)
export const SpaceAxesNamesSchema = z.enum(["z", "y", "x"]);

// Axis Type enum
export const AxisTypeSchema = z.enum(["channel", "space", "time"]);

// Space Units enum
export const SpaceUnitSchema = z.enum([
  "angstrom",
  "attometer",
  "centimeter",
  "decimeter",
  "exameter",
  "femtometer",
  "foot",
  "gigameter",
  "hectometer",
  "inch",
  "kilometer",
  "megameter",
  "meter",
  "micrometer",
  "mile",
  "millimeter",
  "nanometer",
  "parsec",
  "petameter",
  "picometer",
  "terameter",
  "yard",
  "yoctometer",
  "yottameter",
  "zeptometer",
  "zettameter",
]);

// Time Units enum
export const TimeUnitSchema = z.enum([
  "attosecond",
  "centisecond",
  "day",
  "decisecond",
  "exasecond",
  "femtosecond",
  "gigasecond",
  "hectosecond",
  "hour",
  "kilosecond",
  "megasecond",
  "microsecond",
  "millisecond",
  "minute",
  "nanosecond",
  "petasecond",
  "picosecond",
  "second",
  "terasecond",
  "yoctosecond",
  "yottasecond",
  "zeptosecond",
  "zettasecond",
]);

// Base Axis schema
export const BaseAxisSchema = z.object({
  name: z.string(),
  type: AxisTypeSchema,
});

// Channel Axis schema
export const ChannelAxisSchema = z.object({
  name: z.literal("c"),
  type: z.literal("channel"),
});

// Space Axis schema (includes orientation)
export const SpaceAxisSchema = z.object({
  name: SpaceAxesNamesSchema,
  type: z.literal("space"),
  unit: SpaceUnitSchema,
  orientation: z.union([AnatomicalOrientationSchema, z.null()]).optional(),
});

// Time Axis schema
export const TimeAxisSchema = z.object({
  name: z.literal("t"),
  type: z.literal("time"),
  unit: TimeUnitSchema,
});

// Union of all axis types
export const AxisSchema = z.union([
  ChannelAxisSchema,
  SpaceAxisSchema,
  TimeAxisSchema,
]);

// Axes collection schema
export const AxesSchema = z.object({
  axes: z.array(AxisSchema).nullable().optional(),
});

// Export type definitions for TypeScript usage
export type AnatomicalOrientationValues = z.infer<
  typeof AnatomicalOrientationValuesSchema
>;
export type AnatomicalOrientation = z.infer<typeof AnatomicalOrientationSchema>;
export type Orientation = z.infer<typeof OrientationSchema>;
export type AxesNames = z.infer<typeof AxesNamesSchema>;
export type SpaceAxesNames = z.infer<typeof SpaceAxesNamesSchema>;
export type AxisType = z.infer<typeof AxisTypeSchema>;
export type SpaceUnit = z.infer<typeof SpaceUnitSchema>;
export type TimeUnit = z.infer<typeof TimeUnitSchema>;
export type BaseAxis = z.infer<typeof BaseAxisSchema>;
export type ChannelAxis = z.infer<typeof ChannelAxisSchema>;
export type SpaceAxis = z.infer<typeof SpaceAxisSchema>;
export type TimeAxis = z.infer<typeof TimeAxisSchema>;
export type Axis = z.infer<typeof AxisSchema>;
export type Axes = z.infer<typeof AxesSchema>;
