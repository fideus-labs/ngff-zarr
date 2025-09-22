// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";
import {
  spaceUnits,
  type SupportedDims,
  supportedDims,
  timeUnits,
  type Units,
} from "../types/units.ts";

export const SupportedDimsSchema: z.ZodTypeAny = z.enum([
  "c",
  "x",
  "y",
  "z",
  "t",
]);

export const SpatialDimsSchema: z.ZodTypeAny = z.enum(["x", "y", "z"]);

export const AxesTypeSchema: z.ZodTypeAny = z.enum([
  "time",
  "space",
  "channel",
  "array",
  "coordinate",
  "displacement",
]);

export const SpaceUnitsSchema: z.ZodTypeAny = z.enum([
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

export const TimeUnitsSchema: z.ZodTypeAny = z.enum([
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

export const UnitsSchema: z.ZodUnion<
  [typeof SpaceUnitsSchema, typeof TimeUnitsSchema]
> = z.union([SpaceUnitsSchema, TimeUnitsSchema]);

export const dimensionValidator: z.ZodTypeAny = z
  .string()
  .refine(
    (dim): dim is SupportedDims => supportedDims.includes(dim as SupportedDims),
    { message: "Dimension must be one of: c, x, y, z, t" },
  );

export const unitValidator: z.ZodTypeAny = z
  .string()
  .refine(
    (unit): unit is Units =>
      [...timeUnits, ...spaceUnits].includes(unit as Units),
    { message: "Unit must be a valid time or space unit" },
  );
