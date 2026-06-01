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

export const SupportedDimsSchema: z.ZodEnum<{
  c: "c";
  x: "x";
  y: "y";
  z: "z";
  t: "t";
}> = z.enum(["c", "x", "y", "z", "t"]);

export const SpatialDimsSchema: z.ZodEnum<{
  x: "x";
  y: "y";
  z: "z";
}> = z.enum(["x", "y", "z"]);

export const AxesTypeSchema: z.ZodEnum<{
  time: "time";
  space: "space";
  channel: "channel";
  array: "array";
  coordinate: "coordinate";
  displacement: "displacement";
}> = z.enum(["time", "space", "channel", "array", "coordinate", "displacement"]);

export const SpaceUnitsSchema: z.ZodEnum<{
  angstrom: "angstrom";
  attometer: "attometer";
  centimeter: "centimeter";
  decimeter: "decimeter";
  exameter: "exameter";
  femtometer: "femtometer";
  foot: "foot";
  gigameter: "gigameter";
  hectometer: "hectometer";
  inch: "inch";
  kilometer: "kilometer";
  megameter: "megameter";
  meter: "meter";
  micrometer: "micrometer";
  mile: "mile";
  millimeter: "millimeter";
  nanometer: "nanometer";
  parsec: "parsec";
  petameter: "petameter";
  picometer: "picometer";
  terameter: "terameter";
  yard: "yard";
  yoctometer: "yoctometer";
  yottameter: "yottameter";
  zeptometer: "zeptometer";
  zettameter: "zettameter";
}> = z.enum([
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

export const TimeUnitsSchema: z.ZodEnum<{
  attosecond: "attosecond";
  centisecond: "centisecond";
  day: "day";
  decisecond: "decisecond";
  exasecond: "exasecond";
  femtosecond: "femtosecond";
  gigasecond: "gigasecond";
  hectosecond: "hectosecond";
  hour: "hour";
  kilosecond: "kilosecond";
  megasecond: "megasecond";
  microsecond: "microsecond";
  millisecond: "millisecond";
  minute: "minute";
  nanosecond: "nanosecond";
  petasecond: "petasecond";
  picosecond: "picosecond";
  second: "second";
  terasecond: "terasecond";
  yoctosecond: "yoctosecond";
  yottasecond: "yottasecond";
  zeptosecond: "zeptosecond";
  zettasecond: "zettasecond";
}> = z.enum([
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
