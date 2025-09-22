// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
export type SupportedDims = "c" | "x" | "y" | "z" | "t";

export type SpatialDims = "x" | "y" | "z";

export type AxesType =
  | "time"
  | "space"
  | "channel"
  | "array"
  | "coordinate"
  | "displacement";

export type SpaceUnits =
  | "angstrom"
  | "attometer"
  | "centimeter"
  | "decimeter"
  | "exameter"
  | "femtometer"
  | "foot"
  | "gigameter"
  | "hectometer"
  | "inch"
  | "kilometer"
  | "megameter"
  | "meter"
  | "micrometer"
  | "mile"
  | "millimeter"
  | "nanometer"
  | "parsec"
  | "petameter"
  | "picometer"
  | "terameter"
  | "yard"
  | "yoctometer"
  | "yottameter"
  | "zeptometer"
  | "zettameter";

export type TimeUnits =
  | "attosecond"
  | "centisecond"
  | "day"
  | "decisecond"
  | "exasecond"
  | "femtosecond"
  | "gigasecond"
  | "hectosecond"
  | "hour"
  | "kilosecond"
  | "megasecond"
  | "microsecond"
  | "millisecond"
  | "minute"
  | "nanosecond"
  | "petasecond"
  | "picosecond"
  | "second"
  | "terasecond"
  | "yoctosecond"
  | "yottasecond"
  | "zeptosecond"
  | "zettasecond";

export type Units = SpaceUnits | TimeUnits;

export const supportedDims: SupportedDims[] = ["x", "y", "z", "c", "t"];

export const spaceUnits: SpaceUnits[] = [
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
];

export const timeUnits: TimeUnits[] = [
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
];

export function isDimensionSupported(dim: string): dim is SupportedDims {
  return supportedDims.includes(dim as SupportedDims);
}

export function isUnitSupported(unit: string): unit is Units {
  return (
    timeUnits.includes(unit as TimeUnits) ||
    spaceUnits.includes(unit as SpaceUnits)
  );
}
