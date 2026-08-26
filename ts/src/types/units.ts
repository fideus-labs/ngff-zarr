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

/**
 * Axis name. RFC-3 (OME-Zarr 0.9.dev1) permits any string; below 0.9.dev1 the
 * `SupportedDims` convention is enforced by structural validation, not the
 * type system. The union with the literal set keeps editor completion.
 */
export type AxisName = SupportedDims | (string & Record<never, never>);

/**
 * Axis type. RFC-3 (OME-Zarr 0.9.dev1) permits any string alongside the
 * spec-defined `AxesType` set, and `type` may be omitted entirely.
 */
export type AxisType = AxesType | (string & Record<never, never>);

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

/**
 * Axis unit. RFC-3 (OME-Zarr 0.9.dev1) permits any string alongside the
 * spec-defined `Units` vocabulary, and `unit` may be omitted entirely.
 */
export type AxisUnit = Units | (string & Record<never, never>);

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
