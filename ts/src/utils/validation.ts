// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z as _z } from "zod";
import type { Metadata } from "../types/zarr_metadata.ts";
import { MetadataSchema } from "../schemas/zarr_metadata.ts";

export function validateMetadata(metadata: unknown): Metadata {
  const result = MetadataSchema.safeParse(metadata);

  if (!result.success) {
    throw new Error(`Invalid metadata: ${result.error.message}`);
  }

  return result.data as Metadata;
}

export function validateColor(color: string): void {
  if (!/^[0-9A-Fa-f]{6}$/.test(color)) {
    throw new Error(`Invalid color '${color}'. Must be 6 hex digits.`);
  }
}

export function isValidDimension(dim: string): boolean {
  return ["c", "x", "y", "z", "t"].includes(dim);
}

export function isValidUnit(unit: string): boolean {
  const timeUnits = [
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

  const spaceUnits = [
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

  return timeUnits.includes(unit) || spaceUnits.includes(unit);
}
