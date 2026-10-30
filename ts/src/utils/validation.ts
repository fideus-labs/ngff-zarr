// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z as _z } from "zod";
import type { Metadata } from "../types/zarr_metadata.ts";
import { MetadataSchema } from "../schemas/zarr_metadata.ts";
import { spaceUnits, supportedDims, timeUnits } from "../types/units.ts";

export function validateMetadata(metadata: unknown): Metadata {
  const result = MetadataSchema.safeParse(metadata);

  if (!result.success) {
    throw new Error(`Invalid metadata: ${result.error.message}`);
  }

  return result.data as Metadata;
}

export function isValidDimension(dim: string): boolean {
  return (supportedDims as string[]).includes(dim);
}

export function isValidUnit(unit: string): boolean {
  return (
    (spaceUnits as string[]).includes(unit) ||
    (timeUnits as string[]).includes(unit)
  );
}
