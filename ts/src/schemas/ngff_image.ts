// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";
import { UnitsSchema } from "./units.ts";

// Since NgffImage now uses zarr.Array directly, we create a basic schema for validation
const ZarrArraySchema: z.ZodObject<{
  shape: z.ZodArray<z.ZodNumber>;
  dtype: z.ZodString;
  chunks: z.ZodArray<z.ZodNumber>;
  path: z.ZodOptional<z.ZodString>;
  name: z.ZodOptional<z.ZodString>;
}> = z.object({
  shape: z.array(z.number()),
  dtype: z.string(),
  chunks: z.array(z.number()),
  path: z.string().optional(),
  name: z.string().optional(),
});

export const NgffImageOptionsSchema: z.ZodObject<{
  data: typeof ZarrArraySchema;
  dims: z.ZodArray<z.ZodString>;
  scale: z.ZodRecord<z.ZodString, z.ZodNumber>;
  translation: z.ZodRecord<z.ZodString, z.ZodNumber>;
  name: z.ZodDefault<z.ZodString>;
  axesUnits: z.ZodOptional<z.ZodRecord<z.ZodString, typeof UnitsSchema>>;
  computedCallbacks: z.ZodDefault<z.ZodArray<z.ZodAny>>;
}> = z.object({
  data: ZarrArraySchema,
  dims: z.array(z.string()),
  scale: z.record(z.string(), z.number()),
  translation: z.record(z.string(), z.number()),
  name: z.string().default("image"),
  axesUnits: z.record(z.string(), UnitsSchema).optional(),
  computedCallbacks: z.array(z.any()).default([]),
});
