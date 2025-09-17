// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";
import { NgffImageOptionsSchema } from "./ngff_image.ts";
import { MetadataSchema } from "./zarr_metadata.ts";
import { MethodsSchema } from "./methods.ts";

export const ChunkSpecSchema: z.ZodTypeAny = z.union([
  z.number(),
  z.array(z.number()),
  z.array(z.array(z.number())),
  z.record(z.string(), z.union([z.number(), z.array(z.number())]).optional()),
]);

export const MultiscalesOptionsSchema: z.ZodTypeAny = z.object({
  images: z.array(NgffImageOptionsSchema),
  metadata: MetadataSchema,
  scaleFactors: z
    .array(z.union([z.record(z.string(), z.number()), z.number()]))
    .optional(),
  method: MethodsSchema.optional(),
  chunks: ChunkSpecSchema.optional(),
});
