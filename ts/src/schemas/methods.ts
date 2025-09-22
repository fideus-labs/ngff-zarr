// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { z } from "zod";

export const MethodsSchema: z.ZodTypeAny = z.enum([
  "itkwasm_gaussian",
  "itkwasm_bin_shrink",
  "itkwasm_label_image",
]);
