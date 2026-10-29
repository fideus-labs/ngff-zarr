// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
export const Methods = {
  ITKWASM_GAUSSIAN: "itkwasm_gaussian",
  ITKWASM_BIN_SHRINK: "itkwasm_bin_shrink",
  ITKWASM_LABEL_IMAGE: "itkwasm_label_image",
} as const;
export type Methods = (typeof Methods)[keyof typeof Methods];

export const methodsValues: Methods[] = Object.values(Methods);
