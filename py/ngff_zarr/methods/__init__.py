# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from enum import Enum

methods = [
    ("ITKWASM_GAUSSIAN", "itkwasm_gaussian"),
    ("ITKWASM_BIN_SHRINK", "itkwasm_bin_shrink"),
    ("ITKWASM_LABEL_IMAGE", "itkwasm_label_image"),
    ("ITK_GAUSSIAN", "itk_gaussian"),
    ("ITK_BIN_SHRINK", "itk_bin_shrink"),
    #    ("ITK_LABEL_GAUSSIAN", "itk_label_gaussian"),
    ("DASK_IMAGE_GAUSSIAN", "dask_image_gaussian"),
    ("DASK_IMAGE_MODE", "dask_image_mode"),
    ("DASK_IMAGE_NEAREST", "dask_image_nearest"),
]
methods_values = [m[1] for m in methods]
Methods = Enum("Methods", methods)
