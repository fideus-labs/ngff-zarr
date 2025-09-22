# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Optional, Set

from .ngff_image import NgffImage


def memory_usage(image: NgffImage, constrained_dims: Optional[Set[str]] = None) -> int:
    """Approximate array partition memory usage in bytes.

    Assumes array will have the same memory usage resulting from an array chunk."""
    arr = image.data
    dims = image.dims
    shape = arr.shape
    block = [c[0] for c in arr.chunks]
    partition_size = 1
    if constrained_dims is None:
        constrained_dims = set()
    for dim in range(arr.ndim):
        if dims[dim] in constrained_dims:
            partition_size *= block[dim] * arr.itemsize
        else:
            partition_size *= shape[dim] * arr.itemsize
    return partition_size
