# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Optional, Set

from .ngff_image import NgffImage


def task_count(image: NgffImage, constrained_dims: Optional[Set[str]] = None) -> int:
    """Approximate dask array partition task count."""
    if constrained_dims is None:
        constrained_dims = set()
    arr = image.data
    dims = image.dims
    slices = []
    for index, dim in enumerate(dims):
        if dim in constrained_dims:
            slices.append(slice(0))
        else:
            slices.append(slice(arr.shape[index]))
    return len(arr[tuple(slices)].dask)
