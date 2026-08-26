# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import math

import dask.array
import numpy as np

from ..ngff_image import NgffImage
from ._support import (
    _dim_scale_factors,
    _next_scale_metadata,
    _spatial_dims,
    _update_previous_dim_factors,
)


def _bin_shrink(data: dask.array.Array, axes: dict[int, int]) -> dask.array.Array:
    """Local mean over ``axes`` windows, remainder trimmed, integers rounded half up.

    The mean accumulates in float64, which is exact for integers of up to 32
    bits; 64-bit integers take the integer path. Rounding half up on integer
    input gives the same samples as ITK's BinShrinkImageFilter.
    """
    if all(factor == 1 for factor in axes.values()):
        return data
    if np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize == 8:
        return _bin_shrink_wide_integers(data, axes)
    mean = dask.array.coarsen(np.mean, data.astype(np.float64), axes, trim_excess=True)
    if np.issubdtype(data.dtype, np.integer):
        return dask.array.floor(mean + 0.5).astype(data.dtype)
    return mean.astype(data.dtype)


def _bin_shrink_wide_integers(
    data: dask.array.Array, axes: dict[int, int]
) -> dask.array.Array:
    """Half-up window mean of 64-bit integers in integer arithmetic.

    float64 carries 53 bits, so a 64-bit sample converted to it can lose low
    bits. Each sample is split by the window size into a quotient and a
    remainder; the quotient sum stays within the dtype and the remainder sum
    is small, so the mean is exact.
    """
    # Scalars carry the array dtype: a uint64 array combined with an int64
    # scalar is promoted to float64.
    count = data.dtype.type(math.prod(axes.values()))
    two = data.dtype.type(2)
    quotient = dask.array.coarsen(np.sum, data // count, axes, trim_excess=True)
    remainder = dask.array.coarsen(np.sum, data % count, axes, trim_excess=True)
    rounded = (two * remainder + count) // (two * count)
    return (quotient + rounded).astype(data.dtype)


def _reaches_target(ngff_image, previous_image, scale_factor, dim_factors) -> bool:
    """Whether shrinking ``previous_image`` by ``dim_factors`` hits the level size."""
    for dim, factor in dim_factors.items():
        if dim not in _spatial_dims:
            continue
        original_size = ngff_image.data.shape[ngff_image.dims.index(dim)]
        absolute = (
            scale_factor.get(dim, 1) if isinstance(scale_factor, dict) else scale_factor
        )
        target_size = int(original_size / absolute)
        previous_size = previous_image.data.shape[previous_image.dims.index(dim)]
        if int(previous_size / factor) != target_size:
            return False
    return True


def _downsample_dask_bin_shrink(
    ngff_image: NgffImage, default_chunks, out_chunks, scale_factors
):
    multiscales = [ngff_image]
    dims = tuple(ngff_image.dims)
    spatial_dims = tuple(dim for dim in dims if dim in _spatial_dims)
    previous_image = ngff_image
    previous_dim_factors = dict.fromkeys(dims, 1)

    for scale_factor in scale_factors:
        dim_factors = _dim_scale_factors(
            dims,
            scale_factor,
            previous_dim_factors,
            original_image=ngff_image,
            previous_image=previous_image,
        )
        source_image = previous_image
        if not _reaches_target(ngff_image, previous_image, scale_factor, dim_factors):
            dim_factors = _dim_scale_factors(dims, scale_factor, dict.fromkeys(dims, 1))
            source_image = ngff_image

        axes = {
            index: dim_factors.get(dim, 1) if dim in _spatial_dims else 1
            for index, dim in enumerate(dims)
        }
        data = _bin_shrink(source_image.data, axes)
        data = data.rechunk(tuple(out_chunks.get(dim, 1) for dim in dims))

        translation, scale = _next_scale_metadata(
            source_image, dim_factors, spatial_dims
        )
        current_image = NgffImage(data, dims, scale, translation)
        multiscales.append(current_image)

        previous_image = current_image
        previous_dim_factors = _update_previous_dim_factors(
            scale_factor, spatial_dims, previous_dim_factors
        )

    return multiscales
