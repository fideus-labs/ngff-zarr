# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import List
import copy

import numpy as np
from numpy.typing import ArrayLike

from ..ngff_image import NgffImage

_spatial_dims = {"x", "y", "z"}


def _spatial_dims_last(ngff_image: NgffImage) -> NgffImage:
    dims = list(ngff_image.dims)
    spatial_dims = [dim for dim in dims if dim in _spatial_dims]

    dims_spatial_channel = len(spatial_dims)
    if dims[-1] == "c":
        dims_spatial_channel += 1

    # If spatial dimensions are already last (and 'c' can be last), return the original image
    if all(dim in dims[-dims_spatial_channel:] for dim in spatial_dims + ["c"]):
        return ngff_image

    # Move spatial dimensions to the end, keeping 'c' as the last pre-spatial dimension if present
    non_spatial_dims = [dim for dim in dims if dim not in _spatial_dims]
    if "c" in non_spatial_dims:
        non_spatial_dims.remove("c")
        new_dims = non_spatial_dims + ["c"] + spatial_dims
    else:
        new_dims = non_spatial_dims + spatial_dims

    new_order = [dims.index(dim) for dim in new_dims]

    if tuple(new_dims) == tuple(ngff_image.dims):
        return ngff_image

    # Reorder the data array
    reordered_data = ngff_image.data.transpose(new_order)

    result = copy.copy(ngff_image)
    result.data = reordered_data
    result.dims = tuple(new_dims)

    return result


def _spatial_dims_last_zyx(ngff_image: NgffImage) -> NgffImage:
    dims = list(ngff_image.dims)
    spatial_dims = [dim for dim in dims if dim in _spatial_dims]

    # If spatial dimensions are already zyx, return the original image
    if spatial_dims == ["z", "y", "x"] or spatial_dims == ["y", "x"]:
        dims_spatial_channel = len(spatial_dims)
        if dims[-1] == "c":
            dims_spatial_channel += 1

        # If spatial dimensions are already last (and 'c' can be last), return the original image
        if all(dim in dims[-dims_spatial_channel:] for dim in spatial_dims + ["c"]):
            return ngff_image

    # Move spatial dimensions to the end, keeping 'c' as the last pre-spatial dimension if present
    non_spatial_dims = [dim for dim in dims if dim not in _spatial_dims]
    new_spatial_dims = ["z", "y", "x"][-len(spatial_dims) :]
    if "c" in non_spatial_dims:
        non_spatial_dims.remove("c")
        new_dims = non_spatial_dims + ["c"] + new_spatial_dims
    else:
        new_dims = non_spatial_dims + new_spatial_dims

    new_order = [dims.index(dim) for dim in new_dims]

    if tuple(new_dims) == tuple(ngff_image.dims):
        return ngff_image

    # Reorder the data array
    reordered_data = ngff_image.data.transpose(new_order)

    result = copy.copy(ngff_image)
    result.data = reordered_data
    result.dims = tuple(new_dims)

    return result


def _channel_dim_last(ngff_image: NgffImage) -> NgffImage:
    if "c" not in ngff_image.dims or ngff_image.dims[-1] == "c":
        return ngff_image

    dims = list(ngff_image.dims)
    # Move 'c' dimension to the end
    dims.remove("c")
    dims.append("c")

    # Reorder the data array
    new_order = [ngff_image.dims.index(dim) for dim in dims]
    reordered_data = ngff_image.data.transpose(new_order)

    result = copy.copy(ngff_image)
    result.data = reordered_data
    result.dims = tuple(dims)

    return result


def _dim_scale_factors(dims, scale_factor, previous_dim_factors):
    if isinstance(scale_factor, int):
        result_scale_factors = {
            dim: int(scale_factor / previous_dim_factors[dim])
            for dim in dims
            if dim in _spatial_dims
        }
    else:
        result_scale_factors = {
            d: int(scale_factor[d] / previous_dim_factors[d]) for d in scale_factor
        }
        # if a dim is not in the scale_factors, add it with a scale factor of 1
        for d in dims:
            if d not in result_scale_factors:
                result_scale_factors[d] = 1

    return result_scale_factors


def _update_previous_dim_factors(scale_factor, spatial_dims, previous_dim_factors):
    previous_dim_factors = copy.copy(previous_dim_factors)
    if isinstance(scale_factor, int):
        for d in spatial_dims:
            previous_dim_factors[d] = scale_factor
    else:
        for d in scale_factor:
            previous_dim_factors[d] = scale_factor[d]
    return previous_dim_factors


def _align_chunks(previous_image, default_chunks, dim_factors):
    block_0_shape = [c[0] for c in previous_image.data.chunks]

    rechunk = False
    aligned_chunks = {}
    for dim, factor in dim_factors.items():
        dim_index = previous_image.dims.index(dim)
        if block_0_shape[dim_index] % factor:
            aligned_chunks[dim] = block_0_shape[dim_index] * factor
            rechunk = True
        else:
            aligned_chunks[dim] = default_chunks[dim]
    if rechunk:
        dask_aligned_chunks = {
            previous_image.dims.index(dim): aligned_chunks[dim]
            for dim in aligned_chunks
        }
        previous_image.data = previous_image.data.rechunk(dask_aligned_chunks)

    return previous_image


def _compute_sigma(shrink_factors: List[int]) -> List[float]:
    """Compute Gaussian kernel sigma values in pixel units for downsampling.
    sigma = sqrt((k^2 - 1^2)/(2*sqrt(2*ln(2)))^2)
    Ref https://discourse.itk.org/t/resampling-to-isotropic-signal-processing-theory/1403/16
    https://doi.org/10.1007/978-3-319-24571-3_81
    http://discovery.ucl.ac.uk/1469251/1/scale-factor-point-5.pdf

    Note: If input spacing / output sigma in physical units, the function would be
        sigma = sqrt((input_spacing^2*(k^2 - 1^2))/(2*sqrt(2*ln(2)))^2)

    input spacings: List
        Input image physical spacings in xyzt order

    shrink_factors: List
        Shrink ratio along each axis in xyzt order

    result: List
        Standard deviation of Gaussian kernel along each axis in xyzt order
    """
    import math

    denominator = (2 * ((2 * math.log(2)) ** 0.5)) ** 2
    return [((factor**2 - 1**2) / denominator) ** 0.5 for factor in shrink_factors]


def _get_block(previous_image: NgffImage, block_index: int):
    """Helper method for accessing an enumerated chunk from input"""
    block_shape = [c[block_index] for c in previous_image.data.chunks]
    block = previous_image.data[tuple([slice(0, s) for s in block_shape])]
    return block


def _next_scale_metadata(
    previous_image: NgffImage, dim_factors: dict, spatial_dims: tuple
) -> tuple:
    """Compute the next scale metadata based on the previous image and scale factor."""

    scale = {}
    for dim in previous_image.dims:
        if dim in spatial_dims:
            scale[dim] = previous_image.scale[dim] * dim_factors[dim]
        elif dim in previous_image.scale:
            scale[dim] = previous_image.scale[dim]

    translation = {}
    for dim in previous_image.dims:
        if dim in spatial_dims:
            translation[dim] = previous_image.translation[dim] + (
                0.5 * (dim_factors[dim] - 1) * previous_image.scale[dim]
            )
        elif dim in previous_image.translation:
            translation[dim] = previous_image.translation[dim]

    return translation, scale


def _next_block_shape(
    previous_image: NgffImage,
    dim_factors: dict,
    spatial_dims: tuple,
    block_input: ArrayLike,
) -> tuple:
    """Compute the next block shape based on the previous image and scale factor."""

    shape = []
    for i, dim in enumerate(previous_image.dims):
        if dim in spatial_dims:
            shape.append(int(np.floor(block_input.shape[i] / dim_factors[dim])))
        else:
            shape.append(block_input.shape[i])

    return tuple(shape)
