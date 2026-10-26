# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np

from ..ngff_image import NgffImage
from ._support import _align_chunks, _compute_sigma, _dim_scale_factors, _spatial_dims


def _compute_next_scale(previous_image: NgffImage, dim_factors):
    """Helper method to manually compute output image spacing.

    previous_image: NgffImage
        The image for which voxel spacings are use to compute spacing for the next scale

    dim_factors: Dict
        Shrink ratio along each enumerated axis

    result: Dict
        Spacing along each enumerated image axis
        Example {'x': 2.0, 'y': 1.0}
    """
    input_scale = previous_image.scale
    scale = {}
    for dim in previous_image.dims:
        if dim in _spatial_dims and dim in dim_factors:
            # Apply scale factor for spatial dims being downsampled
            scale[dim] = input_scale[dim] * dim_factors[dim]
        elif dim in input_scale:
            # Preserve scale for non-spatial dims and spatial dims not being downsampled
            scale[dim] = input_scale[dim]
    return scale


def _compute_next_translation(previous_image: NgffImage, dim_factors):
    """Helper method to manually compute output image physical offset.
        Note that this method does not account for an image direction matrix.

    previous_image: NgffImage
        The image for which voxel offsets are input

    dim_factors: Dict
        Shrink ratio along each enumerated axis

    result: Dict
        Offset in physical space of first voxel in output image
        Example {'x': 0.5, 'y': 1.0}
    """
    input_scale = previous_image.scale
    input_translation = previous_image.translation

    translation = {}
    for dim in previous_image.dims:
        if dim in _spatial_dims and dim in dim_factors:
            # Calculate new translation for spatial dims being downsampled
            input_index = 0.5 * (dim_factors[dim] - 1)
            translation[dim] = input_index * input_scale[dim] + input_translation[dim]
        elif dim in input_translation:
            # Preserve translation for non-spatial dims and spatial dims not being downsampled
            translation[dim] = input_translation[dim]
    return translation


def _get_truncate(previous_image, sigma_values, truncate_start=4.0) -> float:
    """Discover truncate parameter yielding a viable kernel width
        for dask_image.ndfilters.gaussian_filter processing. Block overlap
        cannot be greater than image size, so kernel radius is more limited
        for small images. A lower stddev truncation ceiling for kernel
        generation can result in a less precise kernel.

    previous_image: _NgffImage
        Chunked image to be smoothed

    sigma_values: List
        Gaussian kernel standard deviations in tzyx order

    truncate_start: float
        First truncation value to try.

    result: float
        Truncation value found to yield largest possible kernel width without
        extending beyond one chunk such that chunked smoothing would fail.
    """

    from dask_image.ndfilters._gaussian import _get_border

    truncate = truncate_start
    stddev_step = 0.5  # search by stepping down by 0.5 stddev in each iteration

    border = _get_border(previous_image.data, sigma_values, truncate)
    while any(
        border_len > image_len
        for border_len, image_len in zip(border, previous_image.data.shape)
    ):
        truncate = truncate - stddev_step
        if truncate <= 0.0:
            break
        border = _get_border(previous_image.data, sigma_values, truncate)

    return truncate


def _downsample_dask_image(
    ngff_image: NgffImage, default_chunks, out_chunks, scale_factors, label=False
):
    import dask_image.ndfilters
    import dask_image.ndinterp

    multiscales = [
        ngff_image,
    ]
    previous_image = ngff_image
    dims = ngff_image.dims
    previous_absolute_dim_factors = dict.fromkeys(dims, 1)
    for scale_factor in scale_factors:
        dim_factors = _dim_scale_factors(
            dims, scale_factor, previous_absolute_dim_factors
        )
        # Update absolute factors for all dims
        new_absolute_dim_factors = {}
        for d in dims:
            if d in dim_factors:
                # Multiply by incremental factor for dims being downsampled
                new_absolute_dim_factors[d] = dim_factors[
                    d
                ] * previous_absolute_dim_factors.get(d, 1.0)
            else:
                # Keep existing absolute factor for dims not being downsampled
                new_absolute_dim_factors[d] = previous_absolute_dim_factors.get(d, 1.0)
        previous_absolute_dim_factors = new_absolute_dim_factors
        if not isinstance(scale_factor, (dict, int)):
            msg = "Unexpected scale_factor type"
            raise ValueError(msg)
        previous_image = _align_chunks(previous_image, default_chunks, dim_factors)

        shrink_factors = []
        for dim in dims:
            if dim in dim_factors:
                shrink_factors.append(dim_factors[dim])
            else:
                shrink_factors.append(1)

        # Compute output shape and metadata
        output_shape = [
            int(image_len / shrink_factor)
            for image_len, shrink_factor in zip(
                previous_image.data.shape, shrink_factors
            )
        ]
        output_scale = _compute_next_scale(previous_image, dim_factors)
        output_translation = _compute_next_translation(previous_image, dim_factors)

        if label == "nearest":
            blurred_array = previous_image.data
        else:
            input_scale_list = []
            for dim in dims:
                if dim in previous_image.scale:
                    input_scale_list.append(previous_image.scale[dim])
                else:
                    input_scale_list.append(1.0)

            sigma_values = _compute_sigma(shrink_factors)
            truncate = _get_truncate(previous_image, sigma_values)

            blurred_array = dask_image.ndfilters.gaussian_filter(
                image=previous_image.data,
                sigma=sigma_values,  # tzyx order
                mode="nearest",
                truncate=truncate,
            )

        # Construct downsample parameters
        image_dimension = len(shrink_factors)
        transform = np.eye(image_dimension)
        order = 0 if label else 1
        for dim, shrink_factor in enumerate(shrink_factors):
            transform[dim, dim] = shrink_factor

        out_chunks_list = []
        for dim in dims:
            if dim in out_chunks:
                out_chunks_list.append(out_chunks[dim])
            else:
                out_chunks_list.append(1)
        output_chunks = tuple(out_chunks_list)

        downscaled_array = dask_image.ndinterp.affine_transform(
            blurred_array,
            matrix=transform,
            order=order,
            output_shape=output_shape,  # tzyx order
        )

        downscaled_array = downscaled_array.rechunk(output_chunks)

        previous_image = NgffImage(
            downscaled_array, dims, output_scale, output_translation
        )
        multiscales.append(previous_image)

    return multiscales
