# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from collections import OrderedDict

import dask.array
import numpy as np

from .._array_split import _array_split
from ..config import config
from ..memory_usage import memory_usage
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
    return {
        dim: input_scale[dim] * dim_factors[dim]
        for dim in previous_image.dims
        if dim in _spatial_dims
    }


def _compute_next_translation(previous_image, dim_factors):
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

    # Index in input image space corresponding to offset after shrink
    input_index = {
        dim: 0.5 * (dim_factors[dim] - 1)
        for dim in previous_image.dims
        if dim in dim_factors
    }
    # Translate input index coordinate to offset in physical space
    # NOTE: This method fails to account for direction matrix
    return {
        dim: input_index[dim] * input_scale[dim] + input_translation[dim]
        for dim in previous_image.dims
        if dim in dim_factors
    }


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
    previous_absolute_dim_factors = {d: 1 for d in dims}
    previous_scale_factors = {d: 1 for d in dims}
    for scale_factor in scale_factors:
        dim_factors = _dim_scale_factors(
            dims, scale_factor, previous_absolute_dim_factors
        )
        previous_absolute_dim_factors = {
            d: v * previous_scale_factors[d] for d, v in dim_factors.items()
        }
        if isinstance(scale_factor, dict):
            previous_scale_factors = scale_factor
        elif isinstance(scale_factor, (int,)):
            previous_scale_factors = {d: scale_factor for d in dims}
        else:
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

        if label == "mode":

            def largest_mode(arr):
                values, counts = np.unique(arr, return_counts=True)
                m = counts.argmax()
                return values[m]

            size = tuple(shrink_factors)
            blurred_array = dask_image.ndfilters.generic_filter(
                image=previous_image.data,
                function=largest_mode,
                size=size,
                mode="nearest",
            )
        elif label == "nearest":
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
        depth = [
            1,
        ] * blurred_array.ndim
        for dim, shrink_factor in enumerate(shrink_factors):
            transform[dim, dim] = shrink_factor
            depth[dim] = shrink_factor + 1 + order

        out_chunks_list = []
        for dim in dims:
            if dim in out_chunks:
                out_chunks_list.append(out_chunks[dim])
            else:
                out_chunks_list.append(1)
        output_chunks = tuple(out_chunks_list)

        if memory_usage(previous_image) > config.memory_target:
            chunks = tuple([c[0] for c in blurred_array.chunks])
            x_index = dims.index("x")
            y_index = dims.index("y")
            ndim = blurred_array.ndim
            shape = blurred_array.shape
            if "z" in dims:
                z_index = dims.index("z")
                # TODD address, c, t, large 2D
                slice_bytes = memory_usage(previous_image, {"z"})
                slab_slices = min(
                    int(np.ceil(config.memory_target / slice_bytes)),
                    blurred_array.shape[z_index],
                )
                z_chunks = chunks[z_index]
                slice_planes = False
                if slab_slices < z_chunks:
                    slab_slices = z_chunks
                    slice_planes = True
                if slab_slices > blurred_array.shape[z_index]:
                    slab_slices = blurred_array.shape[z_index]
                slab_slices = int(slab_slices / z_chunks) * z_chunks
                num_z_splits = int(np.ceil(shape[z_index] / slab_slices))
                while num_z_splits % shrink_factors[z_index] > 1:
                    num_z_splits += 1
                num_y_splits = 1
                num_x_splits = 1
                regions = OrderedDict()
                if slice_planes:
                    # TODO
                    plane_bytes = memory_usage(previous_image, {"z", "y"})
                    plane_slices = min(
                        int(np.ceil(config.memory_target / plane_bytes)), shape[y_index]
                    )
                    y_chunks = chunks[y_index]
                    slice_strips = False
                    if plane_slices < y_chunks:
                        plane_slices = y_chunks
                        slice_strips = True
                    if plane_slices > shape[y_index]:
                        plane_slices = shape[y_index]
                    plane_slices = int(plane_slices / y_chunks) * y_chunks
                    num_y_splits = int(np.ceil(shape[y_index] / plane_slices))
                    while num_y_splits % shrink_factors[y_index] > 1:
                        num_y_splits += 1
                    if slice_strips:
                        strip_bytes = memory_usage(previous_image, {"z", "y", "x"})
                        strip_slices = min(
                            int(np.ceil(config.memory_target / strip_bytes)),
                            shape[x_index],
                        )
                        x_chunks = chunks[x_index]
                        strip_slices = max(strip_slices, x_chunks)
                        slice_strips = False
                        if strip_slices > shape[x_index]:
                            strip_slices = shape[x_index]
                        strip_slices = int(strip_slices / x_chunks) * x_chunks
                        num_x_splits = int(np.ceil(shape[x_index] / strip_slices))
                        while num_x_splits % shrink_factors[x_index] > 1:
                            num_x_splits += 1
                        z_splits = _array_split(blurred_array, num_z_splits, z_index)
                        for split_index, split in enumerate(z_splits):
                            y_splits = _array_split(split, num_y_splits, y_index)
                            for y_split_index, y_split in enumerate(y_splits):
                                x_splits = _array_split(y_split, num_x_splits, x_index)
                                y_splits[y_split_index] = x_splits
                            z_splits[split_index] = y_splits
                        z_offset = 0
                        for z_split_index, z_split in enumerate(z_splits):
                            y_offset = 0
                            for y_split_index, y_split in enumerate(z_split):
                                x_offset = 0
                                for x_split_index, x_split in enumerate(y_split):
                                    region = [slice(shape[i]) for i in range(ndim)]
                                    region[z_index] = slice(
                                        z_offset, z_offset + x_split.shape[z_index]
                                    )
                                    region[y_index] = slice(
                                        y_offset, y_offset + x_split.shape[y_index]
                                    )
                                    region[x_index] = slice(
                                        x_offset, x_offset + x_split.shape[x_index]
                                    )
                                    offset = [
                                        0,
                                    ] * ndim
                                    offset[z_index] = z_offset
                                    offset[y_index] = y_offset
                                    offset[x_index] = x_offset
                                    regions[
                                        (z_split_index, y_split_index, x_split_index)
                                    ] = tuple(region), tuple(offset)
                                    x_offset += x_split.shape[x_index]
                                y_offset += y_split[0].shape[y_index]
                            z_offset += z_split[0][0].shape[z_index]
                    else:
                        z_splits = _array_split(blurred_array, num_z_splits, z_index)
                        for split_index, split in enumerate(z_splits):
                            y_splits = _array_split(split, num_y_splits, y_index)
                            z_splits[split_index] = y_splits
                        z_offset = 0
                        for z_split_index, z_split in enumerate(z_splits):
                            y_offset = 0
                            for y_split_index, y_split in enumerate(z_split):
                                region = [slice(shape[i]) for i in range(ndim)]
                                region[z_index] = slice(
                                    z_offset, z_offset + y_split.shape[z_index]
                                )
                                region[y_index] = slice(
                                    y_offset, y_offset + y_split.shape[y_index]
                                )
                                offset = [
                                    0,
                                ] * ndim
                                offset[z_index] = z_offset
                                offset[y_index] = y_offset
                                regions[(z_split_index, y_split_index, 0)] = (
                                    tuple(region),
                                    tuple(offset),
                                )
                                y_offset += y_split.shape[y_index]
                            z_offset += z_split[0].shape[z_index]
                else:
                    z_splits = _array_split(blurred_array, num_z_splits, z_index)
                    z_offset = 0
                    for split_index, split in enumerate(z_splits):
                        region = [slice(shape[i]) for i in range(ndim)]
                        region[z_index] = slice(
                            z_offset, z_offset + split.shape[z_index]
                        )
                        offset = [
                            0,
                        ] * ndim
                        offset[z_index] = z_offset
                        regions[(split_index, 0, 0)] = tuple(region), tuple(offset)
                        z_offset += split.shape[z_index]

                blurred_region = blurred_array[regions[(0, 0, 0)][0]]
                region_output_shape = [
                    arr_len // shrink_factor
                    for arr_len, shrink_factor in zip(
                        blurred_region.shape, shrink_factors
                    )
                ]
                while region_output_shape[z_index] % shrink_factors[z_index] != 0:
                    region_output_shape[z_index] -= 1
                while region_output_shape[y_index] % shrink_factors[y_index] != 0:
                    region_output_shape[y_index] -= 1
                while region_output_shape[x_index] % shrink_factors[x_index] != 0:
                    region_output_shape[x_index] -= 1

                def downscale_region(
                    region,  # noqa: ARG001
                    offset,
                    blurred_array=blurred_array,
                    transform=transform,
                    region_output_shape=region_output_shape,
                    order=order,
                ):
                    # TODO: this should be transformed??
                    # blurred_region = blurred_array[region]
                    return dask_image.ndinterp.affine_transform(
                        blurred_array,
                        matrix=transform,
                        offset=offset,
                        output_shape=tuple(region_output_shape),  # tzyx order
                        order=order,
                    )

                z_splits = []
                for z_split in range(num_z_splits):
                    y_splits = []
                    for y_split in range(num_y_splits):
                        x_splits = []
                        for x_split in range(num_x_splits):
                            region, offset = regions[(z_split, y_split, x_split)]
                            x_splits.append(downscale_region(region, offset))
                        y_splits.append(dask.array.concatenate(x_splits, axis=x_index))
                    z_splits.append(dask.array.concatenate(y_splits, axis=y_index))
                downscaled_array = dask.array.concatenate(z_splits, axis=z_index)
                output_shape = [
                    arr_len // shrink_factor
                    for arr_len, shrink_factor in zip(
                        blurred_array.shape, shrink_factors
                    )
                ]
                # trim
                downscaled_array = downscaled_array[
                    tuple([slice(output_shape[i]) for i in range(ndim)])
                ]
            else:
                # Todo: downsample 2D images
                downscaled_array = dask_image.ndinterp.affine_transform(
                    blurred_array,
                    matrix=transform,
                    order=order,
                    output_shape=output_shape,  # tzyx order
                )
        else:
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
