# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Tuple
from itertools import product

import numpy as np
from dask.array import map_blocks, map_overlap
import dask.array

from ..ngff_image import NgffImage
from ._support import (
    _align_chunks,
    _compute_sigma,
    _dim_scale_factors,
    _get_block,
    _spatial_dims,
    _spatial_dims_last_zyx,
    _next_scale_metadata,
    _next_block_shape,
    _update_previous_dim_factors,
)

_image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")


def _itkwasm_blur_and_downsample(
    image_data,
    shrink_factors,
    kernel_radius,
    smoothing,
    is_vector=False,
):
    """Blur and then downsample a given image chunk"""
    import itkwasm

    # chunk does not have metadata attached, values are ITK defaults
    image = itkwasm.image_from_array(image_data, is_vector=is_vector)

    # Skip this image block if it has 0 voxels
    block_size = image.size
    if any(block_len == 0 for block_len in block_size):
        return None

    if smoothing == "gaussian":
        from itkwasm_downsample import downsample

        downsampled = downsample(
            image, shrink_factors=shrink_factors, crop_radius=kernel_radius
        )
    elif smoothing == "label_image":
        from itkwasm_downsample import downsample_label_image

        downsampled = downsample_label_image(
            image, shrink_factors=shrink_factors, crop_radius=kernel_radius
        )
    else:
        msg = f"Unknown smoothing method: {smoothing}"
        raise ValueError(msg)

    return downsampled.data


def _itkwasm_chunk_bin_shrink(
    image_data,
    shrink_factors,
    is_vector=False,
):
    """Compute the local mean and downsample on a given image chunk"""
    import itkwasm
    from itkwasm_downsample import downsample_bin_shrink

    # chunk does not have metadata attached, values are ITK defaults
    image = itkwasm.image_from_array(image_data, is_vector=is_vector)

    # Skip this image block if it has 0 voxels
    block_size = image.size
    if any(block_len == 0 for block_len in block_size):
        return None

    downsampled = downsample_bin_shrink(image, shrink_factors=shrink_factors)
    return downsampled.data


def _downsample_itkwasm(
    ngff_image: NgffImage, default_chunks, out_chunks, scale_factors, smoothing
):
    from itkwasm_downsample import gaussian_kernel_radius

    multiscales = [
        ngff_image,
    ]
    dims = tuple(ngff_image.dims)
    spatial_dims = [dim for dim in dims if dim in _spatial_dims]
    spatial_dims = _image_dims[: len(spatial_dims)]
    transposed_dims = False

    # Track previous image for incremental downsampling
    previous_image = ngff_image
    previous_dim_factors = {d: 1 for d in dims}

    for scale_factor in scale_factors:
        # Calculate incremental factors to achieve exact target size
        dim_factors = _dim_scale_factors(
            dims,
            scale_factor,
            previous_dim_factors,
            original_image=ngff_image,
            previous_image=previous_image,
        )

        # Check if we can achieve exact target with incremental downsampling
        # If not, downsample from original instead
        can_downsample_incrementally = True
        for dim in dim_factors:
            if dim in _spatial_dims:
                dim_index = ngff_image.dims.index(dim)
                original_size = ngff_image.data.shape[dim_index]
                # Handle both int and dict scale_factor
                dim_scale_factor = (
                    scale_factor[dim]
                    if isinstance(scale_factor, dict)
                    else scale_factor
                )
                target_size = int(original_size / dim_scale_factor)

                prev_dim_index = previous_image.dims.index(dim)
                previous_size = previous_image.data.shape[prev_dim_index]

                # Check if floor(previous_size / dim_factors[dim]) == target_size
                if int(previous_size / dim_factors[dim]) != target_size:
                    can_downsample_incrementally = False
                    break

        if can_downsample_incrementally:
            # Downsample from previous image (incremental - more accurate, less memory)
            current_image = _align_chunks(previous_image, default_chunks, dim_factors)
        else:
            # Must downsample from original to get exact target size
            # Recalculate factors from original
            original_dim_factors = {d: 1 for d in dims}
            dim_factors = _dim_scale_factors(dims, scale_factor, original_dim_factors)
            current_image = _align_chunks(ngff_image, default_chunks, dim_factors)

        # Operate on a contiguous spatial block
        current_image = _spatial_dims_last_zyx(current_image)
        if tuple(current_image.dims) != dims:
            transposed_dims = True
            reorder = [current_image.dims.index(dim) for dim in dims]

        translation, scale = _next_scale_metadata(
            current_image, dim_factors, spatial_dims
        )

        # Blocks 0, ..., N-2 have the same shape
        block_0_input = _get_block(current_image, 0)
        next_block_0_shape = _next_block_shape(
            current_image, dim_factors, spatial_dims, block_0_input
        )
        block_0_size = []
        for dim in spatial_dims:
            if dim in current_image.dims:
                block_0_size.append(block_0_input.shape[current_image.dims.index(dim)])
            else:
                block_0_size.append(1)
        block_0_size.reverse()

        # Block N-1 may be smaller than preceding blocks
        block_neg1_input = _get_block(current_image, -1)
        next_block_neg1_shape = _next_block_shape(
            current_image, dim_factors, spatial_dims, block_neg1_input
        )

        # Compute overlap for Gaussian blurring for all blocks
        is_vector = current_image.dims[-1] == "c"

        # pixel units
        # Compute metadata for region splitting
        shrink_factors = [dim_factors[sd] for sd in spatial_dims]
        sigma_values = _compute_sigma(shrink_factors)
        kernel_radius = gaussian_kernel_radius(size=block_0_size, sigma=sigma_values)

        dtype = block_0_input.dtype

        output_chunks = list(current_image.data.chunks)
        output_chunks_start = 0
        while current_image.dims[output_chunks_start] not in _spatial_dims:
            output_chunks_start += 1
        output_chunks = output_chunks[output_chunks_start:]
        next_block_0_shape = next_block_0_shape[output_chunks_start:]
        for i, c in enumerate(output_chunks):
            output_chunks[i] = [
                next_block_0_shape[i],
            ] * len(c)

        next_block_neg1_shape = next_block_neg1_shape[output_chunks_start:]
        for i in range(len(output_chunks)):
            output_chunks[i][-1] = next_block_neg1_shape[i]
            output_chunks[i] = tuple(output_chunks[i])
        output_chunks = tuple(output_chunks)

        non_spatial_dims = [d for d in dims if d not in _spatial_dims]
        if "c" in non_spatial_dims and current_image.dims[-1] == "c":
            non_spatial_dims.remove("c")

        if output_chunks_start > 0:
            # We'll iterate over each index for the non-spatial dimensions, run the desired
            # map_overlap, and aggregate the outputs into a final result.

            # Determine the size for each non-spatial dimension
            non_spatial_shapes = current_image.data.shape[:output_chunks_start]

            # Collect results for each sub-block
            aggregated_blocks = []
            for idx in product(*(range(s) for s in non_spatial_shapes)):
                # Build the slice object for indexing
                slice_obj = []
                non_spatial_index = 0
                for dim in current_image.dims:
                    if dim in non_spatial_dims:
                        # Take a single index (like "t=0,1,...") for the non-spatial dimension
                        slice_obj.append(idx[non_spatial_index])
                        non_spatial_index += 1
                    else:
                        # Keep full slice for spatial/channel dims
                        slice_obj.append(slice(None))

                slice_obj = tuple(slice_obj)
                # Extract the sub-block data for the chosen index from the non-spatial dims
                sub_block_data = current_image.data[slice_obj]

                if smoothing == "bin_shrink":
                    downscaled_sub_block = map_blocks(
                        _itkwasm_chunk_bin_shrink,
                        sub_block_data,
                        shrink_factors=shrink_factors,
                        is_vector=is_vector,
                        dtype=dtype,
                        chunks=output_chunks,
                    )
                else:
                    downscaled_sub_block = map_overlap(
                        _itkwasm_blur_and_downsample,
                        sub_block_data,
                        shrink_factors=shrink_factors,
                        kernel_radius=kernel_radius,
                        smoothing=smoothing,
                        is_vector=is_vector,
                        dtype=dtype,
                        depth=dict(
                            enumerate(np.flip(kernel_radius))
                        ),  # overlap is in tzyx
                        boundary="nearest",
                        trim=False,  # Overlapped region is trimmed in blur_and_downsample to output size
                        chunks=output_chunks,
                    )
                aggregated_blocks.append(downscaled_sub_block)
            downscaled_array_shape = non_spatial_shapes + downscaled_sub_block.shape
            downscaled_array = dask.array.empty(downscaled_array_shape, dtype=dtype)
            for sub_block_idx, idx in enumerate(
                product(*(range(s) for s in non_spatial_shapes))
            ):
                # Build the slice object for indexing
                slice_obj = []
                non_spatial_index = 0
                for dim in current_image.dims:
                    if dim in non_spatial_dims:
                        # Take a single index (like "t=0,1,...") for the non-spatial dimension
                        slice_obj.append(idx[non_spatial_index])
                        non_spatial_index += 1
                    else:
                        # Keep full slice for spatial/channel dims
                        slice_obj.append(slice(None))

                slice_obj = tuple(slice_obj)
                downscaled_array[slice_obj] = aggregated_blocks[sub_block_idx]
        else:
            data = current_image.data
            if smoothing == "bin_shrink":
                downscaled_array = map_blocks(
                    _itkwasm_chunk_bin_shrink,
                    data,
                    shrink_factors=shrink_factors,
                    is_vector=is_vector,
                    dtype=dtype,
                    chunks=output_chunks,
                )
            else:
                downscaled_array = map_overlap(
                    _itkwasm_blur_and_downsample,
                    data,
                    shrink_factors=shrink_factors,
                    kernel_radius=kernel_radius,
                    smoothing=smoothing,
                    is_vector=is_vector,
                    dtype=dtype,
                    depth=dict(enumerate(np.flip(kernel_radius))),  # overlap is in tzyx
                    boundary="nearest",
                    trim=False,  # Overlapped region is trimmed in blur_and_downsample to output size
                    chunks=output_chunks,
                )

        out_chunks_list = []
        for dim in current_image.dims:
            if dim in out_chunks:
                out_chunks_list.append(out_chunks[dim])
            else:
                out_chunks_list.append(1)
        downscaled_array = downscaled_array.rechunk(tuple(out_chunks_list))

        # transpose back to original order if needed (_spatial_dims_zyx transposed the order)
        if transposed_dims:
            downscaled_array = downscaled_array.transpose(reorder)

        current_image = NgffImage(downscaled_array, dims, scale, translation)
        multiscales.append(current_image)

        # Update for next iteration
        previous_image = current_image
        previous_dim_factors = _update_previous_dim_factors(
            scale_factor, spatial_dims, previous_dim_factors
        )

    return multiscales
