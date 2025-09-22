# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Tuple

import numpy as np
from dask.array import concatenate, expand_dims, map_blocks, map_overlap, take

from ..ngff_image import NgffImage
from ._support import (
    _align_chunks,
    _compute_sigma,
    _dim_scale_factors,
    _update_previous_dim_factors,
    _get_block,
    _spatial_dims,
    _next_scale_metadata,
    _next_block_shape,
)

_image_dims: Tuple[str, str, str, str] = ("x", "y", "z", "t")


def _compute_itk_gaussian_kernel_radius(input_size, sigma_values) -> list:
    """Get kernel radius in xyzt directions"""
    DEFAULT_MAX_KERNEL_WIDTH = 32
    MAX_KERNEL_ERROR = 0.01
    image_dimension = len(input_size)

    import itk

    # Constrain kernel width to be at most the size of one chunk
    max_kernel_width = min(DEFAULT_MAX_KERNEL_WIDTH, *input_size)
    variance = [sigma**2 for sigma in sigma_values]

    def generate_radius(direction: int) -> int:
        """Follow itk.DiscreteGaussianImageFilter procedure to generate directional kernels"""
        oper = itk.GaussianOperator[itk.F, image_dimension]()
        oper.SetDirection(direction)
        oper.SetMaximumError(MAX_KERNEL_ERROR)
        oper.SetMaximumKernelWidth(max_kernel_width)
        oper.SetVariance(variance[direction])
        oper.CreateDirectional()
        return oper.GetRadius(direction)

    return [generate_radius(dim) for dim in range(image_dimension)]


def _itk_blur_and_downsample(
    image_data,
    gaussian_filter_name,
    interpolator_name,
    shrink_factors,
    sigma_values,
    kernel_radius,
):
    """Blur and then downsample a given image chunk"""
    import itk

    # chunk does not have metadata attached, values are ITK defaults
    image = itk.image_view_from_array(image_data)

    # Skip this image block if it has 0 voxels
    block_size = itk.size(image)
    if any(block_len == 0 for block_len in block_size):
        return None

    input_origin = itk.origin(image)
    input_spacing = itk.spacing(image)

    # Output values are relative to input
    itk_shrink_factors = shrink_factors  # xyzt
    itk_kernel_radius = kernel_radius
    output_origin = [
        val + radius * spacing
        for val, spacing, radius in zip(input_origin, input_spacing, itk_kernel_radius)
    ]
    output_spacing = [s * f for s, f in zip(input_spacing, itk_shrink_factors)]
    output_size = [
        max(0, int((image_len - 2 * radius) / shrink_factor))
        for image_len, radius, shrink_factor in zip(
            itk.size(image), itk_kernel_radius, itk_shrink_factors
        )
    ]

    # Optionally run accelerated smoothing with itk-vkfft
    if gaussian_filter_name == "VkDiscreteGaussianImageFilter":
        smoothing_filter_template = itk.VkDiscreteGaussianImageFilter
    elif gaussian_filter_name == "DiscreteGaussianImageFilter":
        smoothing_filter_template = itk.DiscreteGaussianImageFilter
    else:
        msg = f"Unsupported gaussian_filter {gaussian_filter_name}"
        raise ValueError(msg)

    # Construct pipeline
    smoothing_filter = smoothing_filter_template.New(
        image, sigma_array=sigma_values, use_image_spacing=False
    )

    if interpolator_name == "LinearInterpolateImageFunction":
        interpolator_instance = itk.LinearInterpolateImageFunction.New(
            smoothing_filter.GetOutput()
        )
    elif interpolator_name == "LabelImageGaussianInterpolateImageFunction":
        interpolator_instance = itk.LabelImageGaussianInterpolateImageFunction.New(
            smoothing_filter.GetOutput()
        )
        # Similar approach as compute_sigma
        # Ref: https://link.springer.com/content/pdf/10.1007/978-3-319-24571-3_81.pdf
        sigma = [s * 0.7355 for s in output_spacing]
        sigma_max = max(sigma)
        interpolator_instance.SetSigma(sigma)
        interpolator_instance.SetAlpha(sigma_max * 2.5)
    else:
        msg = f"Unsupported interpolator_name {interpolator_name}"
        raise ValueError(msg)

    shrink_filter = itk.ResampleImageFilter.New(
        smoothing_filter.GetOutput(),
        interpolator=interpolator_instance,
        size=output_size,
        output_spacing=output_spacing,
        output_origin=output_origin,
    )
    shrink_filter.Update()

    return np.asarray(shrink_filter.GetOutput())


def _downsample_itk_bin_shrink(
    ngff_image: NgffImage, default_chunks, out_chunks, scale_factors
):
    import itk

    multiscales = [
        ngff_image,
    ]
    previous_image = ngff_image
    dims = ngff_image.dims
    previous_dim_factors = {d: 1 for d in dims}
    spatial_dims = [dim for dim in dims if dim in _spatial_dims]
    spatial_dims = _image_dims[: len(spatial_dims)]
    for scale_factor in scale_factors:
        dim_factors = _dim_scale_factors(dims, scale_factor, previous_dim_factors)
        previous_dim_factors = _update_previous_dim_factors(
            scale_factor, spatial_dims, previous_dim_factors
        )
        previous_image = _align_chunks(previous_image, default_chunks, dim_factors)

        translation, scale = _next_scale_metadata(
            previous_image, dim_factors, spatial_dims
        )

        # Blocks 0, ..., N-2 have the same shape
        block_0_input = _get_block(previous_image, 0)
        next_block_0_shape = _next_block_shape(
            previous_image, dim_factors, spatial_dims, block_0_input
        )
        block_0_size = []
        for dim in spatial_dims:
            if dim in previous_image.dims:
                block_0_size.append(block_0_input.shape[previous_image.dims.index(dim)])
            else:
                block_0_size.append(1)
        block_0_size.reverse()

        # Block N-1 may be smaller than preceding blocks
        block_neg1_input = _get_block(previous_image, -1)
        next_block_neg1_shape = _next_block_shape(
            previous_image, dim_factors, spatial_dims, block_neg1_input
        )

        shrink_factors = [dim_factors[sd] for sd in spatial_dims]

        dtype = block_0_input.dtype

        output_chunks = list(previous_image.data.chunks)
        output_chunks_start = 0
        while previous_image.dims[output_chunks_start] not in _spatial_dims:
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

        downscaled_array = map_blocks(
            itk.bin_shrink_image_filter,
            previous_image.data,
            shrink_factors=shrink_factors,
            dtype=dtype,
            chunks=output_chunks,
        )
        out_chunks_list = []
        for dim in dims:
            if dim in out_chunks:
                out_chunks_list.append(out_chunks[dim])
            else:
                out_chunks_list.append(1)
        downscaled_array = downscaled_array.rechunk(tuple(out_chunks_list))

        previous_image = NgffImage(downscaled_array, dims, scale, translation)
        multiscales.append(previous_image)

    return multiscales


def _downsample_itk_gaussian(
    ngff_image: NgffImage, default_chunks, out_chunks, scale_factors
):
    import itk
    from itkwasm_downsample import gaussian_kernel_radius

    # Optionally run accelerated smoothing with itk-vkfft
    if "VkFFTBackend" in dir(itk):
        gaussian_filter_name = "VkDiscreteGaussianImageFilter"
    else:
        gaussian_filter_name = "DiscreteGaussianImageFilter"

    interpolator_name = "LinearInterpolateImageFunction"

    multiscales = [
        ngff_image,
    ]
    previous_image = ngff_image
    dims = ngff_image.dims
    previous_dim_factors = {d: 1 for d in dims}
    spatial_dims = [dim for dim in dims if dim in _spatial_dims]
    spatial_dims = _image_dims[: len(spatial_dims)]
    for scale_factor in scale_factors:
        dim_factors = _dim_scale_factors(dims, scale_factor, previous_dim_factors)
        previous_dim_factors = _update_previous_dim_factors(
            scale_factor, spatial_dims, previous_dim_factors
        )
        previous_image = _align_chunks(previous_image, default_chunks, dim_factors)

        translation, scale = _next_scale_metadata(
            previous_image, dim_factors, spatial_dims
        )

        # Blocks 0, ..., N-2 have the same shape
        block_0_input = _get_block(previous_image, 0)
        next_block_0_shape = _next_block_shape(
            previous_image, dim_factors, spatial_dims, block_0_input
        )
        block_0_size = []
        for dim in spatial_dims:
            if dim in previous_image.dims:
                block_0_size.append(block_0_input.shape[previous_image.dims.index(dim)])
            else:
                block_0_size.append(1)
        block_0_size.reverse()

        # Block N-1 may be smaller than preceding blocks
        block_neg1_input = _get_block(previous_image, -1)
        next_block_neg1_shape = _next_block_shape(
            previous_image, dim_factors, spatial_dims, block_neg1_input
        )

        # pixel units
        # Compute metadata for region splitting
        shrink_factors = [dim_factors[sd] for sd in spatial_dims]
        sigma_values = _compute_sigma(shrink_factors)
        kernel_radius = gaussian_kernel_radius(size=block_0_size, sigma=sigma_values)

        dtype = block_0_input.dtype

        output_chunks = list(previous_image.data.chunks)
        output_chunks_start = 0
        while previous_image.dims[output_chunks_start] not in _spatial_dims:
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

        if "t" in previous_image.dims:
            t_index = previous_image.dims.index("t")

        if "t" in previous_image.dims:
            all_timepoints = []
            for timepoint in range(previous_image.data.shape[t_index]):
                data = take(previous_image.data, timepoint, t_index)

                downscaled_timepoint = map_overlap(
                    _itk_blur_and_downsample,
                    data,
                    gaussian_filter_name=gaussian_filter_name,
                    interpolator_name=interpolator_name,
                    shrink_factors=shrink_factors,
                    sigma_values=sigma_values,
                    kernel_radius=kernel_radius,
                    dtype=dtype,
                    depth=dict(enumerate(np.flip(kernel_radius))),  # overlap is in tzyx
                    boundary="nearest",
                    trim=False,  # Overlapped region is trimmed in blur_and_downsample to output size
                    chunks=output_chunks,
                )
                expanded = expand_dims(downscaled_timepoint, t_index)
                all_timepoints.append(expanded)
            downscaled_array = concatenate(all_timepoints, t_index)
        else:
            data = previous_image.data
            downscaled_array = map_overlap(
                _itk_blur_and_downsample,
                data,
                gaussian_filter_name=gaussian_filter_name,
                interpolator_name=interpolator_name,
                shrink_factors=shrink_factors,
                sigma_values=sigma_values,
                kernel_radius=kernel_radius,
                dtype=dtype,
                depth=dict(enumerate(np.flip(kernel_radius))),  # overlap is in tzyx
                boundary="nearest",
                trim=False,  # Overlapped region is trimmed in blur_and_downsample to output size
                chunks=output_chunks,
            )

        out_chunks_list = []
        for dim in dims:
            if dim in out_chunks:
                out_chunks_list.append(out_chunks[dim])
            else:
                out_chunks_list.append(1)
        downscaled_array = downscaled_array.rechunk(tuple(out_chunks_list))

        previous_image = NgffImage(downscaled_array, dims, scale, translation)
        multiscales.append(previous_image)

    return multiscales


# todo
# def _downsample_itk_label(
#     current_input,
#     default_chunks,
#     out_chunks,
#     scale_factors,
#     data_objects,
#     image,
# ):
# Uses the LabelImageGaussianInterpolateImageFunction. More appropriate for integer label images.
# import itk

# gaussian_filter_name = "DiscreteGaussianImageFilter"
# interpolator_name = "LabelImageGaussianInterpolateImageFunction"

# for _factor_index, scale_factor in enumerate(scale_factors):
#     dim_factors = _dim_scale_factors(image.dims, scale_factor)
#     current_input = _align_chunks(current_input, default_chunks, dim_factors)

#     shrink_factors = [dim_factors[sf] for sf in _image_dims if sf in dim_factors]

#     # Compute metadata for region splitting

#     # Blocks 0, ..., N-2 have the same shape
#     block_0_input = _get_block(current_input, 0)
#     # Block N-1 may be smaller than preceding blocks
#     block_neg1_input = _get_block(current_input, -1)

#     # Compute overlap for Gaussian blurring for all blocks
#     block_0_image = itk.image_from_xarray(block_0_input)
#     input_spacing = itk.spacing(block_0_image)
#     sigma_values = _compute_sigma(shrink_factors)
#     kernel_radius = _compute_itk_gaussian_kernel_radius(
#         itk.size(block_0_image), sigma_values
#     )

#     # Compute output size and spatial metadata for blocks 0, .., N-2
#     filt = itk.BinShrinkImageFilter.New(
#         block_0_image, shrink_factors=shrink_factors
#     )
#     filt.UpdateOutputInformation()
#     block_output = filt.GetOutput()
#     block_0_output_spacing = block_output.GetSpacing()
#     block_0_output_origin = block_output.GetOrigin()

#     block_0_scale = {
#         _image_dims[i]: s for (i, s) in enumerate(block_0_output_spacing)
#     }
#     block_0_translation = {
#         _image_dims[i]: s for (i, s) in enumerate(block_0_output_origin)
#     }
#     dtype = block_output.dtype

#     computed_size = [
#         int(block_len / shrink_factor)
#         for block_len, shrink_factor in zip(itk.size(block_0_image), shrink_factors)
#     ]
#     assert all(
#         itk.size(block_output)[dim] == computed_size[dim]
#         for dim in range(block_output.ndim)
#     )
#     output_chunks = list(current_input.chunks)
#     for i, c in enumerate(output_chunks):
#         output_chunks[i] = [
#             block_output.shape[i],
#         ] * len(c)

#     # Compute output size for block N-1
#     block_neg1_image = itk.image_from_xarray(block_neg1_input)
#     filt.SetInput(block_neg1_image)
#     filt.UpdateOutputInformation()
#     block_output = filt.GetOutput()
#     computed_size = [
#         int(block_len / shrink_factor)
#         for block_len, shrink_factor in zip(
#             itk.size(block_neg1_image), shrink_factors
#         )
#     ]
#     assert all(
#         itk.size(block_output)[dim] == computed_size[dim]
#         for dim in range(block_output.ndim)
#     )
#     for i in range(len(output_chunks)):
#         output_chunks[i][-1] = block_output.shape[i]
#         output_chunks[i] = tuple(output_chunks[i])
#     output_chunks = tuple(output_chunks)

#     downscaled_array = map_overlap(
#         _itk_blur_and_downsample,
#         current_input.data,
#         gaussian_filter_name=gaussian_filter_name,
#         interpolator_name=interpolator_name,
#         shrink_factors=shrink_factors,
#         sigma_values=sigma_values,
#         kernel_radius=kernel_radius,
#         dtype=dtype,
#         depth=dict(enumerate(np.flip(kernel_radius))),  # overlap is in tzyx
#         boundary="nearest",
#         trim=False,  # Overlapped region is trimmed in blur_and_downsample to output size
#     ).compute()

#     # todo
#     # downscaled = to_spatial_image(
#     #     downscaled_array,
#     #     dims=image.dims,
#     #     scale=block_0_scale,
#     #     translation=block_0_translation,
#     #     name=current_input.name,
#     #     axis_names={
#     #         d: image.coords[d].attrs.get("long_name", d) for d in image.dims
#     #     },
#     #     axis_units={d: image.coords[d].attrs.get("units", "") for d in image.dims},
#     #     t_coords=image.coords.get("t", None),
#     #     c_coords=image.coords.get("c", None),
#     # )
#     # downscaled = downscaled.chunk(out_chunks)
#     # data_objects[f"scale{factor_index+1}"] = downscaled.to_dataset(
#     #     name=image.name, promote_attrs=True
#     # )
#     # current_input = downscaled

# return data_objects
