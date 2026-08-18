# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Resample a moving image onto a fixed image grid, one block at a time."""

import numpy as np

from .itk_transform_resample_bounding_box import (
    _as_itk_transform_list,
    _check_geometry,
    _itk_direction,
    _metadata_only_itk_image,
    _spatial_dims,
    itk_transform_resample_bounding_box,
)
from .ngff_image import NgffImage

_INTERPOLATORS = (
    "linear",
    "nearest_neighbor",
    "label_image",
    "b_spline",
    "windowed_sinc",
    "gaussian",
)

#: Padding each interpolator needs for a block-wise resample to reproduce an
#: undecomposed one exactly. Measured rather than assumed: see
#: ``test_default_padding_reproduces_an_undecomposed_resample``. The wide
#: values are not kernel radii. ``b_spline`` prefilters coefficients over the
#: whole image it is handed, so cropping perturbs them everywhere and the
#: perturbation only decays with distance; 16 is where it falls below float32
#: resolution.
_INTERPOLATOR_PADDING = {
    "linear": 1,
    "nearest_neighbor": 1,
    "label_image": 1,
    "gaussian": 2,
    "windowed_sinc": 3,
    "b_spline": 16,
}


def _component_type(out_dtype) -> str:
    from itkwasm import FloatTypes, IntTypes

    mapping = {
        "uint8": IntTypes.UInt8,
        "int8": IntTypes.Int8,
        "uint16": IntTypes.UInt16,
        "int16": IntTypes.Int16,
        "uint32": IntTypes.UInt32,
        "int32": IntTypes.Int32,
        "uint64": IntTypes.UInt64,
        "int64": IntTypes.Int64,
        "float32": FloatTypes.Float32,
        "float64": FloatTypes.Float64,
    }
    name = np.dtype(out_dtype).name
    if name not in mapping:
        msg = f"unsupported dtype {name!r} for resampling"
        raise ValueError(msg)
    return mapping[name]


def _block_grid(fixed: NgffImage, starts: dict, shape: tuple) -> NgffImage:
    """A geometry-only stand-in for one output block of the fixed grid."""
    translation = {
        dim: fixed.translation[dim] + starts[dim] * fixed.scale[dim]
        for dim in fixed.dims
    }
    return NgffImage(
        data=np.empty(shape, dtype=np.uint8),
        dims=tuple(fixed.dims),
        scale=dict(fixed.scale),
        translation=translation,
        name=fixed.name,
        axes_units=fixed.axes_units,
        axes_orientations=fixed.axes_orientations,
        axes_types=fixed.axes_types,
    )


def _resample_block(
    block,
    *,
    block_info,
    transform_list,
    fixed: NgffImage,
    moving: NgffImage,
    padding: int,
    interpolator: str,
    default_value: float,
    out_dtype,
):
    from itkwasm_downsample import resample_to_reference

    from .ngff_image_to_itk_image import ngff_image_to_itk_image

    location = block_info[0]["array-location"]
    starts = {dim: start for dim, (start, _stop) in zip(fixed.dims, location)}
    grid = _block_grid(fixed, starts, block.shape)

    region = itk_transform_resample_bounding_box(
        transform_list, grid, moving, padding=padding
    )
    crop = region.crop(moving)
    if crop is None:
        return np.full(block.shape, default_value, dtype=out_dtype)

    itk_dims = list(reversed(_spatial_dims(fixed)))
    reference = _metadata_only_itk_image(grid, itk_dims, _itk_direction(grid, itk_dims))
    # resample_to_reference reads only the reference geometry, but it aborts
    # when the component type disagrees with the moving image, so align it.
    reference.imageType.componentType = _component_type(out_dtype)

    resampled = resample_to_reference(
        ngff_image_to_itk_image(crop, wasm=True),
        reference,
        transform=transform_list,
        interpolator=interpolator,
        default_value=float(default_value),
    )
    return np.asarray(resampled.data).reshape(block.shape).astype(out_dtype, copy=False)


def itk_transform_resample(
    transform,
    fixed: NgffImage,
    moving: NgffImage,
    padding: int | None = None,
    interpolator: str = "linear",
    default_value: float = 0.0,
) -> NgffImage:
    """Resample ``moving`` onto the grid of ``fixed`` through ``transform``.

    The result is a lazy :class:`NgffImage`: nothing is read or computed until
    the returned Dask array is. Each output block is resampled on its own, and
    a block only ever materializes the region of ``moving`` its own resample
    reads, as reported by
    :func:`ngff_zarr.itk_transform_resample_bounding_box`. The full moving
    image is never loaded, which is what makes this usable when it is larger
    than memory, remote, or chunked.

    Resampling runs through ``itkwasm-downsample``, so no native ITK build is
    required and the result is identical to one across platforms.

    :param transform: An ``itk.Transform`` (including the ``CompositeTransform``
        an Elastix registration returns), or an ITK-Wasm ``Transform`` /
        ``TransformList``. It maps *fixed* points into *moving* space.

    :param fixed: The image whose grid defines the output. Geometry only, its
        pixels are never read.
    :type  fixed: NgffImage

    :param moving: The image to sample.
    :type  moving: NgffImage

    :param padding: Pixels of padding added per side when computing each
        block's source region. Defaults to what ``interpolator`` reads beyond
        the continuous index bound, so the two stay in step.
    :type  padding: int | None

    :param interpolator: One of ``"linear"`` (default),
        ``"nearest_neighbor"``, ``"label_image"``, ``"b_spline"``,
        ``"windowed_sinc"`` or ``"gaussian"``.
    :type  interpolator: str

    :param default_value: Value written where a block samples outside
        ``moving``.
    :type  default_value: float

    :return: The resampled image, carrying the geometry of ``fixed`` and the
        dtype of ``moving``.
    :rtype: NgffImage
    """
    import dask.array as da

    if interpolator not in _INTERPOLATORS:
        msg = f"interpolator must be one of {_INTERPOLATORS}, got {interpolator!r}"
        raise ValueError(msg)
    if padding is None:
        padding = _INTERPOLATOR_PADDING[interpolator]
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        msg = f"padding must be a non-negative integer, got {padding!r}"
        raise ValueError(msg)

    fixed_spatial = _spatial_dims(fixed)
    moving_spatial = _spatial_dims(moving)
    if fixed_spatial != moving_spatial:
        msg = (
            f"fixed image has spatial dims {tuple(fixed_spatial)} but moving "
            f"image has {tuple(moving_spatial)}; they must match"
        )
        raise ValueError(msg)
    if len(fixed_spatial) < 2:
        msg = (
            f"images have {len(fixed_spatial)} spatial dims "
            f"{tuple(fixed_spatial)}; only 2 and 3 are supported"
        )
        raise ValueError(msg)
    if tuple(fixed.dims) != tuple(moving.dims):
        msg = (
            f"fixed image has dims {tuple(fixed.dims)} but moving image has "
            f"{tuple(moving.dims)}; they must match"
        )
        raise ValueError(msg)
    for label, image in (("fixed", fixed), ("moving", moving)):
        _check_geometry(label, image, fixed_spatial)

    dtype = moving.data.dtype
    transform_list = _as_itk_transform_list(transform)

    data = da.map_blocks(
        _resample_block,
        fixed.data,
        dtype=dtype,
        transform_list=transform_list,
        fixed=fixed,
        moving=moving,
        padding=padding,
        interpolator=interpolator,
        default_value=default_value,
        out_dtype=dtype,
    )

    return NgffImage(
        data=data,
        dims=tuple(fixed.dims),
        scale=dict(fixed.scale),
        translation=dict(fixed.translation),
        name=moving.name,
        axes_units=fixed.axes_units,
        axes_orientations=fixed.axes_orientations,
        axes_types=fixed.axes_types,
        channel_names=moving.channel_names,
        channel_colors=moving.channel_colors,
    )
