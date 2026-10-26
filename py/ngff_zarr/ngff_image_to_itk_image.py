# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT

from dataclasses import replace

import numpy as np
from dask.array.core import Array as DaskArray

from .ngff_image import NgffImage


def _dtype_to_component_type(dtype):
    from itkwasm import FloatTypes, IntTypes

    if dtype == np.uint8:
        return IntTypes.UInt8
    if dtype == np.int8:
        return IntTypes.Int8
    if dtype == np.uint16:
        return IntTypes.UInt16
    if dtype == np.int16:
        return IntTypes.Int16
    if dtype == np.uint32:
        return IntTypes.UInt32
    if dtype == np.int32:
        return IntTypes.Int32
    if dtype == np.uint64:
        return IntTypes.UInt64
    if dtype == np.int64:
        return IntTypes.Int64
    if dtype == np.float32:
        return FloatTypes.Float32
    if dtype == np.float64:
        return FloatTypes.Float64
    msg = f"Unsupported dtype {dtype}"
    raise ValueError(msg)


def _select(ngff_image: NgffImage, dim: str, index: int) -> NgffImage:
    """*ngff_image* at *index* along *dim*, without that dim."""
    axis = ngff_image.dims.index(dim)
    if isinstance(ngff_image.data, DaskArray):
        from dask.array import take

        data = take(ngff_image.data, index, axis=axis)
    else:
        data = ngff_image.data.take(index, axis=axis)

    def without(mapping):
        if mapping is None:
            return None
        return {key: value for key, value in mapping.items() if key != dim}

    return replace(
        ngff_image,
        data=data,
        dims=tuple(d for d in ngff_image.dims if d != dim),
        scale=without(ngff_image.scale),
        translation=without(ngff_image.translation),
        axes_units=without(ngff_image.axes_units),
        axes_orientations=without(ngff_image.axes_orientations),
        axes_types=without(ngff_image.axes_types),
    )


def ngff_image_to_itk_image(
    ngff_image: NgffImage,
    wasm: bool = True,
    t_index: int | None = None,
    c_index: int | None = None,
):
    """Convert a NgffImage to an ITK image."""
    from itkwasm import IntTypes, PixelTypes

    from .resample_bounding_box import _itk_direction

    if t_index is not None and "t" in ngff_image.dims:
        ngff_image = _select(ngff_image, "t", t_index)
    if c_index is not None and "c" in ngff_image.dims:
        ngff_image = _select(ngff_image, "c", c_index)

    dims = ngff_image.dims
    itk_dimension_names = {"x", "y", "z", "t"}
    itk_dims = [dim for dim in dims if dim in itk_dimension_names]
    itk_dims.sort()
    if "t" in itk_dims:
        itk_dims.remove("t")
        itk_dims.append("t")
    # to_ngff_image gives scale and translation to x, y and z only.
    spacing = [ngff_image.scale.get(dim, 1.0) for dim in itk_dims]
    origin = [ngff_image.translation.get(dim, 0.0) for dim in itk_dims]
    size = [ngff_image.data.shape[dims.index(d)] for d in itk_dims]
    dimension = len(itk_dims)

    componentType = _dtype_to_component_type(ngff_image.data.dtype)

    components = 1
    pixelType = PixelTypes.Scalar
    if "c" in dims:
        components = ngff_image.data.shape[dims.index("c")]
        if components == 3 and componentType == IntTypes.UInt8:
            pixelType = PixelTypes.RGB
        else:
            pixelType = PixelTypes.VariableLengthVector
    imageType = {
        "dimension": dimension,
        "componentType": str(componentType),
        "pixelType": str(pixelType),
        "components": components,
    }

    # ITK arrays hold the axes in reverse ITK order, components last.
    array_dims = list(reversed(itk_dims)) + (["c"] if "c" in dims else [])
    data = np.asarray(ngff_image.data).transpose([dims.index(d) for d in array_dims])

    # Only spatial dims (not "t") carry an anatomical direction.
    spatial_itk_dims = [d for d in itk_dims if d != "t"]
    n_spatial = len(spatial_itk_dims)
    direction = np.eye(dimension)
    direction[:n_spatial, :n_spatial] = _itk_direction(ngff_image, spatial_itk_dims)

    image_dict = {
        "imageType": imageType,
        "name": ngff_image.name,
        "origin": origin,
        "spacing": spacing,
        "direction": direction,
        "size": size,
        "metadata": {},
        "data": data,
    }

    if wasm:
        from itkwasm import Image

        return Image(**image_dict)

    import itk

    return itk.image_from_dict(image_dict)
