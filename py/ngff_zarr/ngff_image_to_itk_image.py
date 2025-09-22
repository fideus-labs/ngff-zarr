# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Optional

import numpy as np
from dask.array.core import Array as DaskArray

from .ngff_image import NgffImage
from .methods._support import _channel_dim_last


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


def ngff_image_to_itk_image(
    ngff_image: NgffImage,
    wasm: bool = True,
    t_index: Optional[int] = None,
    c_index: Optional[int] = None,
):
    """Convert a NgffImage to an ITK image."""
    from itkwasm import IntTypes, PixelTypes

    if t_index is not None and "t" in ngff_image.dims:
        t_dim_index = ngff_image.dims.index("t")
        new_dims = list(ngff_image.dims)
        new_dims.remove("t")
        new_dims = tuple(new_dims)
        new_scale = {dim: ngff_image.scale[dim] for dim in new_dims}
        new_translation = {dim: ngff_image.translation[dim] for dim in new_dims}
        new_axes_units = {dim: ngff_image.axes_units[dim] for dim in new_dims}
        if isinstance(ngff_image.data, DaskArray):
            from dask.array import take

            new_data = take(ngff_image.data, t_index, axis=t_dim_index)
        else:
            new_data = ngff_image.data.take(t_index, axis=t_dim_index)
        ngff_image = NgffImage(
            data=new_data,
            dims=new_dims,
            name=ngff_image.name,
            scale=new_scale,
            translation=new_translation,
            axes_units=new_axes_units,
        )

    if c_index is not None and "c" in ngff_image.dims:
        c_dim_index = ngff_image.dims.index("c")
        new_dims = list(ngff_image.dims)
        new_dims.remove("c")
        new_dims = tuple(new_dims)
        new_scale = {dim: ngff_image.scale[dim] for dim in new_dims}
        new_translation = {dim: ngff_image.translation[dim] for dim in new_dims}
        new_axes_units = {dim: ngff_image.axes_units[dim] for dim in new_dims}
        if isinstance(ngff_image.data, DaskArray):
            from dask.array import take

            new_data = take(ngff_image.data, c_index, axis=c_dim_index)
        else:
            new_data = ngff_image.data.take(c_index, axis=c_dim_index)
        ngff_image = NgffImage(
            data=new_data,
            dims=new_dims,
            name=ngff_image.name,
            scale=new_scale,
            translation=new_translation,
            axes_units=new_axes_units,
        )

    ngff_image = _channel_dim_last(ngff_image)

    dims = ngff_image.dims
    itk_dimension_names = {"x", "y", "z", "t"}
    itk_dims = [dim for dim in dims if dim in itk_dimension_names]
    itk_dims.sort()
    if "t" in itk_dims:
        itk_dims.remove("t")
        itk_dims.append("t")
    spacing = [ngff_image.scale[dim] for dim in itk_dims]
    origin = [ngff_image.translation[dim] for dim in itk_dims]
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

    data = np.asarray(ngff_image.data)

    image_dict = {
        "imageType": imageType,
        "name": ngff_image.name,
        "origin": origin,
        "spacing": spacing,
        "direction": np.eye(dimension),
        "size": size,
        "metadata": {},
        "data": data,
    }

    if wasm:
        from itkwasm import Image

        return Image(**image_dict)

    import itk

    return itk.image_from_dict(image_dict)
