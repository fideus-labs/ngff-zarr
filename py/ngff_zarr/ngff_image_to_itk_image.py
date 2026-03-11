# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT

import numpy as np
from dask.array.core import Array as DaskArray

from .methods._support import _channel_dim_last
from .ngff_image import NgffImage
from .rfc4 import anatomical_orientation_to_itk_direction


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
    t_index: int | None = None,
    c_index: int | None = None,
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
        axes_units = ngff_image.axes_units
        if axes_units is None:
            new_axes_units = None
        else:
            new_axes_units = {
                dim: axes_units.get(dim) for dim in new_dims if dim in axes_units
            }
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
        axes_units = ngff_image.axes_units
        if axes_units is None:
            new_axes_units = None
        else:
            new_axes_units = {
                dim: axes_units.get(dim) for dim in new_dims if dim in axes_units
            }
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

    # Build direction matrix from anatomical orientations when available.
    # itk_dims is in ITK physical-space order [x, y, z (, t)].
    # Column j of the direction matrix corresponds to itk_dims[j].
    direction = np.eye(dimension)
    orientations = ngff_image.axes_orientations
    if orientations:
        # Only spatial dims (not "t") contribute to the direction matrix.
        spatial_itk_dims = [d for d in itk_dims if d != "t"]
        columns: list[list[float]] = []
        use_orientations = True
        for dim in spatial_itk_dims:
            ori = orientations.get(dim)
            if ori is None:
                use_orientations = False
                break
            col = anatomical_orientation_to_itk_direction(ori.value)
            if col is None:
                # Non-LPS orientation: fall back to identity
                use_orientations = False
                break
            columns.append(col)

        if use_orientations and columns:
            n_spatial = len(spatial_itk_dims)
            for col_idx in range(n_spatial):
                for row in range(n_spatial):
                    direction[row, col_idx] = columns[col_idx][row]

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
