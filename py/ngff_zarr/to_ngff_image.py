# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from collections.abc import MutableMapping
from typing import Hashable, Mapping, Optional, Sequence, Union, List

import dask
from dask.array.core import Array as DaskArray
from numpy.typing import ArrayLike

try:
    from zarr.core import Array as ZarrArray
except ImportError:
    from zarr.core.array import Array as ZarrArray

from .methods._support import _spatial_dims
from .ngff_image import NgffImage
from .v04.zarr_metadata import SupportedDims, Units
from .v06.zarr_metadata import CoordinateSystem, Transform, TransformSequence, Axis, Scale, Translation


def to_ngff_image(
    data: Union[ArrayLike, MutableMapping, str, ZarrArray],
    dims: Optional[Sequence[SupportedDims]] = None,
    scale: Optional[Union[Mapping[Hashable, float]]] = None,
    translation: Optional[Union[Mapping[Hashable, float]]] = None,
    coordinateTransformation: Optional[Union[List[Transform], Transform]] = None,
    name: str = "image",
    coordinate_system_name: str = "physical",
    axes_units: Optional[Mapping[str, Units]] = None,
) -> NgffImage:
    """
    Create an image with pixel array and metadata to following the OME-NGFF data model.

    :param data: Multi-dimensional array that provides the image pixel values. It can be a numpy.ndarray
         or another type that behaves like a numpy.ndarray, i.e. an ArrayLike.
         If a ZarrArray, MutableMapping, or str, it will be loaded into Dask lazily
         as a zarr Array.
    :type  data: ArrayLike, ZarrArray, MutableMapping, str

    :param dims: Tuple specifying the data dimensions.
        Values should drawn from: {'t', 'z', 'y', 'x', 'c'} for time, third spatial direction
        second spatial direction, first spatial dimension, and channel or
        component, respectively spatial dimension, and time, respectively.
    :type  dims: sequence of hashable, optional

    :param scale: Pixel spacing for the spatial dims
    :type  scale: dict of floats, optional

    :param translation: Origin or offset of the center of the first pixel.
    :type  translation: dict of floats, optional

    :param name: Name of the resulting image
    :type  name: str, optional

    :param axes_units: Units to associate with the axes. Should be drawn from UDUNITS-2, enumerated at
        https://ngff.openmicroscopy.org/latest/#axes-md
    :type  axes_units: dict of str, optional

    :return: Representation of an image (pixel data + metadata) for a single scale of an NGFF-OME-Zarr multiscale dataset
    :rtype: NgffImage
    """

    ndim = data.ndim
    if dims is None:
        if ndim < 4:
            dims = ("z", "y", "x")[-ndim:]
        elif ndim < 5:
            dims = ("z", "y", "x", "c")
        elif ndim < 6:
            dims = ("t", "z", "y", "x", "c")
        else:
            raise ValueError("Unsupported dimension: " + str(ndim))
    else:
        _supported_dims = {"c", "x", "y", "z", "t"}
        if not set(dims).issubset(_supported_dims):
            msg = "dims not valid"
            raise ValueError(msg)

    axes = []
    for d in dims:
        if d in _spatial_dims:
            axes.append(Axis(name=d, type='space'))
        elif d == 'c':
            axes.append(Axis(name=d, type='channel'))
        elif d == 't':
            axes.append(Axis(name=d, type='time'))

    if axes_units is not None:
        for ax in axes:
            if ax.name in axes_units:
                ax.unit = axes_units[ax.name]

    output_coordinate_system = CoordinateSystem(
        name=coordinate_system_name,
        axes=axes)

    if scale is None:
        scale ={dim: 1.0 for dim in dims if dim in _spatial_dims}
    if translation is None:
        translation = {d: 0.0 for dim in dims if dim in _spatial_dims}

    # assert correct scale factor ordering
    scale = Scale([float(scale[d]) for d in _spatial_dims if d in scale], name="scale")
    translation = Translation([float(translation[d]) for d in _spatial_dims if d in translation], name="translation")

    # passed transformation supersedes scale and translation from old spec
    if coordinateTransformation is None:
        coordinateTransformation = TransformSequence([scale, translation], name="transforms", output=output_coordinate_system)

    if not isinstance(data, DaskArray):
        if isinstance(data, (ZarrArray, str, MutableMapping)):
            data = dask.array.from_zarr(data)
        else:
            data = dask.array.from_array(data)

    return NgffImage(
        data=data,
        coordinateTransformations=coordinateTransformation,
        name=name,
    )
