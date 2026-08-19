# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import math
from collections.abc import Hashable, Mapping, MutableMapping, Sequence

import dask
from dask.array.core import Array as DaskArray
from numpy.typing import ArrayLike

try:
    # Zarr v3 imports
    from zarr.core.array import Array as ZarrArray
    from zarr.core.group import Group as ZarrGroup
except ImportError:
    # Zarr v2 imports
    from zarr.core import Array as ZarrArray
    from zarr.hierarchy import Group as ZarrGroup

from ._v2_store_reader import V2Group, V3Group
from ._zarrista_utils import (
    LocalZarrGroup,
    is_zarrista_array,
    open_lazy_array,
    zarrista_array_to_dask,
)
from .methods._support import _spatial_dims
from .ngff_image import NgffImage
from .v04.zarr_metadata import SupportedDims, Units

# Group node types across the read backends: zarr-python, the compat-layer
# local reader, and the pure-Python mapping-store reader.
_group_types = (ZarrGroup, LocalZarrGroup, V2Group, V3Group)


def _extract_array_from_group(group):
    """Extract the full-resolution array from a zarr group node.

    For multi-level TIFFs (e.g., pyramidal OME-TIFF), tifffile returns a zarr
    group containing multiple arrays (one per resolution level). This function
    extracts the full-resolution array, using multiscales metadata if available.
    Accepts any backend's group node (see ``_group_types``).
    """
    # Try using multiscales metadata (OME-NGFF style)
    if "multiscales" in group.attrs:
        multiscales = group.attrs["multiscales"]
        if multiscales and len(multiscales) > 0:
            datasets = multiscales[0].get("datasets", [])
            if datasets:
                # First dataset is typically the full resolution
                path = datasets[0].get("path", "0")
                if path in group:
                    return group[path]

    # Fallback: find the largest array by size
    largest_key = None
    largest_size = 0
    for key in group.keys():
        item = group[key]
        shape = getattr(item, "shape", None)
        if shape is not None and math.prod(shape) > largest_size:
            largest_size = math.prod(shape)
            largest_key = key

    if largest_key is not None:
        return group[largest_key]

    raise ValueError(
        "Cannot convert zarr.Group to NgffImage: no arrays found. "
        "Pass a specific array from the Group instead."
    )


def _as_dask_array(data) -> DaskArray:
    """Convert *data* to a dask array, lazily for every array-handle form.

    The read counterpart of the write-side engine dispatch: compat-layer
    ``LocalZarrArray`` and v2-reader ``V2Array`` nodes expose ``to_dask()``,
    raw zarrista arrays wrap through the adapter, and ``str``/mapping stores
    dispatch through :func:`~._zarrista_utils.open_lazy_array`. zarr-python
    array objects keep the legacy ``dask.array.from_zarr`` engine; anything
    else is treated as an in-memory array.
    """
    if isinstance(data, DaskArray):
        return data
    if hasattr(data, "to_dask"):
        return data.to_dask()
    if is_zarrista_array(data):
        return zarrista_array_to_dask(data)
    if isinstance(data, ZarrArray):
        return dask.array.from_zarr(data)
    if isinstance(data, (str, MutableMapping)):
        return open_lazy_array(data)
    return dask.array.from_array(data)


def to_ngff_image(
    data: ArrayLike | MutableMapping | str | ZarrArray | ZarrGroup,
    dims: Sequence[SupportedDims] | None = None,
    scale: Mapping[Hashable, float] | None = None,
    translation: Mapping[Hashable, float] | None = None,
    name: str = "image",
    axes_units: Mapping[str, Units] | None = None,
    channel_names: Sequence[str] | None = None,
) -> NgffImage:
    """
    Create an image with pixel array and metadata to following the OME-NGFF data model.

    :param data: Multi-dimensional array that provides the image pixel
         values. It can be a numpy.ndarray or another type that behaves like
         a numpy.ndarray, i.e. an ArrayLike. If a ZarrArray, MutableMapping,
         or str, it will be loaded into Dask lazily as a zarr Array. If a
         ZarrGroup, the first array in the group will be used.
    :type  data: ArrayLike, ZarrArray, ZarrGroup, MutableMapping, str

    :param dims: Tuple specifying the data dimensions.
        Values should drawn from: {'t', 'z', 'y', 'x', 'c'} for time, third
        spatial direction, second spatial direction, first spatial dimension,
        and channel or component, respectively spatial dimension, and time,
        respectively.
    :type  dims: sequence of hashable, optional

    :param scale: Pixel spacing for the spatial dims
    :type  scale: dict of floats, optional

    :param translation: Origin or offset of the center of the first pixel.
    :type  translation: dict of floats, optional

    :param name: Name of the resulting image
    :type  name: str, optional

    :param axes_units: Units to associate with the axes. Should be drawn from
        UDUNITS-2, enumerated at
        https://ngff.openmicroscopy.org/latest/#axes-md
    :type  axes_units: dict of str, optional

    :param channel_names: Optional list of channel names. Length should match
        the number of channels in the data (size of 'c' dimension).
    :type  channel_names: sequence of str, optional

    :return: Representation of an image (pixel data + metadata) for a single
        scale of an NGFF-OME-Zarr multiscale dataset
    :rtype: NgffImage
    """

    # Handle zarr groups from multi-level TIFFs (e.g., pyramidal OME-TIFF)
    if isinstance(data, _group_types):
        data = _extract_array_from_group(data)

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

    if scale is None:
        scale = {dim: 1.0 for dim in dims if dim in _spatial_dims}

    if translation is None:
        translation = {dim: 0.0 for dim in dims if dim in _spatial_dims}

    data = _as_dask_array(data)

    return NgffImage(
        data=data,
        dims=dims,
        scale=scale,
        translation=translation,
        name=name,
        axes_units=axes_units,
        channel_names=list(channel_names) if channel_names else None,
    )
