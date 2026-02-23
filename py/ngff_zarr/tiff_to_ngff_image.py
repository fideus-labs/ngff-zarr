# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Convert TIFF files to OME-NGFF.

This module provides functions to convert TIFF files (including multi-series
OME-TIFF files) to NgffImage objects for further processing into OME-Zarr format.

When reading OME-TIFF files, physical size metadata (PhysicalSizeX/Y/Z and their
units) is extracted and used to set the scale and axes_units of the resulting
NgffImage.

Requires the `tifffile` package: pip install tifffile
"""

import re
import warnings
import xml.etree.ElementTree as ET
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import zarr

from .ngff_image import NgffImage
from .to_ngff_image import to_ngff_image

if TYPE_CHECKING:
    import tifffile

    from .multiscales import Multiscales

# Mapping from TIFF axis characters to NGFF-compatible dimension names
# Based on tifffile axis conventions:
#   X: width, Y: height, Z: depth, S: sample (RGB), C: channel, T: time
#   I: sequence, Q: other/unknown, R: reduction/tiles, etc.
TIFF_AXIS_TO_NGFF: dict[str, str | None] = {
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "c": "c",
    "s": "c",  # Sample (RGB interleaved) -> channel
    # Unsupported axes (will be dropped with warning)
    "i": None,  # Sequence
    "q": None,  # Other/unknown (often used for extra dimensions)
    "r": None,  # Reduction/tiles
    "a": None,  # Angle/rotation
    "p": None,  # Position
    "e": None,  # Exposure
    "l": None,  # Lifetime
    "h": None,  # Phase
}

# Channel-like dimensions that should be flattened into a single 'c' dimension
TIFF_CHANNEL_AXES = {"c", "s"}

# Spatial dimensions in NGFF
SPATIAL_DIMS = {"x", "y", "z"}

# Mapping from OME unit symbols/names to NGFF-compatible unit names
# Based on OME-XML specification and UDUNITS-2
OME_UNIT_TO_NGFF: dict[str, str] = {
    # Micrometers (most common in microscopy)
    "µm": "micrometer",
    "um": "micrometer",
    "micrometer": "micrometer",
    "micron": "micrometer",
    # Nanometers
    "nm": "nanometer",
    "nanometer": "nanometer",
    # Millimeters
    "mm": "millimeter",
    "millimeter": "millimeter",
    # Centimeters
    "cm": "centimeter",
    "centimeter": "centimeter",
    # Meters
    "m": "meter",
    "meter": "meter",
    # Angstroms
    "Å": "angstrom",
    "A": "angstrom",
    "angstrom": "angstrom",
    # Picometers
    "pm": "picometer",
    "picometer": "picometer",
}


def _normalize_unit(ome_unit: str | None) -> str | None:
    """
    Convert an OME unit string to an NGFF-compatible unit name.

    Parameters
    ----------
    ome_unit : str or None
        The unit string from OME-XML metadata.

    Returns
    -------
    str or None
        The NGFF-compatible unit name, or None if not recognized.
    """
    if ome_unit is None:
        return None
    # Try direct lookup first
    if ome_unit in OME_UNIT_TO_NGFF:
        return OME_UNIT_TO_NGFF[ome_unit]
    # Try lowercase
    lower_unit = ome_unit.lower()
    if lower_unit in OME_UNIT_TO_NGFF:
        return OME_UNIT_TO_NGFF[lower_unit]
    # Not recognized
    return None


def _map_tiff_axes_to_ngff(
    tiff_axes: str,
    tiff_shape: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[int, ...], list[int], list[int]]:
    """
    Map TIFF axes to NGFF-compatible dimensions, flattening channel-like dims.

    Parameters
    ----------
    tiff_axes : str
        TIFF axis string (e.g., 'ZYXS', 'TCYX').
    tiff_shape : Tuple[int, ...]
        Shape corresponding to the axes.

    Returns
    -------
    ngff_dims : Tuple[str, ...]
        NGFF-compatible dimension names.
    ngff_shape : Tuple[int, ...]
        Shape after processing (may flatten channel dims).
    channel_indices : List[int]
        Indices of channel-like dimensions in original axes.
    dropped_indices : List[int]
        Indices of dropped (unsupported) dimensions in original axes.

    Warns
    -----
    UserWarning
        If unsupported axes are encountered and dropped.
    """
    axes_lower = tiff_axes.lower()

    # Track channel-like dimensions for potential flattening
    channel_indices: list[int] = []
    channel_sizes: list[int] = []
    other_dims: list[tuple[int, str, int]] = []  # (orig_idx, ngff_dim, size)
    dropped_axes: list[str] = []
    dropped_indices: list[int] = []

    for i, (axis, size) in enumerate(zip(axes_lower, tiff_shape)):
        ngff_dim = TIFF_AXIS_TO_NGFF.get(axis)

        if axis in TIFF_CHANNEL_AXES:
            channel_indices.append(i)
            channel_sizes.append(size)
        elif ngff_dim is not None:
            other_dims.append((i, ngff_dim, size))
        else:
            # Unknown/unsupported axis - drop with warning
            dropped_axes.append(axis.upper())
            dropped_indices.append(i)

    if dropped_axes:
        supported = ", ".join(
            sorted(
                k.upper() for k in TIFF_AXIS_TO_NGFF if TIFF_AXIS_TO_NGFF[k] is not None
            )
        )
        warnings.warn(
            f"Dropping unsupported TIFF axes: {', '.join(dropped_axes)}. "
            f"Supported axes are: {supported}",
            UserWarning,
            stacklevel=3,
        )

    # Build output dimensions and shape
    ngff_dims_list: list[str] = []
    ngff_shape_list: list[int] = []

    # Sort other_dims by original index to maintain order
    other_dims.sort(key=lambda x: x[0])

    # Calculate total channel size (flatten multiple channel dims)
    # Only add channel dimension if there were actual channel-like axes
    has_channels = len(channel_sizes) > 0
    total_channels = reduce(lambda x, y: x * y, channel_sizes, 1) if has_channels else 0

    # Insert channel dimension at appropriate position
    # We want to preserve the relative position of channels in the axis order
    channel_inserted = False

    # Find where the first channel dimension was in the original order
    first_channel_pos = channel_indices[0] if channel_indices else -1

    for orig_idx, ngff_name, size in other_dims:
        # Insert channel at its original relative position
        if not channel_inserted and has_channels:
            # Insert channel if we've passed its original position
            if first_channel_pos >= 0 and orig_idx > first_channel_pos:
                ngff_dims_list.append("c")
                ngff_shape_list.append(total_channels)
                channel_inserted = True

        ngff_dims_list.append(ngff_name)
        ngff_shape_list.append(size)

    # If we haven't inserted channels yet and we have them, append at end
    if not channel_inserted and has_channels:
        ngff_dims_list.append("c")
        ngff_shape_list.append(total_channels)

    return (
        tuple(ngff_dims_list),
        tuple(ngff_shape_list),
        channel_indices,
        dropped_indices,
    )


def _reshape_tiff_for_channels(
    data: da.Array,
    tiff_axes: str,
    tiff_shape: tuple[int, ...],
    ngff_dims: tuple[str, ...],
    ngff_shape: tuple[int, ...],
    channel_indices: list[int],
    dropped_indices: list[int],
) -> da.Array:
    """
    Reshape TIFF data array to match NGFF dimensions.

    Handles:
    - Dropping unsupported dimensions (takes first slice)
    - Flattening multiple channel-like dimensions (C, S) into single 'c'
    - Moving/reordering dimensions as needed

    Parameters
    ----------
    data : da.Array
        Input dask array with original TIFF shape.
    tiff_axes : str
        Original TIFF axis string.
    tiff_shape : Tuple[int, ...]
        Original shape.
    ngff_dims : Tuple[str, ...]
        Target NGFF dimension names.
    ngff_shape : Tuple[int, ...]
        Target shape.
    channel_indices : List[int]
        Indices of channel dimensions in original data.
    dropped_indices : List[int]
        Indices of dimensions to drop.

    Returns
    -------
    da.Array
        Reshaped array matching ngff_shape.
    """
    # If shapes already match, nothing to do
    if data.shape == ngff_shape:
        return data

    # Step 1: Handle dropped axes by selecting first slice
    if dropped_indices:
        slices: list[int | slice] = [slice(None)] * len(tiff_shape)
        for idx in sorted(dropped_indices, reverse=True):
            slices[idx] = 0  # Take first slice of dropped dimension
        data = data[tuple(slices)]

        # Update tracking: remove dropped indices from channel_indices
        # and adjust remaining indices
        new_channel_indices = []
        for ci in channel_indices:
            offset = sum(1 for di in dropped_indices if di < ci)
            new_channel_indices.append(ci - offset)
        channel_indices = new_channel_indices

    # Step 2: If only one or zero channel dimensions, simple reshape suffices
    if len(channel_indices) <= 1:
        if data.shape != ngff_shape:
            return data.reshape(ngff_shape)
        return data

    # Step 3: Multiple channel dimensions - need to flatten them
    # Check if channel dimensions are contiguous
    are_contiguous = all(
        channel_indices[i] + 1 == channel_indices[i + 1]
        for i in range(len(channel_indices) - 1)
    )

    if are_contiguous:
        # Dimensions are contiguous, safe to use reshape directly
        return data.reshape(ngff_shape)

    # Non-contiguous channel dimensions: need to transpose first
    # Move all channel dimensions to be adjacent at the first channel position
    first_channel_pos = channel_indices[0]
    non_channel_indices = [i for i in range(data.ndim) if i not in channel_indices]

    # Build new axis order: dims before first channel, all channels, dims after
    new_axis_order = (
        non_channel_indices[:first_channel_pos]
        + channel_indices
        + non_channel_indices[first_channel_pos:]
    )

    # Transpose to new order
    data = data.transpose(new_axis_order)

    # Now reshape to flatten the channel dimensions
    return data.reshape(ngff_shape)


def _extract_ome_pixel_metadata(
    tif: "tifffile.TiffFile",
    series_index: int = 0,
) -> tuple[dict[str, float] | None, dict[str, str] | None]:
    """
    Extract scale and units from OME-XML metadata for a specific series.

    Parameters
    ----------
    tif : tifffile.TiffFile
        An open TiffFile instance.
    series_index : int, optional
        Index of the series to extract metadata for. Default is 0.

    Returns
    -------
    Tuple[Optional[Dict[str, float]], Optional[Dict[str, str]]]
        A tuple of (scale_dict, units_dict) where:
        - scale_dict maps dimension names ('x', 'y', 'z') to physical sizes
        - units_dict maps dimension names to NGFF-compatible unit strings
        Returns (None, None) if OME metadata is not available or cannot be parsed.

    Examples
    --------
    >>> with tifffile.TiffFile("sample.ome.tiff") as tif:
    ...     scale, units = _extract_ome_pixel_metadata(tif, series_index=0)
    ...     if scale:
    ...         print(f"Scale: {scale}")  # {'x': 0.5, 'y': 0.5, 'z': 2.0}
    ...         print(f"Units: {units}")  # {'x': 'micrometer', ...}
    """
    if not tif.is_ome or not tif.ome_metadata:
        return None, None

    try:
        root = ET.fromstring(tif.ome_metadata)
    except ET.ParseError:
        return None, None

    # OME namespace - try common versions
    namespaces = [
        {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"},
        {"ome": "http://www.openmicroscopy.org/Schemas/OME/2015-01"},
        {"ome": "http://www.openmicroscopy.org/Schemas/OME/2013-06"},
    ]

    pixels = None
    for ns in namespaces:
        # Find the Image element for the requested series
        images = root.findall(".//ome:Image", ns)
        if series_index < len(images):
            image = images[series_index]
            pixels = image.find("ome:Pixels", ns)
            if pixels is not None:
                break

    # Fallback: try without namespace (for non-standard OME-XML)
    if pixels is None:
        count = 0
        for elem in root.iter():
            if "Pixels" in elem.tag:
                if count == series_index:
                    pixels = elem
                    break
                count += 1

    if pixels is None:
        return None, None

    scale: dict[str, float] = {}
    units: dict[str, str] = {}

    # Extract physical sizes for each dimension
    dim_mapping = {
        "PhysicalSizeX": "x",
        "PhysicalSizeY": "y",
        "PhysicalSizeZ": "z",
    }
    unit_mapping = {
        "PhysicalSizeXUnit": "x",
        "PhysicalSizeYUnit": "y",
        "PhysicalSizeZUnit": "z",
    }

    for ome_attr, dim in dim_mapping.items():
        value = pixels.get(ome_attr)
        if value is not None:
            try:
                scale[dim] = float(value)
            except (ValueError, TypeError):
                warnings.warn(
                    f"Invalid physical size value '{value}' for dimension '{dim}' "
                    "in OME-XML metadata; ignoring this value.",
                    RuntimeWarning,
                )

    for ome_attr, dim in unit_mapping.items():
        value = pixels.get(ome_attr)
        if value is not None:
            ngff_unit = _normalize_unit(value)
            if ngff_unit is not None:
                units[dim] = ngff_unit

    if not scale:
        return None, None

    return scale, units if units else None


def _extract_ome_channel_names(
    tif: "tifffile.TiffFile",
    series_index: int = 0,
) -> list[str] | None:
    """
    Extract channel names from OME-XML metadata for a specific series.

    Parameters
    ----------
    tif : tifffile.TiffFile
        An open TiffFile instance.
    series_index : int, optional
        Index of the series to extract metadata for. Default is 0.

    Returns
    -------
    Optional[List[str]]
        List of channel names, or None if OME metadata is not available or
        cannot be parsed. Empty strings are used for channels without names.

    Examples
    --------
    >>> with tifffile.TiffFile("sample.ome.tiff") as tif:
    ...     channel_names = _extract_ome_channel_names(tif, series_index=0)
    ...     if channel_names:
    ...         print(f"Channels: {channel_names}")  # ['DAPI', 'GFP', 'RFP']
    """
    if not tif.is_ome or not tif.ome_metadata:
        return None

    try:
        root = ET.fromstring(tif.ome_metadata)
    except ET.ParseError:
        return None

    # OME namespace - try common versions
    namespaces = [
        {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"},
        {"ome": "http://www.openmicroscopy.org/Schemas/OME/2015-01"},
        {"ome": "http://www.openmicroscopy.org/Schemas/OME/2013-06"},
    ]

    pixels = None
    for ns in namespaces:
        # Find the Image element for the requested series
        images = root.findall(".//ome:Image", ns)
        if series_index < len(images):
            image = images[series_index]
            pixels = image.find("ome:Pixels", ns)
            if pixels is not None:
                # Extract channel elements
                channels = pixels.findall("ome:Channel", ns)
                if channels:
                    # Extract Name attribute from each channel, use empty
                    # string if not present
                    channel_names = [ch.get("Name", "") for ch in channels]
                    return channel_names if channel_names else None

    # Fallback: try without namespace (for non-standard OME-XML)
    if pixels is None:
        count = 0
        for elem in root.iter():
            if "Pixels" in elem.tag:
                if count == series_index:
                    pixels = elem
                    break
                count += 1

    if pixels is not None:
        # Look for Channel elements
        channels = [ch for ch in pixels if "Channel" in ch.tag]
        if channels:
            channel_names = [ch.get("Name", "") for ch in channels]
            return channel_names if channel_names else None

    return None


def _sanitize_series_name(name: str) -> str:
    """
    Sanitize a series name to prevent path traversal attacks.

    Removes path separators and parent directory references to ensure
    the name is safe to use in filesystem paths.

    Parameters
    ----------
    name : str
        Raw series name from TIFF metadata.

    Returns
    -------
    str
        Sanitized series name safe for use in file paths.
    """
    # Remove path separators
    name = name.replace("/", "_").replace("\\", "_")
    # Replace all occurrences of '..' with '_' in a loop to handle '....' etc.
    while ".." in name:
        name = name.replace("..", "_")
    # Remove any leading/trailing whitespace or dots (avoid hidden files)
    name = name.strip().lstrip(".")
    # If the name is empty after sanitization, use a default
    return name if name else "unnamed"


_SPATIAL_DIMS = {"x", "y", "z"}


def _build_multiscales_from_pyramid(
    tif: "tifffile.TiffFile",
    series_idx: int,
    tiff_series: "tifffile.TiffPageSeries",
    ome_scale: dict[str, float] | None,
    ome_units: dict[str, str] | None,
    ome_channel_names: list[str] | None,
) -> "Multiscales":
    """Build a Multiscales object from a pyramidal TIFF's existing pyramid levels.

    Instead of regenerating the pyramid via downsampling, this reuses the
    resolution levels already stored in the TIFF file.

    Parameters
    ----------
    tif : tifffile.TiffFile
        An open TiffFile instance.
    series_idx : int
        Index of the series in the TIFF file.
    tiff_series : tifffile.TiffPageSeries
        The TIFF series object (used for axis/shape info).
    ome_scale : dict or None
        Physical pixel sizes from OME metadata (e.g., {'x': 0.5, 'y': 0.5}).
    ome_units : dict or None
        Unit strings from OME metadata (e.g., {'x': 'micrometer'}).
    ome_channel_names : list or None
        Channel names from OME metadata (e.g., ['DAPI', 'GFP', 'RFP']).

    Returns
    -------
    Multiscales
        A Multiscales object containing all pyramid levels.
    """
    from .multiscales import Multiscales
    from .v04.zarr_metadata import Axis, Dataset, Metadata, Scale, Translation

    try:
        from zarr.core import Array as ZarrArray
    except ImportError:
        from zarr.core.array import Array as ZarrArray

    # Open the full zarr store (all levels) for this series
    store = tif.aszarr(series=series_idx)
    root = zarr.open(store, mode="r")

    # Get TIFF axis mapping
    tiff_axes = tiff_series.axes if tiff_series.axes else None

    # Get ordered dataset paths from multiscales metadata
    paths: list[str] = []
    if isinstance(root, zarr.Group) and "multiscales" in root.attrs:
        multiscales_meta = root.attrs["multiscales"]
        if multiscales_meta:
            datasets = multiscales_meta[0].get("datasets", [])
            paths = [d["path"] for d in datasets if d["path"] in root]

    if not paths and isinstance(root, zarr.Group):
        # Fallback: sort arrays by size (largest first = full resolution)
        items = [(k, root[k]) for k in root.keys() if isinstance(root[k], ZarrArray)]
        items.sort(key=lambda x: x[1].size, reverse=True)
        paths = [k for k, _ in items]

    if not paths:
        # Single array (not a Group), shouldn't happen for pyramidal TIFFs
        msg = "Expected pyramidal TIFF but got a single array"
        raise ValueError(msg)

    # Build the base level NgffImage (level 0)
    level0_data = root[paths[0]]

    # Map TIFF axes for level 0
    dims: tuple[str, ...] | None = None
    ngff_shape: tuple[int, ...] | None = None
    channel_indices: list[int] = []
    dropped_indices: list[int] = []

    if tiff_axes:
        dims, ngff_shape, channel_indices, dropped_indices = _map_tiff_axes_to_ngff(
            tiff_axes, tiff_series.shape
        )

    # Build scale dict from OME metadata
    scale = None
    if ome_scale:
        scale = {}
        spatial_dims_for_scale = dims if dims else ("z", "y", "x")
        for dim in spatial_dims_for_scale:
            if dim in ome_scale:
                scale[dim] = ome_scale[dim]
            elif dim in ("x", "y", "z"):
                scale[dim] = 1.0

    # Build units dict from OME metadata
    axes_units = None
    if ome_units:
        axes_units = {}
        spatial_dims_for_units = dims if dims else ("z", "y", "x")
        for dim in spatial_dims_for_units:
            if dim in ome_units:
                axes_units[dim] = ome_units[dim]

    # Create base NgffImage
    ngff_image_0 = to_ngff_image(
        level0_data,
        dims=dims,
        scale=scale,
        axes_units=axes_units,
        channel_names=ome_channel_names,
    )

    # Reshape if needed (channel flattening, dropped axes)
    if (
        tiff_axes
        and dims is not None
        and ngff_shape is not None
        and ngff_image_0.data.shape != ngff_shape
    ):
        reshaped_data = _reshape_tiff_for_channels(
            ngff_image_0.data,
            tiff_axes,
            tiff_series.shape,
            dims,
            ngff_shape,
            channel_indices,
            dropped_indices,
        )
        ngff_image_0 = NgffImage(
            data=reshaped_data,
            dims=dims,
            scale=ngff_image_0.scale,
            translation=ngff_image_0.translation,
            name=ngff_image_0.name,
            axes_units=ngff_image_0.axes_units,
            channel_names=ngff_image_0.channel_names,
        )

    level0_shape = ngff_image_0.data.shape
    base_scale = ngff_image_0.scale
    base_translation = ngff_image_0.translation

    # Build NgffImages for all levels
    images = [ngff_image_0]
    for path in paths[1:]:
        arr = root[path]
        level_scale: dict[str, float] = {}
        level_translation: dict[str, float] = {}

        for dim in ngff_image_0.dims:
            dim_idx = list(ngff_image_0.dims).index(dim)
            if dim in _SPATIAL_DIMS:
                factor = level0_shape[dim_idx] / arr.shape[dim_idx]
                level_scale[dim] = base_scale[dim] * factor
                level_translation[dim] = (
                    base_translation[dim] + 0.5 * (factor - 1) * base_scale[dim]
                )
            else:
                if dim in base_scale:
                    level_scale[dim] = base_scale[dim]
                if dim in base_translation:
                    level_translation[dim] = base_translation[dim]

        # Convert to dask array first
        level_data = da.from_zarr(arr)

        # Apply the same axis mapping and channel reshaping that was applied to level 0
        # Pyramid levels have the same axis structure as the original TIFF,
        # just with downsampled spatial dimensions
        if tiff_axes and dims is not None and ngff_shape is not None:
            # For this pyramid level, calculate what shape it should have after reshaping
            # by substituting the actual spatial dimensions from this level
            level_expected_shape = list(ngff_shape)

            # Map TIFF axes to find where spatial dimensions are in the ORIGINAL shape
            tiff_axes_lower = tiff_axes.lower()
            for tiff_idx, tiff_ax in enumerate(tiff_axes_lower):
                ngff_ax = TIFF_AXIS_TO_NGFF.get(tiff_ax)
                if ngff_ax and ngff_ax in _SPATIAL_DIMS and ngff_ax in dims:
                    # This is a spatial dimension - use the downsampled size
                    ngff_idx = list(dims).index(ngff_ax)
                    level_expected_shape[ngff_idx] = arr.shape[tiff_idx]

            # Only reshape if shapes don't already match
            if level_data.shape != tuple(level_expected_shape):
                level_data = _reshape_tiff_for_channels(
                    level_data,
                    tiff_axes,
                    arr.shape,  # This level's TIFF shape
                    dims,
                    tuple(level_expected_shape),
                    channel_indices,
                    dropped_indices,
                )

        level_image = NgffImage(
            data=level_data,
            dims=ngff_image_0.dims,
            scale=level_scale,
            translation=level_translation,
            name=ngff_image_0.name,
            axes_units=ngff_image_0.axes_units,
            channel_names=ngff_image_0.channel_names,
        )
        images.append(level_image)

    # Build Metadata (axes, datasets, coordinate transforms)
    axes = []
    for dim in ngff_image_0.dims:
        unit = None
        if ngff_image_0.axes_units and dim in ngff_image_0.axes_units:
            unit = ngff_image_0.axes_units[dim]

        orientation = None
        if ngff_image_0.axes_orientations and dim in ngff_image_0.axes_orientations:
            orientation = ngff_image_0.axes_orientations[dim]

        if dim in {"x", "y", "z"}:
            axes.append(
                Axis(name=dim, type="space", unit=unit, orientation=orientation)
            )
        elif dim == "c":
            axes.append(Axis(name=dim, type="channel", unit=unit))
        elif dim == "t":
            axes.append(Axis(name=dim, type="time", unit=unit))

    datasets = []
    for index, image in enumerate(images):
        path = f"scale{index}/{ngff_image_0.name}"
        scale_values = [image.scale.get(d, 1.0) for d in image.dims]
        trans_values = [image.translation.get(d, 0.0) for d in image.dims]
        coord_transforms = [Scale(scale_values), Translation(trans_values)]
        datasets.append(Dataset(path=path, coordinateTransformations=coord_transforms))

    metadata = Metadata(
        axes=axes,
        datasets=datasets,
        name=ngff_image_0.name,
        coordinateTransformations=None,
    )

    return Multiscales(images, metadata)


def tiff_file_to_ngff_images(
    tiff_path: str | Path,
    series: int | str | list[int | str] | None = None,
    reuse_existing_pyramids: bool = False,
) -> "list[tuple[str, NgffImage | Multiscales]]":
    """
    Convert a TIFF file to a list of (name, NgffImage) or (name, Multiscales) pairs.

    This function properly handles multi-series TIFF files (including OME-TIFF)
    and extracts physical size metadata when available.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF file.
    series : int, str, or list, optional
        Series to convert:
        - None or "all": Convert all series
        - int: Convert series at that index
        - str: Convert series matching regex pattern (use '*' as wildcard)
        - list: Convert multiple series by index or pattern
    reuse_existing_pyramids : bool, optional
        If True and a series has multiple pyramid levels, return a
        ``Multiscales`` object built from the existing levels instead of a
        single ``NgffImage``. Default is False (always return NgffImage).

    Returns
    -------
    List[Tuple[str, Union[NgffImage, Multiscales]]]
        List of (series_name, result) tuples. When ``reuse_existing_pyramids``
        is True and the series is pyramidal, result is a ``Multiscales``.
        Otherwise it is an ``NgffImage`` with:
        - data: Dask array with lazy loading from the TIFF
        - dims: Dimension labels (e.g., ('z', 'y', 'x'))
        - scale: Physical pixel sizes (from OME metadata if available)
        - translation: Origin offset (defaults to 0.0)
        - axes_units: Unit strings (from OME metadata if available)

    Raises
    ------
    ImportError
        If tifffile is not installed.
    IndexError
        If a specified series index is out of bounds.
    ValueError
        If the series specification is invalid.

    Examples
    --------
    >>> from ngff_zarr import tiff_file_to_ngff_images
    >>> # Convert all series
    >>> images = tiff_file_to_ngff_images("sample.ome.tiff")
    >>> # Convert specific series by index
    >>> images = tiff_file_to_ngff_images("sample.ome.tiff", series=0)
    >>> # Convert series matching pattern
    >>> images = tiff_file_to_ngff_images("sample.ome.tiff", series="*area_1*")
    >>> # Reuse existing pyramid levels
    >>> images = tiff_file_to_ngff_images(
    ...     "pyramidal.ome.tiff", reuse_existing_pyramids=True
    ... )
    """
    try:
        import tifffile
    except ImportError as e:
        msg = "tifffile package is required. Install with: pip install tifffile"
        raise ImportError(msg) from e

    results: list = []

    with tifffile.TiffFile(tiff_path) as tif:
        all_series = tif.series

        # Determine which series to convert
        if series is None or series == "all":
            indices_to_convert = list(range(len(all_series)))
        elif isinstance(series, int):
            # Explicit index - validate it exists
            if series < 0 or series >= len(all_series):
                msg = (
                    f"Series index {series} is out of bounds. "
                    f"File has {len(all_series)} series "
                    f"(indices 0-{len(all_series) - 1})."
                )
                raise IndexError(msg)
            indices_to_convert = [series]
        elif isinstance(series, str):
            # Treat as regex pattern - match on series name
            pattern = re.compile(series.replace("*", ".*"), re.IGNORECASE)
            indices_to_convert = [
                i
                for i, series_item in enumerate(all_series)
                if series_item.name and pattern.search(series_item.name)
            ]
            # If no names match and there's only one series, include it
            if not indices_to_convert and len(all_series) == 1:
                indices_to_convert = [0]
        elif isinstance(series, list):
            indices_to_convert = []
            for s in series:
                if isinstance(s, int):
                    # Explicit index - validate it exists
                    if s < 0 or s >= len(all_series):
                        msg = (
                            f"Series index {s} is out of bounds. "
                            f"File has {len(all_series)} series "
                            f"(indices 0-{len(all_series) - 1})."
                        )
                        raise IndexError(msg)
                    indices_to_convert.append(s)
                elif isinstance(s, str):
                    pattern = re.compile(s.replace("*", ".*"), re.IGNORECASE)
                    indices_to_convert.extend(
                        i
                        for i, series_item in enumerate(all_series)
                        if series_item.name and pattern.search(series_item.name)
                    )
        else:
            msg = f"Invalid series specification: {series}"
            raise ValueError(msg)

        # Deduplicate indices while preserving order
        indices_to_convert = list(dict.fromkeys(indices_to_convert))

        # Convert selected series
        for idx in indices_to_convert:
            tiff_series = all_series[idx]

            # Extract OME metadata for this series
            ome_scale, ome_units = _extract_ome_pixel_metadata(tif, series_index=idx)
            ome_channel_names = _extract_ome_channel_names(tif, series_index=idx)

            # Generate series name early (used for both paths)
            raw_name = tiff_series.name if tiff_series.name else f"series_{idx}"
            series_name = _sanitize_series_name(raw_name)

            # Check if this series has existing pyramid levels we can reuse
            n_levels = len(tiff_series.levels) if hasattr(tiff_series, "levels") else 1
            if reuse_existing_pyramids and n_levels > 1:
                multiscales = _build_multiscales_from_pyramid(
                    tif, idx, tiff_series, ome_scale, ome_units, ome_channel_names
                )
                results.append((series_name, multiscales))
                continue

            # Get the zarr store for this series, using level=0 for base resolution
            # This ensures we always get an Array, not a Group, for pyramidal TIFFs
            store = tif.aszarr(series=idx, level=0)
            root = zarr.open(store, mode="r")

            # Get dims from the series axes with proper mapping
            # TIFF axes like 'S' (sample/RGB) need to be mapped to NGFF 'c' (channel)
            tiff_axes = tiff_series.axes if tiff_series.axes else None
            dims: tuple[str, ...] | None = None
            ngff_shape: tuple[int, ...] | None = None
            channel_indices: list[int] = []
            dropped_indices: list[int] = []

            if tiff_axes:
                # Map TIFF axes to NGFF dims (handles S->c, drops unsupported axes)
                dims, ngff_shape, channel_indices, dropped_indices = (
                    _map_tiff_axes_to_ngff(tiff_axes, tiff_series.shape)
                )

            # Build scale dict from OME metadata, using defaults for missing dims
            scale = None
            if ome_scale:
                scale = {}
                spatial_dims = dims if dims else ("z", "y", "x")
                for dim in spatial_dims:
                    if dim in ome_scale:
                        scale[dim] = ome_scale[dim]
                    elif dim in ("x", "y", "z"):
                        scale[dim] = 1.0  # Default scale for spatial dims

            # Build units dict from OME metadata
            axes_units = None
            if ome_units:
                axes_units = {}
                spatial_dims = dims if dims else ("z", "y", "x")
                for dim in spatial_dims:
                    if dim in ome_units:
                        axes_units[dim] = ome_units[dim]

            # Convert to NgffImage
            ngff_image = to_ngff_image(
                root,
                dims=dims,
                scale=scale,
                axes_units=axes_units,
                channel_names=ome_channel_names,
            )

            # Reshape data if needed (for channel flattening, dropped axes, etc.)
            if (
                tiff_axes
                and dims is not None
                and ngff_shape is not None
                and ngff_image.data.shape != ngff_shape
            ):
                reshaped_data = _reshape_tiff_for_channels(
                    ngff_image.data,
                    tiff_axes,
                    tiff_series.shape,
                    dims,
                    ngff_shape,
                    channel_indices,
                    dropped_indices,
                )
                ngff_image = NgffImage(
                    data=reshaped_data,
                    dims=dims,
                    scale=ngff_image.scale,
                    translation=ngff_image.translation,
                    name=ngff_image.name,
                    axes_units=ngff_image.axes_units,
                    channel_names=ngff_image.channel_names,
                )

            results.append((series_name, ngff_image))

    return results
