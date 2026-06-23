# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Convert Leica Image Format (LIF) files to OME-NGFF.

This module provides functions to convert LIF files (and related formats
like LOF, XLIF, XLEF, XLCF) to NgffImage objects for further processing
into OME-Zarr format.

Requires the `liffile` package: pip install liffile
"""

import os
import re
from collections.abc import Sequence
from functools import reduce
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np

from .ngff_image import NgffImage
from .v04.zarr_metadata import (
    Plate,
    PlateColumn,
    PlateRow,
    PlateWell,
)

# Mapping from LIF dimension names to OME-NGFF dimension names
# LIF uses uppercase, OME-NGFF uses lowercase for spatial dims
LIF_DIM_TO_NGFF: dict[str, str | None] = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "T": "t",
    "C": "c",
    "λ": "c",  # Emission wavelength -> channel
    "Λ": "c",  # Excitation wavelength -> channel
    "S": "c",  # RGB samples -> channel
    "N": None,  # XT slices - special handling
    "Q": None,  # T slices - special handling
    "A": None,  # Rotation - special handling
    "L": None,  # Loop - special handling
    "M": None,  # Mosaic - special handling for HCS
    "H": None,  # TCSPC histogram - not supported
}

# Spatial dimensions in OME-NGFF
SPATIAL_DIMS = {"x", "y", "z"}


def _get_lif_image_data_delayed(lif_image: Any) -> np.ndarray:
    """Load LIF image data (for use with dask.delayed)."""
    return lif_image.asarray()


def _read_lif_image_array(lif_path: str, index: int) -> np.ndarray:
    """Re-open the LIF file and read a single image's array.

    Re-opening per read keeps the dask graph self-contained: the ``LifFile``
    handle that produced the graph is typically closed (and the graph may be
    serialized to other worker processes) by the time chunks are computed, so
    holding a reference to a ``LifImage`` from that handle raises
    ``ValueError: seek of closed file``.
    """
    from liffile import LifFile

    with LifFile(lif_path) as lif:
        return lif.images[index].asarray()


def _lif_reopen_locator(lif_image: Any) -> tuple[str, int] | None:
    """Derive a ``(filepath, index)`` that re-opens the source LIF and
    reselects ``lif_image``.

    Returns ``None`` when the source file path cannot be determined (e.g. the
    image is a test mock or was created without a backing file), in which case
    the caller falls back to referencing the ``LifImage`` directly.
    """
    parent = getattr(lif_image, "parent", None)
    filepath = getattr(parent, "filepath", None)
    if not isinstance(filepath, (str, os.PathLike)):
        return None
    try:
        images = list(parent.images)
    except TypeError:
        return None
    uuid = getattr(lif_image, "uuid", None)
    for i, img in enumerate(images):
        if img is lif_image:
            return (os.fspath(filepath), i)
    if uuid is not None:
        for i, img in enumerate(images):
            if getattr(img, "uuid", None) == uuid:
                return (os.fspath(filepath), i)
    return None


def _delayed_lif_data(lif_image: Any, shape: Sequence[int], dtype: Any) -> da.Array:
    """Build a lazy dask array for ``lif_image`` that survives the source
    ``LifFile`` being closed by re-opening the file at compute time when
    possible.
    """
    locator = _lif_reopen_locator(lif_image)
    if locator is not None:
        lif_path, index = locator
        delayed_data = dask.delayed(_read_lif_image_array)(lif_path, index)
    else:
        delayed_data = dask.delayed(_get_lif_image_data_delayed)(lif_image)
    return da.from_delayed(delayed_data, shape=tuple(shape), dtype=dtype)


def _extract_scale_translation(
    lif_image: Any, ngff_dims: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Extract scale (pixel spacing) and translation (origin) from LIF image coordinates.

    Parameters
    ----------
    lif_image : LifImage
        The LIF image object with coords attribute.
    ngff_dims : Sequence[str]
        The mapped NGFF dimension names.

    Returns
    -------
    scale : Dict[str, float]
        Pixel spacing for spatial dimensions.
    translation : Dict[str, float]
        Origin offset for spatial dimensions.
    """
    scale: dict[str, float] = {}
    translation: dict[str, float] = {}

    coords = getattr(lif_image, "coords", {})

    for lif_dim, ngff_dim in LIF_DIM_TO_NGFF.items():
        if ngff_dim is None or ngff_dim not in SPATIAL_DIMS:
            continue
        if ngff_dim not in ngff_dims:
            continue

        if lif_dim in coords:
            coord_array = coords[lif_dim]
            if len(coord_array) > 1:
                # Scale is the spacing between coordinates
                spacing = abs(coord_array[1] - coord_array[0])
                scale[ngff_dim] = float(spacing)
            else:
                scale[ngff_dim] = 1.0
            # Translation is the origin (first coordinate)
            translation[ngff_dim] = float(coord_array[0])
        else:
            # Default values if coordinates not available
            if ngff_dim not in scale:
                scale[ngff_dim] = 1.0
            if ngff_dim not in translation:
                translation[ngff_dim] = 0.0

    return scale, translation


def _map_lif_dims_to_ngff(
    lif_dims: Sequence[str], lif_shape: Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """
    Map LIF dimensions to OME-NGFF dimensions, flattening channel-like dims.

    Parameters
    ----------
    lif_dims : Sequence[str]
        Original LIF dimension names (e.g., ('T', 'Z', 'C', 'Y', 'X')).
    lif_shape : Sequence[int]
        Original shape corresponding to lif_dims.

    Returns
    -------
    ngff_dims : Tuple[str, ...]
        Mapped NGFF dimension names.
    ngff_shape : Tuple[int, ...]
        Shape after flattening channel-like dimensions.
    """
    # Track which indices map to channels (to be flattened)
    channel_like_dims = {"C", "λ", "Λ", "S"}

    # First pass: identify channel indices and their sizes
    channel_indices: list[int] = []
    channel_sizes: list[int] = []
    other_dims: list[tuple[int, str, int]] = []  # (original_idx, ngff_name, size)

    for i, (dim, size) in enumerate(zip(lif_dims, lif_shape)):
        ngff_dim = LIF_DIM_TO_NGFF.get(dim)

        if dim in channel_like_dims:
            channel_indices.append(i)
            channel_sizes.append(size)
        elif ngff_dim is not None:
            other_dims.append((i, ngff_dim, size))
        elif dim == "M":
            # Mosaic dimension - keep for now, will be handled specially
            other_dims.append((i, "m", size))
        else:
            # Unknown dimension - use lowercase version
            other_dims.append((i, dim.lower(), size))

    # Build output dimensions and shape
    ngff_dims_list: list[str] = []
    ngff_shape_list: list[int] = []

    # Sort other_dims by original index to maintain order
    other_dims.sort(key=lambda x: x[0])

    # Calculate total channel size
    total_channels = reduce(lambda x, y: x * y, channel_sizes, 1)

    # Find where to insert the channel dimension
    # Convention: channels typically come after spatial dims in OME-NGFF
    # Order is typically: t, c, z, y, x or t, z, c, y, x
    channel_inserted = False

    for orig_idx, ngff_name, size in other_dims:
        # Insert channel before spatial dims if we have channels
        if not channel_inserted and total_channels > 1 and ngff_name in SPATIAL_DIMS:
            ngff_dims_list.append("c")
            ngff_shape_list.append(total_channels)
            channel_inserted = True

        ngff_dims_list.append(ngff_name)
        ngff_shape_list.append(size)

    # If we haven't inserted channels yet and we have them, append at end
    if not channel_inserted and total_channels > 1:
        ngff_dims_list.append("c")
        ngff_shape_list.append(total_channels)

    return tuple(ngff_dims_list), tuple(ngff_shape_list)


def _reshape_for_flattened_channels(
    data: da.Array,
    lif_dims: Sequence[str],
    lif_shape: Sequence[int],
    ngff_dims: Sequence[str],
    ngff_shape: Sequence[int],
) -> da.Array:
    """
    Reshape data array to flatten channel-like dimensions.

    This handles the case where LIF has multiple channel-like dimensions
    (C, λ, Λ, S) that need to be flattened into a single channel dimension.

    The function ensures channel dimensions are moved to be adjacent before
    flattening, regardless of their original positions in the array.
    """
    channel_like_dims = {"C", "λ", "Λ", "S"}

    # Count channel-like dimensions in the original
    channel_dim_indices = [i for i, d in enumerate(lif_dims) if d in channel_like_dims]

    if len(channel_dim_indices) <= 1:
        # No flattening needed, just return reshaped if dims changed
        if data.shape != ngff_shape:
            return data.reshape(ngff_shape)
        return data

    # Need to flatten multiple channel dimensions
    # Strategy:
    # 1. Move all channel dimensions to be contiguous
    # 2. Reshape to flatten them into a single dimension
    # 3. Move the flattened channel dimension to its target position

    # Check if channel dimensions are already contiguous
    are_contiguous = all(
        channel_dim_indices[i] + 1 == channel_dim_indices[i + 1]
        for i in range(len(channel_dim_indices) - 1)
    )

    if are_contiguous:
        # Dimensions are contiguous, safe to use reshape directly
        return data.reshape(ngff_shape)

    # Move channel dimensions to be adjacent
    # Move them to the position of the first channel dimension
    first_channel_pos = channel_dim_indices[0]

    # Calculate new axis order: non-channel dims + channel dims
    non_channel_indices = [
        i for i in range(len(lif_dims)) if i not in channel_dim_indices
    ]

    # Insert channel dims at the position where the first one was
    new_axis_order = (
        non_channel_indices[:first_channel_pos]
        + channel_dim_indices
        + non_channel_indices[first_channel_pos:]
    )

    # Transpose to new order
    data = data.transpose(new_axis_order)

    # Now reshape to flatten the channel dimensions
    # Calculate the intermediate shape after transposing
    intermediate_shape = [lif_shape[i] for i in new_axis_order]

    # Calculate flattened shape: replace the channel dimensions with their product
    flattened_shape = (
        intermediate_shape[:first_channel_pos]
        + [
            reduce(
                lambda x, y: x * y,
                [
                    intermediate_shape[i]
                    for i in range(
                        first_channel_pos, first_channel_pos + len(channel_dim_indices)
                    )
                ],
            )
        ]
        + intermediate_shape[first_channel_pos + len(channel_dim_indices) :]
    )

    data = data.reshape(flattened_shape)

    # Now the data has the right shape, but might not match ngff_shape dimension order
    # Find where 'c' is in ngff_dims
    if "c" in ngff_dims:
        target_c_pos = list(ngff_dims).index("c")
        if target_c_pos != first_channel_pos:
            # Move channel dimension to target position
            current_order = list(range(len(flattened_shape)))
            current_order.pop(first_channel_pos)
            current_order.insert(target_c_pos, first_channel_pos)
            data = data.transpose(current_order)

    return data


def lif_to_ngff_image(
    lif_image: Any,
    name: str | None = None,
) -> NgffImage:
    """
    Convert a single LifImage to an NgffImage.

    Parameters
    ----------
    lif_image : LifImage
        A LifImage object from the liffile library.
    name : str, optional
        Name for the output image. If None, uses lif_image.name.

    Returns
    -------
    NgffImage
        The converted image with metadata.

    Examples
    --------
    >>> from liffile import LifFile
    >>> from ngff_zarr import lif_to_ngff_image
    >>> with LifFile("sample.lif") as lif:
    ...     ngff_image = lif_to_ngff_image(lif.images[0])
    """
    # Get basic info from LIF image
    lif_dims = lif_image.dims
    lif_shape = lif_image.shape
    dtype = lif_image.dtype

    # Map dimensions
    ngff_dims, ngff_shape = _map_lif_dims_to_ngff(lif_dims, lif_shape)

    # Extract scale and translation from coordinates
    scale, translation = _extract_scale_translation(lif_image, ngff_dims)

    # Ensure all spatial dims have scale/translation
    for dim in ngff_dims:
        if dim in SPATIAL_DIMS:
            if dim not in scale:
                scale[dim] = 1.0
            if dim not in translation:
                translation[dim] = 0.0

    # Create lazy dask array from the LIF image. Re-open the source file at
    # compute time so the array remains valid after the producing LifFile
    # handle is closed.
    data = _delayed_lif_data(lif_image, lif_shape, dtype)

    # Reshape if needed for flattened channels
    if data.shape != ngff_shape:
        data = _reshape_for_flattened_channels(
            data, lif_dims, lif_shape, ngff_dims, ngff_shape
        )

    # Get image name
    if name is None:
        name = getattr(lif_image, "name", "image")

    return NgffImage(
        data=data,
        dims=ngff_dims,
        scale=scale,
        translation=translation,
        name=name,
    )


def lif_file_to_ngff_images(
    lif_path: str | Path,
    series: int | str | list[int | str] | None = None,
) -> list[tuple[str, NgffImage]]:
    """
    Convert a LIF file to a list of (name, NgffImage) pairs.

    Parameters
    ----------
    lif_path : str or Path
        Path to the LIF file.
    series : int, str, or list, optional
        Series to convert:
        - None or "all": Convert all series
        - int: Convert series at that index
        - str: Convert series matching regex pattern
        - list: Convert multiple series by index or pattern

    Returns
    -------
    List[Tuple[str, NgffImage]]
        List of (series_name, ngff_image) tuples.

    Examples
    --------
    >>> from ngff_zarr import lif_file_to_ngff_images
    >>> # Convert all series
    >>> images = lif_file_to_ngff_images("sample.lif")
    >>> # Convert specific series by index
    >>> images = lif_file_to_ngff_images("sample.lif", series=0)
    >>> # Convert series matching pattern
    >>> images = lif_file_to_ngff_images("sample.lif", series="*area_1*")
    """
    try:
        from liffile import LifFile
    except ImportError as e:
        msg = "liffile package is required. Install with: pip install liffile"
        raise ImportError(msg) from e

    results: list[tuple[str, NgffImage]] = []

    with LifFile(lif_path) as lif:
        all_images = list(lif.images)

        # Determine which series to convert
        if series is None or series == "all":
            indices_to_convert = list(range(len(all_images)))
        elif isinstance(series, int):
            # Explicit index - validate it exists
            if series < 0 or series >= len(all_images):
                msg = f"Series index {series} is out of bounds. File has {len(all_images)} series (indices 0-{len(all_images) - 1})."
                raise IndexError(msg)
            indices_to_convert = [series]
        elif isinstance(series, str):
            # Treat as regex pattern
            pattern = re.compile(series.replace("*", ".*"), re.IGNORECASE)
            indices_to_convert = [
                i for i, img in enumerate(all_images) if pattern.search(img.name)
            ]
        elif isinstance(series, list):
            indices_to_convert = []
            for s in series:
                if isinstance(s, int):
                    # Explicit index - validate it exists
                    if s < 0 or s >= len(all_images):
                        msg = f"Series index {s} is out of bounds. File has {len(all_images)} series (indices 0-{len(all_images) - 1})."
                        raise IndexError(msg)
                    indices_to_convert.append(s)
                elif isinstance(s, str):
                    pattern = re.compile(s.replace("*", ".*"), re.IGNORECASE)
                    indices_to_convert.extend(
                        i
                        for i, img in enumerate(all_images)
                        if pattern.search(img.name)
                    )
        else:
            msg = f"Invalid series specification: {series}"
            raise ValueError(msg)

        # Convert selected series
        for idx in indices_to_convert:
            lif_image = all_images[idx]
            ngff_image = lif_to_ngff_image(lif_image)
            # Use path as name to preserve hierarchy
            series_name = lif_image.path.replace("/", "_")
            results.append((series_name, ngff_image))

    return results


def has_mosaic_dimension(lif_image: Any) -> bool:
    """
    Check if a LIF image has a mosaic dimension with size > 1.

    Parameters
    ----------
    lif_image : LifImage
        The LIF image to check.

    Returns
    -------
    bool
        True if the image has M dimension with size > 1.
    """
    dims = lif_image.dims
    sizes = lif_image.sizes if hasattr(lif_image, "sizes") else {}

    if "M" in dims:
        idx = dims.index("M") if isinstance(dims, (list, tuple)) else None
        if idx is not None:
            return lif_image.shape[idx] > 1
        if "M" in sizes:
            return sizes["M"] > 1

    return False


def lif_to_hcs_plate(
    lif_image: Any,
    plate_name: str = "plate",
    version: str = "0.4",
) -> tuple[Plate, list[tuple[str, str, int, NgffImage]]]:
    """
    Convert a LIF image with mosaic dimension to HCS plate structure.

    When a LIF image has M > 1 (multiple mosaic positions), this function
    creates an HCS plate structure where each position becomes a field of view.

    Parameters
    ----------
    lif_image : LifImage
        A LIF image with mosaic dimension (M > 1).
    plate_name : str, optional
        Name for the plate. Default is "plate".
    version : str, optional
        OME-Zarr version for the plate. Default is "0.4".

    Returns
    -------
    plate : Plate
        Plate metadata for HCS structure.
    well_images : List[Tuple[str, str, int, NgffImage]]
        List of (row_name, column_name, field_index, ngff_image) tuples.

    Raises
    ------
    ValueError
        If the image does not have a mosaic dimension with M > 1.

    Examples
    --------
    >>> from liffile import LifFile
    >>> from ngff_zarr import lif_to_hcs_plate
    >>> with LifFile("mosaic.lif") as lif:
    ...     plate, well_images = lif_to_hcs_plate(lif.images[0])
    ...     # plate contains HCS metadata
    ...     # well_images contains individual field images
    """
    if not has_mosaic_dimension(lif_image):
        msg = "LIF image does not have a mosaic dimension (M > 1)"
        raise ValueError(msg)

    # Get mosaic size
    dims = lif_image.dims
    shape = lif_image.shape
    m_idx = list(dims).index("M")
    n_positions = shape[m_idx]

    # Create plate structure
    # For simplicity, use a single row with M columns
    # Each mosaic position becomes a separate well
    row_name = "A"
    columns = [PlateColumn(name=str(i + 1)) for i in range(n_positions)]
    rows = [PlateRow(name=row_name)]
    wells = [
        PlateWell(path=f"{row_name}/{i + 1}", rowIndex=0, columnIndex=i)
        for i in range(n_positions)
    ]

    plate = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name=plate_name,
        field_count=1,  # 1 field per well (the mosaic position)
        version=version,
    )

    # Extract individual mosaic positions as separate images
    well_images: list[tuple[str, str, int, NgffImage]] = []

    # Get the full data
    lif_dims = lif_image.dims
    lif_shape = lif_image.shape

    # Create a delayed load for the full data. Re-open the source file at
    # compute time so the array remains valid after the producing LifFile
    # handle is closed.
    full_data = _delayed_lif_data(lif_image, lif_shape, lif_image.dtype)

    # Build slicing for each mosaic position
    for pos_idx in range(n_positions):
        # Create slice that selects this mosaic position
        slices = [slice(None)] * len(lif_dims)
        slices[m_idx] = pos_idx

        # Extract the position data
        pos_data = full_data[tuple(slices)]

        # Remove M dimension from dims and shape
        new_dims = tuple(d for i, d in enumerate(lif_dims) if i != m_idx)
        new_shape = tuple(s for i, s in enumerate(lif_shape) if i != m_idx)

        # Map to NGFF dims
        ngff_dims, ngff_shape = _map_lif_dims_to_ngff(new_dims, new_shape)

        # Extract scale and translation
        scale, translation = _extract_scale_translation(lif_image, ngff_dims)

        # Ensure all spatial dims have scale/translation
        for dim in ngff_dims:
            if dim in SPATIAL_DIMS:
                if dim not in scale:
                    scale[dim] = 1.0
                if dim not in translation:
                    translation[dim] = 0.0

        # Reshape if needed
        if pos_data.shape != ngff_shape:
            pos_data = pos_data.reshape(ngff_shape)

        ngff_image = NgffImage(
            data=pos_data,
            dims=ngff_dims,
            scale=scale,
            translation=translation,
            name=f"{lif_image.name}_pos{pos_idx}",
        )

        column_name = str(pos_idx + 1)
        well_images.append((row_name, column_name, 0, ngff_image))

    return plate, well_images
