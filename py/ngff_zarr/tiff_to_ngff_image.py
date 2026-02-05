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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import zarr

from .ngff_image import NgffImage
from .to_ngff_image import to_ngff_image

if TYPE_CHECKING:
    import tifffile

# Mapping from OME unit symbols/names to NGFF-compatible unit names
# Based on OME-XML specification and UDUNITS-2
OME_UNIT_TO_NGFF: Dict[str, str] = {
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


def _normalize_unit(ome_unit: Optional[str]) -> Optional[str]:
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


def _extract_ome_pixel_metadata(
    tif: "tifffile.TiffFile",
    series_index: int = 0,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, str]]]:
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

    scale: Dict[str, float] = {}
    units: Dict[str, str] = {}

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
                pass

    for ome_attr, dim in unit_mapping.items():
        value = pixels.get(ome_attr)
        if value is not None:
            ngff_unit = _normalize_unit(value)
            if ngff_unit is not None:
                units[dim] = ngff_unit

    if not scale:
        return None, None

    return scale, units if units else None


def tiff_file_to_ngff_images(
    tiff_path: Union[str, Path],
    series: Optional[Union[int, str, List[Union[int, str]]]] = None,
) -> List[Tuple[str, NgffImage]]:
    """
    Convert a TIFF file to a list of (name, NgffImage) pairs.

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

    Returns
    -------
    List[Tuple[str, NgffImage]]
        List of (series_name, ngff_image) tuples. Each NgffImage includes:
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

    Notes
    -----
    For pyramidal TIFFs, only the base resolution (level 0) is extracted.
    Use `to_multiscales()` to regenerate the pyramid with consistent settings.
    """
    try:
        import tifffile
    except ImportError as e:
        msg = "tifffile package is required. Install with: pip install tifffile"
        raise ImportError(msg) from e

    results: List[Tuple[str, NgffImage]] = []

    with tifffile.TiffFile(tiff_path) as tif:
        all_series = tif.series

        # Determine which series to convert
        if series is None or series == "all":
            indices_to_convert = list(range(len(all_series)))
        elif isinstance(series, int):
            # Explicit index - validate it exists
            if series < 0 or series >= len(all_series):
                msg = f"Series index {series} is out of bounds. File has {len(all_series)} series (indices 0-{len(all_series) - 1})."
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
                        msg = f"Series index {s} is out of bounds. File has {len(all_series)} series (indices 0-{len(all_series) - 1})."
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

        # Convert selected series
        for idx in indices_to_convert:
            tiff_series = all_series[idx]

            # Extract OME metadata for this series
            ome_scale, ome_units = _extract_ome_pixel_metadata(tif, series_index=idx)

            # Get the zarr store for this series, using level=0 for base resolution
            # This ensures we always get an Array, not a Group, for pyramidal TIFFs
            store = tif.aszarr(series=idx, level=0)
            root = zarr.open(store, mode="r")

            # Get dims from the series axes
            axes = tiff_series.axes.lower() if tiff_series.axes else None
            dims = None
            if axes:
                # Map TIFF axes to NGFF dims
                dims = tuple(axes)

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
            )

            # Generate series name
            series_name = tiff_series.name if tiff_series.name else f"series_{idx}"
            results.append((series_name, ngff_image))

    return results
