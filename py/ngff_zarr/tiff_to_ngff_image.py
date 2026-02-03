# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Convert TIFF files to OME-NGFF.

This module provides functions to convert TIFF files (including multi-series
OME-TIFF files) to NgffImage objects for further processing into OME-Zarr format.

Requires the `tifffile` package: pip install tifffile
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import zarr

from .to_ngff_image import to_ngff_image
from .ngff_image import NgffImage


def tiff_file_to_ngff_images(
    tiff_path: Union[str, Path],
    series: Optional[Union[int, str, List[Union[int, str]]]] = None,
) -> List[Tuple[str, NgffImage]]:
    """
    Convert a TIFF file to a list of (name, NgffImage) pairs.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the TIFF file.
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
    >>> from ngff_zarr import tiff_file_to_ngff_images
    >>> # Convert all series
    >>> images = tiff_file_to_ngff_images("sample.ome.tiff")
    >>> # Convert specific series by index
    >>> images = tiff_file_to_ngff_images("sample.ome.tiff", series=0)
    >>> # Convert series matching pattern
    >>> images = tiff_file_to_ngff_images("sample.ome.tiff", series="*area_1*")
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
                msg = f"Series index {series} is out of bounds. File has {len(all_series)} series (indices 0-{len(all_series)-1})."
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
                        msg = f"Series index {s} is out of bounds. File has {len(all_series)} series (indices 0-{len(all_series)-1})."
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

            # Get the zarr store for this series
            store = tif.aszarr(series=idx)
            root = zarr.open(store, mode="r")

            # Convert to NgffImage
            ngff_image = to_ngff_image(root)

            # Generate series name
            series_name = tiff_series.name if tiff_series.name else f"series_{idx}"
            results.append((series_name, ngff_image))

    return results
