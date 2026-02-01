# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Compute OMERO metadata from NgffImage data."""

from typing import List, Optional, Sequence, Tuple, Union

import dask.array as da

from .ngff_image import NgffImage
from .v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow

# Glasbey color palette - optimized for distinguishability in microscopy
# These colors cycle through distinct hues that work well for multi-channel imaging
GLASBEY_COLORS: List[str] = [
    "FF0000",  # Red
    "00FF00",  # Green
    "0000FF",  # Blue
    "FFFF00",  # Yellow
    "FF00FF",  # Magenta
    "00FFFF",  # Cyan
    "FF8000",  # Orange
    "8000FF",  # Purple
    "00FF80",  # Spring Green
    "FF0080",  # Rose
    "80FF00",  # Lime
    "0080FF",  # Sky Blue
    "FF8080",  # Light Coral
    "80FF80",  # Light Green
    "8080FF",  # Light Blue
    "FFFF80",  # Light Yellow
    "FF80FF",  # Light Magenta
    "80FFFF",  # Light Cyan
    "804000",  # Brown
    "008040",  # Dark Cyan
]


def _validate_quantiles(quantiles: Tuple[float, float]) -> None:
    """Validate that quantiles are in valid range and properly ordered.

    Args:
        quantiles: Tuple of (low, high) quantile values

    Raises:
        ValueError: If quantiles are invalid
    """
    if len(quantiles) != 2:
        raise ValueError(f"Expected 2 quantiles, got {len(quantiles)}")

    low, high = quantiles
    if not (0.0 <= low <= 1.0):
        raise ValueError(f"Low quantile must be between 0 and 1, got {low}")
    if not (0.0 <= high <= 1.0):
        raise ValueError(f"High quantile must be between 0 and 1, got {high}")
    if low >= high:
        raise ValueError(
            f"Low quantile must be less than high quantile, got ({low}, {high})"
        )


def _validate_color(color: str) -> None:
    """Validate that a color is a valid 6-digit hexadecimal string.

    Args:
        color: Hex color string (without # prefix)

    Raises:
        ValueError: If color is invalid
    """
    import re

    if not isinstance(color, str):
        raise ValueError(f"Color must be a string, got {type(color).__name__}")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", color):
        raise ValueError(
            f"Color must be a 6-digit hexadecimal string without # prefix, got '{color}'"
        )


def get_default_colors(n_channels: int) -> List[str]:
    """Get default colors for channels.

    For a single channel, returns white (FFFFFF).
    For multiple channels, uses the Glasbey color progression.

    Args:
        n_channels: Number of channels

    Returns:
        List of hex color strings (without # prefix)
    """
    if n_channels == 1:
        return ["FFFFFF"]
    # Cycle through Glasbey colors if we have more channels than colors
    return [GLASBEY_COLORS[i % len(GLASBEY_COLORS)] for i in range(n_channels)]


def _compute_channel_statistics(
    data: da.Array,
    quantiles: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Compute min, max, and quantiles for a single channel.

    Uses dask.array operations to efficiently compute statistics in one pass.
    Handles NaN values by using nan-aware functions.

    Args:
        data: Dask array for a single channel (can be multi-dimensional)
        quantiles: Tuple of (low, high) quantile values (e.g., (0.02, 0.98))

    Returns:
        Tuple of (min, max, q_low, q_high) as floats
    """
    # Flatten the array for statistics computation
    flat_data = data.ravel()

    # Set up lazy computations for min/max
    data_min = da.nanmin(flat_data)
    data_max = da.nanmax(flat_data)

    # For quantile computation, we need to handle the case where dask
    # can't compute quantiles across multiple chunks without an axis.
    # We rechunk to a single chunk for quantile computation.
    # This loads all data into memory, but is necessary for exact quantiles.
    flat_single_chunk = flat_data.rechunk(-1)
    q_values = da.nanquantile(flat_single_chunk, list(quantiles), axis=0)

    # Compute all statistics in a single pass through the data
    min_val, max_val, quantile_vals = da.compute(data_min, data_max, q_values)

    return (
        float(min_val),
        float(max_val),
        float(quantile_vals[0]),
        float(quantile_vals[1]),
    )


def compute_omero_from_ngff_image(
    ngff_image: NgffImage,
    quantiles: Tuple[float, float] = (0.02, 0.98),
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
) -> Omero:
    """Compute OMERO metadata from an NgffImage.

    This function computes visualization parameters (OMERO metadata) from image data:
    - min/max: The actual data range
    - start/end: Display window based on quantiles (default 2% and 98%)

    For multi-channel images (with 'c' dimension), statistics are computed
    separately for each channel, resulting in per-channel OMERO windows.

    Memory requirements: This function loads the entire flattened channel data
    into a single chunk for exact quantile computation. For large images,
    use the lowest resolution image from a multiscales pyramid.

    Edge cases:
    - If all values in a channel are NaN, the statistics will be NaN.
    - If a channel has constant values, min/max/start/end will all be the same.

    Args:
        ngff_image: The NgffImage to compute metadata for
        quantiles: Tuple of (low, high) quantile values for the display window.
                   Must be between 0 and 1, with low < high.
                   Default is (0.02, 0.98) for 2% and 98% quantiles.
        colors: Optional list of hex color strings (without #) for each channel.
                Must be 6-digit hexadecimal strings (e.g., "FF0000" for red).
                If not provided, uses white for single channel or Glasbey
                progression for multi-channel.
        labels: Optional list of label strings for each channel.
                If not provided, uses empty strings.

    Returns:
        Omero metadata with computed window parameters for each channel.

    Raises:
        ValueError: If quantiles are invalid, colors are invalid format,
                    or not enough colors/labels provided.

    Example:
        >>> image = to_ngff_image(data, dims=["c", "z", "y", "x"])
        >>> omero = compute_omero_from_ngff_image(image)
        >>> multiscales.metadata.omero = omero
    """
    # Validate quantiles
    _validate_quantiles(quantiles)

    data = ngff_image.data
    dims = list(ngff_image.dims)

    # Check if there's a channel dimension
    has_channel_dim = "c" in dims
    if has_channel_dim:
        c_index = dims.index("c")
        n_channels = data.shape[c_index]
    else:
        n_channels = 1

    # Get colors
    if colors is not None:
        if len(colors) < n_channels:
            raise ValueError(
                f"Not enough colors provided. Got {len(colors)}, need {n_channels}."
            )
        # Validate each color
        for color in colors[:n_channels]:
            _validate_color(color)
        channel_colors = list(colors[:n_channels])
    else:
        channel_colors = get_default_colors(n_channels)

    # Get labels
    if labels is not None:
        if len(labels) < n_channels:
            raise ValueError(
                f"Not enough labels provided. Got {len(labels)}, need {n_channels}."
            )
        channel_labels = list(labels[:n_channels])
    else:
        channel_labels = [""] * n_channels

    # Compute statistics for each channel
    channels: List[OmeroChannel] = []

    for ch_idx in range(n_channels):
        if has_channel_dim:
            # Extract this channel's data
            # Build a slice tuple to select this channel
            slices: List[Union[int, slice]] = [slice(None)] * len(data.shape)
            slices[c_index] = ch_idx
            channel_data = data[tuple(slices)]
        else:
            channel_data = data

        # Compute statistics
        data_min, data_max, q_low, q_high = _compute_channel_statistics(
            channel_data, quantiles
        )

        # Create OMERO window
        window = OmeroWindow(
            min=data_min,
            max=data_max,
            start=q_low,
            end=q_high,
        )

        # Create channel metadata
        channel = OmeroChannel(
            color=channel_colors[ch_idx],
            window=window,
            label=channel_labels[ch_idx],
        )
        channels.append(channel)

    return Omero(channels=channels)


def compute_omero_from_multiscales(
    multiscales: "Multiscales",  # noqa: F821
    quantiles: Tuple[float, float] = (0.02, 0.98),
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    use_lowest_resolution: bool = True,
) -> Omero:
    """Compute OMERO metadata from a Multiscales object.

    This is a convenience function that computes OMERO metadata from the
    highest or lowest resolution image in a multiscales pyramid.

    Args:
        multiscales: The Multiscales object to compute metadata for
        quantiles: Tuple of (low, high) quantile values for the display window.
        colors: Optional list of hex color strings for each channel.
        labels: Optional list of label strings for each channel.
        use_lowest_resolution: If True (default), use the lowest resolution
            (smallest) image for faster computation. If False, use the highest
            resolution (largest) image for more accurate statistics.

    Returns:
        Omero metadata with computed window parameters for each channel.
    """
    from .multiscales import Multiscales

    if not isinstance(multiscales, Multiscales):
        raise TypeError(f"Expected Multiscales, got {type(multiscales)}")

    if not multiscales.images:
        raise ValueError("Multiscales has no images")

    # Select which image to use based on resolution preference
    if use_lowest_resolution:
        # Use the last image (lowest resolution, smallest)
        image = multiscales.images[-1]
    else:
        # Use the first image (highest resolution, largest)
        image = multiscales.images[0]

    return compute_omero_from_ngff_image(
        image,
        quantiles=quantiles,
        colors=colors,
        labels=labels,
    )
