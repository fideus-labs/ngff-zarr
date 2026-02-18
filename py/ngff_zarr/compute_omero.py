# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Compute OMERO metadata from NgffImage data."""

import re
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import dask.array as da

if TYPE_CHECKING:
    import numpy as np

from .ngff_image import NgffImage
from .v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow

# Glasbey color palette from colorcet's glasbey_hv
# Extended from HoloViews default colors using the Glasbey algorithm
# for maximum distinguishability across 256 categorical colors.
# See: https://colorcet.holoviz.org/user_guide/Categorical.html
GLASBEY_COLORS: List[str] = [
    "30A2DA",
    "FC4F30",
    "E5AE38",
    "6D904F",
    "8B8B8B",
    "17BECF",
    "9467BD",
    "D62728",
    "1F77B4",
    "E377C2",
    "8C564B",
    "BCBD22",
    "3A0183",
    "004300",
    "0FFFA9",
    "5E0040",
    "C6BDFF",
    "425052",
    "B80080",
    "FFB7B3",
    "7D0200",
    "6126FF",
    "FFFF9A",
    "AEC9AB",
    "00867C",
    "553A00",
    "94FCFF",
    "00BF00",
    "7D00A0",
    "AB7200",
    "91FF00",
    "01BE8A",
    "00457B",
    "C8826F",
    "FF1F83",
    "DD00FF",
    "057400",
    "644461",
    "888FFF",
    "FFB6F4",
    "536237",
    "CE85FF",
    "686A84",
    "BEB4BE",
    "A56089",
    "95D3FF",
    "0100F8",
    "FF8002",
    "8B2945",
    "ADA06D",
    "53458B",
    "C8FFD9",
    "AA4600",
    "FF798F",
    "83D371",
    "909EBF",
    "9400F5",
    "EBD09B",
    "AD8BB1",
    "00634A",
    "FFDC00",
    "887751",
    "7EABA3",
    "000097",
    "F500C6",
    "653329",
    "006678",
    "04E3C8",
    "A737AE",
    "C5DBE1",
    "4D6EFF",
    "9B9301",
    "CD586B",
    "EFDEFE",
    "795A00",
    "5F889A",
    "B4FF92",
    "5E726B",
    "520066",
    "058751",
    "84206F",
    "3C9605",
    "657300",
    "F1A06C",
    "5F5045",
    "BD004A",
    "D06827",
    "D796AB",
    "895DFF",
    "826C76",
    "2B55B9",
    "6E7CBB",
    "E7D5D3",
    "5D0018",
    "7C3B01",
    "80B17D",
    "C8D97D",
    "00E83B",
    "7CB2FF",
    "FF55FF",
    "A42721",
    "1DE4FF",
    "7DAF3B",
    "7B4B91",
    "E0FF48",
    "6B00C4",
    "CDA897",
    "BE63C4",
    "89CDCE",
    "4603C8",
    "5E9279",
    "414A01",
    "05A79D",
    "CF8C37",
    "FFF8D0",
    "435471",
    "B544FF",
    "CF4993",
    "CFA4DF",
    "94D400",
    "A794DA",
    "2DA558",
    "8DE3B6",
    "A4A99D",
    "6C5CB7",
    "FF7E5E",
    "A7838A",
    "AFBED8",
    "2AC4FF",
    "A6683D",
    "F691FE",
    "874B64",
    "FF0C4B",
    "215E23",
    "4292FF",
    "87839D",
    "672D45",
    "B14F41",
    "004E53",
    "5F1B00",
    "AD4167",
    "503267",
    "D6FFFD",
    "7FB5D1",
    "A9B969",
    "FF96CB",
    "C87495",
    "365039",
    "FFD063",
    "5E5862",
    "879476",
    "A978FF",
    "03C863",
    "E7BED4",
    "D4E3D0",
    "876790",
    "897C27",
    "CDDCFF",
    "AA676B",
    "323474",
    "FF5EA9",
    "009BB0",
    "71FFDD",
    "785C38",
    "50659B",
    "CC00B3",
    "577B55",
    "516E7B",
    "015F92",
    "AABDBE",
    "017F99",
    "04DD97",
    "873A2C",
    "F0968E",
    "75C6AA",
    "70695D",
    "CCDC09",
    "AF8557",
    "D80075",
    "9D3F81",
    "D94500",
    "DD6754",
    "5FFF79",
    "D5B173",
    "62265E",
    "BAA23D",
    "D9F2B3",
    "57028F",
    "A19BAA",
    "4D4A27",
    "A4A9FF",
    "ACE8DB",
    "995901",
    "AC00E2",
    "47822F",
    "CBC3AD",
    "00C5B6",
    "615378",
    "336D68",
    "A59280",
    "8499A2",
    "FD5764",
    "7096D2",
    "728D07",
    "7F004C",
    "1530A0",
    "D1C1E2",
    "C985D0",
    "6C454B",
    "7F0024",
    "00A279",
    "B2A9CF",
    "F90000",
    "B0E9FF",
    "939E50",
    "727A82",
    "D92E55",
    "476101",
    "0059FF",
    "7740B5",
    "ACE460",
    "674525",
    "525D51",
    "957368",
    "A9E49A",
    "A30058",
    "D962F6",
    "8E7DCF",
    "FFBD93",
    "A30092",
    "9AFFB9",
    "A7C2FF",
    "F46200",
    "E5F0FF",
    "B89CA4",
    "609694",
    "FF9F35",
    "8C2900",
    "726B32",
    "DF824E",
    "AF7BD5",
    "BC2D00",
    "7B6FA3",
    "484362",
    "C7A3FF",
    "004D28",
    "C4C68E",
    "E048D7",
    "E7E965",
    "E5C10B",
    "00F4F1",
    "9F5BA2",
    "4C41B7",
    "65338E",
    "767E6C",
    "A98A36",
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
    if not isinstance(color, str):
        raise ValueError(f"Color must be a string, got {type(color).__name__}")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", color):
        raise ValueError(
            f"Color must be a 6-digit hexadecimal string "
            f"without # prefix, got '{color}'"
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
    dense: bool = False,
) -> Tuple[float, float, float, float]:
    """Compute min, max, and quantiles for a single channel.

    Uses dask.array operations to efficiently compute statistics without
    loading all data into memory.

    Args:
        data: Dask array for a single channel (can be multi-dimensional)
        quantiles: Tuple of (low, high) quantile values (e.g., (0.02, 0.98))
        dense: If True, use histogram-based dense sampling for exact quantile
            computation over the full data. If False (default), use Dask's
            approximate percentile algorithm which is faster but may produce
            less accurate quantiles for large multi-chunk datasets.

    Returns:
        Tuple of (min, max, q_low, q_high) as floats

    Note:
        When dense=False, the quantile computation uses Dask's approximate
        percentile algorithm which processes chunks independently and merges
        results. This is memory-efficient and suitable for visualization
        window parameters, though results may differ slightly from exact
        quantiles.

        When dense=True, a histogram with fine bins is computed over the full
        data via ``da.histogram``, then quantiles are derived from the
        cumulative distribution. This gives exact results within bin
        resolution and is well-suited for large datasets where approximate
        percentiles may be inaccurate.
    """
    import numpy as np

    # Flatten the array for statistics computation
    flat_data = data.ravel()

    # Set up lazy computations for min/max (these are exact)
    data_min = da.nanmin(flat_data)
    data_max = da.nanmax(flat_data)

    # First compute min/max to check for all-NaN case
    min_val, max_val = da.compute(data_min, data_max)

    # Handle all-NaN case: if min is NaN, all values are NaN
    if np.isnan(min_val):
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    # Handle constant-value case: all non-NaN values are the same
    if min_val == max_val:
        return (float(min_val), float(max_val), float(min_val), float(max_val))

    if dense:
        return _compute_quantiles_dense(
            flat_data, quantiles, float(min_val), float(max_val), data.dtype
        )
    else:
        return _compute_quantiles_approximate(
            flat_data, quantiles, float(min_val), float(max_val)
        )


def _compute_quantiles_approximate(
    flat_data: da.Array,
    quantiles: Tuple[float, float],
    min_val: float,
    max_val: float,
) -> Tuple[float, float, float, float]:
    """Compute quantiles using Dask's approximate percentile algorithm.

    This is fast and memory-efficient but may produce less accurate results
    for large datasets with many chunks, since it computes per-chunk
    percentiles and merges them.

    Args:
        flat_data: Flattened 1-D dask array
        quantiles: Tuple of (low, high) quantile values
        min_val: Pre-computed minimum value
        max_val: Pre-computed maximum value

    Returns:
        Tuple of (min, max, q_low, q_high) as floats
    """
    # Use dask's approximate percentile computation which works across chunks
    # without loading all data into memory. Convert quantiles to percentiles (0-100).
    # Filter out NaN values before computing percentiles since da.percentile
    # doesn't have a nan-aware variant for 1-D arrays with approximate methods.
    percentiles = [q * 100 for q in quantiles]
    valid_data = flat_data[~da.isnan(flat_data)]

    # da.percentile works on 1-D arrays and uses approximate methods
    # that compute per-chunk percentiles and merge them efficiently
    q_low = da.percentile(valid_data, percentiles[0])
    q_high = da.percentile(valid_data, percentiles[1])

    # Compute quantile statistics (min/max already computed above)
    low_val, high_val = da.compute(q_low, q_high)

    return (min_val, max_val, float(low_val), float(high_val))


def _compute_quantiles_dense(
    flat_data: da.Array,
    quantiles: Tuple[float, float],
    min_val: float,
    max_val: float,
    dtype: "np.dtype",
) -> Tuple[float, float, float, float]:
    """Compute quantiles using histogram-based dense sampling.

    Builds a fine-grained histogram over the full data using ``da.histogram``,
    then derives exact quantiles from the cumulative distribution function.
    This processes all data in a parallelized pass through dask's lazy
    evaluation and gives exact results within bin resolution.

    For integer dtypes, the number of bins equals the number of distinct
    possible values (capped at 65536). For float dtypes, 65536 bins are used.

    Args:
        flat_data: Flattened 1-D dask array
        quantiles: Tuple of (low, high) quantile values
        min_val: Pre-computed minimum value
        max_val: Pre-computed maximum value
        dtype: Data type of the array (used to choose bin count)

    Returns:
        Tuple of (min, max, q_low, q_high) as floats
    """
    import numpy as np

    # Choose number of bins based on dtype
    if np.issubdtype(dtype, np.integer):
        # For integer types, use exact number of distinct values (capped)
        n_bins = min(int(max_val - min_val + 1), 65536)
    else:
        n_bins = 65536

    # Filter NaN values before histogramming
    valid_data = flat_data[~da.isnan(flat_data)]

    # Compute histogram over the full data range.
    # da.histogram computes per-chunk histograms and merges them, so this
    # is memory-efficient and parallelized.
    hist, bin_edges = da.histogram(valid_data, bins=n_bins, range=[min_val, max_val])
    hist = hist.compute()
    bin_edges = np.asarray(bin_edges)

    # Build cumulative distribution function
    cumsum = np.cumsum(hist)
    total_count = cumsum[-1]

    # Compute quantile values via interpolation on the CDF.
    # cumsum[i] is the count of values <= bin_edges[i+1].
    # We interpolate to find the bin edge where the cumulative count
    # crosses the desired quantile fraction.
    target_low = quantiles[0] * total_count
    target_high = quantiles[1] * total_count

    # Use bin right-edges for CDF interpolation
    low_val = float(np.interp(target_low, cumsum, bin_edges[1:]))
    high_val = float(np.interp(target_high, cumsum, bin_edges[1:]))

    return (min_val, max_val, low_val, high_val)


def compute_omero_from_ngff_image(
    ngff_image: NgffImage,
    quantiles: Tuple[float, float] = (0.02, 0.98),
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    dense: bool = False,
) -> Omero:
    """Compute OMERO metadata from an NgffImage.

    This function computes visualization parameters (OMERO metadata) from image data:
    - min/max: The actual data range (exact values)
    - start/end: Display window based on quantiles (default 2% and 98%)

    For multi-channel images (with 'c' dimension), statistics are computed
    separately for each channel, resulting in per-channel OMERO windows.

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
                If not provided, uses channel_names from NgffImage if available.
                If channel_names is also not available or has fewer entries than
                channels, uses empty strings for remaining channels.
                When explicitly provided, must have at least as many labels as
                channels (ValueError raised if insufficient).
        dense: If True, use histogram-based dense sampling for exact quantile
               computation over the full data. If False (default), use Dask's
               approximate percentile algorithm which is faster but may
               produce less accurate quantiles for large multi-chunk datasets.

    Returns:
        Omero metadata with computed window parameters for each channel.

    Raises:
        ValueError: If quantiles are invalid, colors are invalid format,
                    or not enough colors/labels explicitly provided.

    Note:
        The behavior differs between explicit labels and channel_names from NgffImage:
        - Explicit labels parameter: Must provide at least as many labels as
          channels, otherwise ValueError is raised. This ensures intentional
          labeling is complete.
        - channel_names from NgffImage: Can be shorter than the number of
          channels, in which case remaining channels get empty string labels.
          This allows partial metadata from sources like OME-XML where not all
          channels may have names.

    Example:
        >>> # Using channel_names from NgffImage (e.g., from OME-TIFF)
        >>> image = to_ngff_image(data, dims=["c", "z", "y", "x"])
        >>> image.channel_names = ["DAPI", "GFP", "RFP"]
        >>> omero = compute_omero_from_ngff_image(image)
        >>> omero.channels[0].label  # "DAPI"

        >>> # Explicit labels override channel_names
        >>> omero = compute_omero_from_ngff_image(
        ...     image, labels=["Red", "Green", "Blue"]
        ... )
        >>> omero.channels[0].label  # "Red"
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
    elif ngff_image.channel_names is not None:
        # Use channel names from the image if available
        if len(ngff_image.channel_names) >= n_channels:
            channel_labels = list(ngff_image.channel_names[:n_channels])
        else:
            # Pad with empty strings if not enough names
            channel_labels = list(ngff_image.channel_names) + [""] * (
                n_channels - len(ngff_image.channel_names)
            )
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
            channel_data, quantiles, dense=dense
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
    dense: bool = False,
) -> Omero:
    """Compute OMERO metadata from a Multiscales object.

    This is a convenience function that computes OMERO metadata from the
    highest resolution image in a multiscales pyramid.

    Uses memory-efficient computation that processes data in chunks. This
    makes it safe to use with very large datasets without risk of memory
    exhaustion.

    Args:
        multiscales: The Multiscales object to compute metadata for
        quantiles: Tuple of (low, high) quantile values for the display window.
        colors: Optional list of hex color strings for each channel.
        labels: Optional list of label strings for each channel.
        dense: If True, use histogram-based dense sampling for exact quantile
            computation over the full data. If False (default), use Dask's
            approximate percentile algorithm.

    Returns:
        Omero metadata with computed window parameters for each channel.
    """
    from .multiscales import Multiscales

    if not isinstance(multiscales, Multiscales):
        raise TypeError(f"Expected Multiscales, got {type(multiscales)}")

    if not multiscales.images:
        raise ValueError("Multiscales has no images")

    # Always use the highest resolution (first) image for accurate statistics
    image = multiscales.images[0]

    return compute_omero_from_ngff_image(
        image,
        quantiles=quantiles,
        colors=colors,
        labels=labels,
        dense=dense,
    )
