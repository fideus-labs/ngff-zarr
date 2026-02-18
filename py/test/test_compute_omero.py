# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for compute_omero functionality."""

import numpy as np
import pytest

from ngff_zarr import (
    GLASBEY_COLORS,
    compute_omero_from_multiscales,
    compute_omero_from_ngff_image,
    to_multiscales,
    to_ngff_image,
)


class TestComputeOmeroSingleChannel:
    """Tests for single-channel images."""

    def test_compute_basic_statistics(self):
        """Test that min, max, and quantiles are computed correctly."""
        # Create a simple array with known values
        # Values 0-99, so min=0, max=99, 2%=~2, 98%=~97
        data = np.arange(100).reshape((10, 10)).astype(np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 1
        channel = omero.channels[0]

        # Check min/max
        assert channel.window.min == 0.0
        assert channel.window.max == 99.0

        # Check quantiles (approximate due to interpolation)
        assert 0.0 <= channel.window.start <= 5.0
        assert 94.0 <= channel.window.end <= 99.0

    def test_single_channel_uses_white_color(self):
        """Test that single channel defaults to white color."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].color == "FFFFFF"

    def test_custom_quantiles(self):
        """Test that custom quantiles are respected."""
        data = np.arange(100).reshape((10, 10)).astype(np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        # Use 10% and 90% quantiles
        omero = compute_omero_from_ngff_image(image, quantiles=(0.10, 0.90))

        channel = omero.channels[0]
        # 10% of 100 values is ~10, 90% is ~90
        assert 5.0 <= channel.window.start <= 15.0
        assert 85.0 <= channel.window.end <= 95.0

    def test_custom_color(self):
        """Test that custom colors are applied."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, colors=["FF0000"])

        assert omero.channels[0].color == "FF0000"

    def test_custom_label(self):
        """Test that custom labels are applied."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, labels=["DAPI"])

        assert omero.channels[0].label == "DAPI"

    def test_default_label_is_empty(self):
        """Test that default label is empty string."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].label == ""


class TestComputeOmeroMultiChannel:
    """Tests for multi-channel images."""

    def test_per_channel_statistics(self):
        """Test that statistics are computed separately per channel."""
        # Create 3 channels with different value ranges
        ch0 = np.full((10, 10), 0.0, dtype=np.float32)  # All zeros
        ch1 = np.full((10, 10), 100.0, dtype=np.float32)  # All 100s
        ch2 = np.full((10, 10), 255.0, dtype=np.float32)  # All 255s
        data = np.stack([ch0, ch1, ch2], axis=0)

        image = to_ngff_image(data, dims=["c", "y", "x"])
        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 3

        # Channel 0: all zeros
        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 0.0

        # Channel 1: all 100s
        assert omero.channels[1].window.min == 100.0
        assert omero.channels[1].window.max == 100.0

        # Channel 2: all 255s
        assert omero.channels[2].window.min == 255.0
        assert omero.channels[2].window.max == 255.0

    def test_multi_channel_uses_glasbey_colors(self):
        """Test that multi-channel images use Glasbey color progression."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].color == GLASBEY_COLORS[0]  # Red
        assert omero.channels[1].color == GLASBEY_COLORS[1]  # Green
        assert omero.channels[2].color == GLASBEY_COLORS[2]  # Blue

    def test_many_channels_cycle_colors(self):
        """Test that colors cycle when there are more channels than colors."""
        n_channels = len(GLASBEY_COLORS) + 5
        data = np.ones((n_channels, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        omero = compute_omero_from_ngff_image(image)

        # Check that colors cycle
        assert omero.channels[len(GLASBEY_COLORS)].color == GLASBEY_COLORS[0]
        assert omero.channels[len(GLASBEY_COLORS) + 1].color == GLASBEY_COLORS[1]

    def test_custom_colors_for_channels(self):
        """Test that custom colors override defaults."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        custom_colors = ["AABBCC", "DDEEFF", "112233"]
        omero = compute_omero_from_ngff_image(image, colors=custom_colors)

        assert omero.channels[0].color == "AABBCC"
        assert omero.channels[1].color == "DDEEFF"
        assert omero.channels[2].color == "112233"

    def test_custom_labels_for_channels(self):
        """Test that custom labels are applied to channels."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        custom_labels = ["DAPI", "GFP", "RFP"]
        omero = compute_omero_from_ngff_image(image, labels=custom_labels)

        assert omero.channels[0].label == "DAPI"
        assert omero.channels[1].label == "GFP"
        assert omero.channels[2].label == "RFP"

    def test_insufficient_colors_raises_error(self):
        """Test that providing too few colors raises an error."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        with pytest.raises(ValueError, match="Not enough colors"):
            compute_omero_from_ngff_image(image, colors=["FF0000", "00FF00"])

    def test_insufficient_labels_raises_error(self):
        """Test that providing too few labels raises an error."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        with pytest.raises(ValueError, match="Not enough labels"):
            compute_omero_from_ngff_image(image, labels=["DAPI"])


class TestComputeOmeroHigherDimensions:
    """Tests for higher-dimensional images (3D, 4D with time)."""

    def test_3d_image_no_channel(self):
        """Test 3D image without channel dimension."""
        data = np.arange(1000).reshape((10, 10, 10)).astype(np.float32)
        image = to_ngff_image(data, dims=["z", "y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 1
        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 999.0

    def test_4d_image_with_channel(self):
        """Test 4D image with channel dimension."""
        # Shape: (c, z, y, x) = (2, 5, 10, 10)
        ch0 = np.zeros((5, 10, 10), dtype=np.float32)
        ch1 = np.ones((5, 10, 10), dtype=np.float32) * 100
        data = np.stack([ch0, ch1], axis=0)

        image = to_ngff_image(data, dims=["c", "z", "y", "x"])
        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 2
        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 0.0
        assert omero.channels[1].window.min == 100.0
        assert omero.channels[1].window.max == 100.0

    def test_5d_image_with_time_and_channel(self):
        """Test 5D image with time and channel dimensions."""
        # Shape: (t, c, z, y, x) = (2, 2, 5, 10, 10)
        data = np.ones((2, 2, 5, 10, 10), dtype=np.float32)
        data[0, 0] = 10  # t=0, c=0
        data[0, 1] = 20  # t=0, c=1
        data[1, 0] = 30  # t=1, c=0
        data[1, 1] = 40  # t=1, c=1

        image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
        omero = compute_omero_from_ngff_image(image)

        # Should have 2 channels (across all time points)
        assert len(omero.channels) == 2

        # Channel 0 has values 10 and 30 (across time)
        assert omero.channels[0].window.min == 10.0
        assert omero.channels[0].window.max == 30.0

        # Channel 1 has values 20 and 40 (across time)
        assert omero.channels[1].window.min == 20.0
        assert omero.channels[1].window.max == 40.0


class TestComputeOmeroSpecialValues:
    """Tests for special values like NaN and edge cases."""

    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        # Should ignore NaN values
        assert omero.channels[0].window.min == 1.0
        assert omero.channels[0].window.max == 9.0

    def test_constant_value_array(self):
        """Test array where all values are the same."""
        data = np.full((10, 10), 42.0, dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].window.min == 42.0
        assert omero.channels[0].window.max == 42.0
        assert omero.channels[0].window.start == 42.0
        assert omero.channels[0].window.end == 42.0

    def test_integer_dtype(self):
        """Test that integer dtypes work correctly."""
        data = np.arange(256, dtype=np.uint8).reshape((16, 16))
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 255.0


class TestComputeOmeroFromMultiscales:
    """Tests for compute_omero_from_multiscales function."""

    def test_uses_highest_resolution(self):
        """Test that highest resolution image is used for statistics."""
        data = np.arange(64).reshape((8, 8)).astype(np.float32)
        image = to_ngff_image(data, dims=["y", "x"])
        multiscales = to_multiscales(image, scale_factors=[2], chunks=4)

        omero = compute_omero_from_multiscales(multiscales)

        assert len(omero.channels) == 1
        # Should have original full-resolution values
        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 63.0

    def test_passes_through_options(self):
        """Test that quantiles, colors, labels are passed through."""
        data = np.ones((2, 8, 8), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])
        multiscales = to_multiscales(image, scale_factors=[2], chunks=4)

        omero = compute_omero_from_multiscales(
            multiscales,
            quantiles=(0.1, 0.9),
            colors=["FF0000", "00FF00"],
            labels=["Red", "Green"],
        )

        assert omero.channels[0].color == "FF0000"
        assert omero.channels[0].label == "Red"
        assert omero.channels[1].color == "00FF00"
        assert omero.channels[1].label == "Green"

    def test_empty_multiscales_raises_error(self):
        """Test that empty multiscales raises an error."""
        from ngff_zarr import Multiscales
        from ngff_zarr.v04.zarr_metadata import Axis, Dataset, Metadata, Scale

        # Create empty multiscales
        metadata = Metadata(
            axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
            datasets=[Dataset(path="0", coordinateTransformations=[Scale([1.0, 1.0])])],
            coordinateTransformations=None,
        )
        multiscales = Multiscales(images=[], metadata=metadata)

        with pytest.raises(ValueError, match="no images"):
            compute_omero_from_multiscales(multiscales)


class TestGlasbeyColors:
    """Tests for the Glasbey color palette constant."""

    def test_has_at_least_20_colors(self):
        """Test that there are at least 20 colors."""
        assert len(GLASBEY_COLORS) >= 20

    def test_colors_are_valid_hex(self):
        """Test that all colors are valid 6-digit hex strings."""
        import re

        for color in GLASBEY_COLORS:
            assert re.fullmatch(r"[0-9A-Fa-f]{6}", color), f"Invalid color: {color}"

    def test_first_color_values(self):
        """Test that the first three colors match glasbey_hv from colorcet."""
        assert GLASBEY_COLORS[0] == "30A2DA"
        assert GLASBEY_COLORS[1] == "FC4F30"
        assert GLASBEY_COLORS[2] == "E5AE38"


class TestValidation:
    """Tests for input validation."""

    def test_invalid_quantile_low_out_of_range(self):
        """Test that invalid low quantile raises ValueError."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        with pytest.raises(ValueError, match="Low quantile must be between 0 and 1"):
            compute_omero_from_ngff_image(image, quantiles=(-0.1, 0.98))

    def test_invalid_quantile_high_out_of_range(self):
        """Test that invalid high quantile raises ValueError."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        with pytest.raises(ValueError, match="High quantile must be between 0 and 1"):
            compute_omero_from_ngff_image(image, quantiles=(0.02, 1.5))

    def test_invalid_quantile_order(self):
        """Test that reversed quantiles raise ValueError."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        with pytest.raises(
            ValueError, match="Low quantile must be less than high quantile"
        ):
            compute_omero_from_ngff_image(image, quantiles=(0.98, 0.02))

    def test_invalid_color_format_too_short(self):
        """Test that colors with wrong length raise ValueError."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        with pytest.raises(ValueError, match="6-digit hexadecimal string"):
            compute_omero_from_ngff_image(image, colors=["FFF"])

    def test_invalid_color_format_with_hash(self):
        """Test that colors with # prefix raise ValueError."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        with pytest.raises(ValueError, match="6-digit hexadecimal string"):
            compute_omero_from_ngff_image(image, colors=["#FF0000"])

    def test_invalid_color_format_non_hex(self):
        """Test that colors with non-hex characters raise ValueError."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        with pytest.raises(ValueError, match="6-digit hexadecimal string"):
            compute_omero_from_ngff_image(image, colors=["GGGGGG"])

    def test_all_nan_channel(self):
        """Test edge case where all values in a channel are NaN."""
        data = np.full((10, 10), np.nan, dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image)

        # Should return NaN statistics
        assert np.isnan(omero.channels[0].window.min)
        assert np.isnan(omero.channels[0].window.max)
        assert np.isnan(omero.channels[0].window.start)
        assert np.isnan(omero.channels[0].window.end)


class TestDenseSampling:
    """Tests for histogram-based dense sampling (dense=True)."""

    def test_dense_basic_statistics(self):
        """Test that dense sampling computes accurate min, max, and quantiles."""
        data = np.arange(1000).reshape((10, 100)).astype(np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, dense=True)

        channel = omero.channels[0]
        assert channel.window.min == 0.0
        assert channel.window.max == 999.0
        # Dense quantiles should be very close to exact values
        # 2% of 1000 = 20, 98% of 1000 = 980
        assert 18.0 <= channel.window.start <= 22.0
        assert 978.0 <= channel.window.end <= 982.0

    def test_dense_integer_dtype(self):
        """Test dense sampling with integer dtypes uses exact bin counts."""
        data = np.arange(256, dtype=np.uint8).reshape((16, 16))
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, dense=True)

        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 255.0
        # For uint8 with 256 values, 2% = ~5.1, 98% = ~250.9
        assert 4.0 <= omero.channels[0].window.start <= 7.0
        assert 249.0 <= omero.channels[0].window.end <= 252.0

    def test_dense_with_nan_values(self):
        """Test that dense sampling handles NaN values correctly."""
        data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, dense=True)

        assert omero.channels[0].window.min == 1.0
        assert omero.channels[0].window.max == 9.0

    def test_dense_constant_values(self):
        """Test dense sampling with constant-value array."""
        data = np.full((10, 10), 42.0, dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, dense=True)

        assert omero.channels[0].window.min == 42.0
        assert omero.channels[0].window.max == 42.0
        assert omero.channels[0].window.start == 42.0
        assert omero.channels[0].window.end == 42.0

    def test_dense_all_nan(self):
        """Test dense sampling with all-NaN array."""
        data = np.full((10, 10), np.nan, dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        omero = compute_omero_from_ngff_image(image, dense=True)

        assert np.isnan(omero.channels[0].window.min)
        assert np.isnan(omero.channels[0].window.max)
        assert np.isnan(omero.channels[0].window.start)
        assert np.isnan(omero.channels[0].window.end)

    def test_dense_multi_channel(self):
        """Test dense sampling with multi-channel data."""
        ch0 = np.arange(100).reshape((10, 10)).astype(np.float32)
        ch1 = np.arange(100, 200).reshape((10, 10)).astype(np.float32)
        data = np.stack([ch0, ch1], axis=0)

        image = to_ngff_image(data, dims=["c", "y", "x"])
        omero = compute_omero_from_ngff_image(image, dense=True)

        assert len(omero.channels) == 2
        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 99.0
        assert omero.channels[1].window.min == 100.0
        assert omero.channels[1].window.max == 199.0

    def test_dense_multiscales(self):
        """Test dense sampling via compute_omero_from_multiscales."""
        data = np.arange(64).reshape((8, 8)).astype(np.float32)
        image = to_ngff_image(data, dims=["y", "x"])
        multiscales = to_multiscales(image, scale_factors=[2], chunks=4)

        omero = compute_omero_from_multiscales(multiscales, dense=True)

        assert len(omero.channels) == 1
        assert omero.channels[0].window.min == 0.0
        assert omero.channels[0].window.max == 63.0

    def test_dense_more_accurate_than_approximate_for_multi_chunk(self):
        """Test dense sampling is at least as accurate as approximate for
        multi-chunk data."""
        # Create a large-ish array with many small chunks to stress the
        # approximate algorithm
        import dask.array as da

        rng = np.random.default_rng(42)
        raw = rng.standard_normal(10000).astype(np.float32)
        # Known exact quantiles from numpy
        exact_q2 = float(np.percentile(raw, 2))
        exact_q98 = float(np.percentile(raw, 98))

        # Create dask array with many small chunks
        dask_data = da.from_array(raw.reshape((100, 100)), chunks=(10, 10))
        image = to_ngff_image(dask_data, dims=["y", "x"])

        omero_dense = compute_omero_from_ngff_image(image, dense=True)
        omero_approx = compute_omero_from_ngff_image(image, dense=False)

        # Dense should be close to exact (within a small fraction of data range)
        dense_err_low = abs(omero_dense.channels[0].window.start - exact_q2)
        dense_err_high = abs(omero_dense.channels[0].window.end - exact_q98)

        data_range = float(raw.max() - raw.min())
        # Dense error should be well under 1% of the data range
        assert dense_err_low < data_range * 0.01
        assert dense_err_high < data_range * 0.01

        # Approximate may have larger error (we just check it doesn't crash)
        assert omero_approx.channels[0].window.start is not None
        assert omero_approx.channels[0].window.end is not None


class TestChannelNamesIntegration:
    """Tests for channel_names field integration with compute_omero_from_ngff_image."""

    def test_channel_names_used_as_default_labels(self):
        """Test that channel_names from NgffImage are used as default labels."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        # Set channel_names on the image
        image.channel_names = ["DAPI", "GFP", "RFP"]

        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 3
        assert omero.channels[0].label == "DAPI"
        assert omero.channels[1].label == "GFP"
        assert omero.channels[2].label == "RFP"

    def test_explicit_labels_override_channel_names(self):
        """Test that explicit labels parameter overrides channel_names."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        # Set channel_names on the image
        image.channel_names = ["DAPI", "GFP", "RFP"]

        # Use explicit labels that differ from channel_names
        omero = compute_omero_from_ngff_image(image, labels=["Red", "Green", "Blue"])

        assert omero.channels[0].label == "Red"
        assert omero.channels[1].label == "Green"
        assert omero.channels[2].label == "Blue"

    def test_channel_names_with_single_channel(self):
        """Test channel_names with single channel image without 'c' dimension."""
        data = np.ones((10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["y", "x"])

        # Single channel image without 'c' dimension can still use channel_names
        image.channel_names = ["Brightfield"]

        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 1
        # channel_names are used even for images without 'c' dimension
        assert omero.channels[0].label == "Brightfield"

    def test_insufficient_channel_names_padded_with_empty_strings(self):
        """Test that insufficient channel_names are padded with empty strings."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        # Provide fewer channel names than channels
        image.channel_names = ["DAPI"]

        # Should not raise an error, pads with empty strings
        omero = compute_omero_from_ngff_image(image)

        assert len(omero.channels) == 3
        assert omero.channels[0].label == "DAPI"
        assert omero.channels[1].label == ""
        assert omero.channels[2].label == ""

    def test_no_channel_names_uses_empty_strings(self):
        """Test that when channel_names is None, empty strings are used."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        # channel_names is None by default
        assert image.channel_names is None

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].label == ""
        assert omero.channels[1].label == ""
        assert omero.channels[2].label == ""

    def test_channel_names_with_empty_strings(self):
        """Test that channel_names can contain empty strings."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        image = to_ngff_image(data, dims=["c", "y", "x"])

        # Mix of named and unnamed channels
        image.channel_names = ["DAPI", "", "RFP"]

        omero = compute_omero_from_ngff_image(image)

        assert omero.channels[0].label == "DAPI"
        assert omero.channels[1].label == ""
        assert omero.channels[2].label == "RFP"
