# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test pyramidal TIFF files with RGB/S-axis to ensure consistent reshaping across levels."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_pyramidal_rgb_tiff_channel_consistency():
    """Test that pyramidal RGB TIFF files have consistent channel dimensions across all levels.

    This is a regression test for a bug where level 0 was reshaped correctly (S-axis -> c dimension)
    but other pyramid levels were not, causing shape mismatches and data corruption.
    """
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a pyramidal RGB TIFF with S-axis (sample dimension)
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "pyramid_rgb.ome.tiff"

    # Create multi-resolution RGB data
    # Level 0: 256x256 RGB
    level0 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    # Level 1: 128x128 RGB (downsample level 0)
    level1 = level0[::2, ::2, :]
    # Level 2: 64x64 RGB (downsample level 0)
    level2 = level0[::4, ::4, :]

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        # Write main image with SubIFDs for pyramid levels
        tif.write(
            level0,
            photometric="rgb",
            metadata={"axes": "YXS"},  # S-axis for RGB samples
            tile=(128, 128),
            subifds=2,  # Reserve space for 2 pyramid levels
        )
        # Write pyramid levels into SubIFDs
        tif.write(
            level1,
            photometric="rgb",
            metadata={"axes": "YXS"},
            tile=(128, 128),
            subfiletype=1,  # Reduced resolution
        )
        tif.write(
            level2,
            photometric="rgb",
            metadata={"axes": "YXS"},
            tile=(128, 128),
            subfiletype=1,  # Reduced resolution
        )

    # Convert with pyramid reuse enabled
    results = tiff_file_to_ngff_images(tiff_path, reuse_existing_pyramids=True)

    assert len(results) == 1, "Should have one series"
    name, result = results[0]

    # Should be a NgffMultiscales object (pyramid was reused)
    from ngff_zarr.multiscales import NgffMultiscales

    assert isinstance(result, NgffMultiscales), (
        "Should return NgffMultiscales for pyramidal TIFF"
    )

    # Check that we have 3 pyramid levels
    assert len(result.images) == 3, (
        f"Expected 3 pyramid levels, got {len(result.images)}"
    )

    # Verify all levels have consistent dimensions
    for i, img in enumerate(result.images):
        assert img.dims == ("y", "x", "c"), (
            f"Level {i} should have dims ('y', 'x', 'c'), got {img.dims}"
        )

        # Check shape consistency
        expected_size = 256 // (2**i)  # 256, 128, 64
        expected_shape = (expected_size, expected_size, 3)
        assert img.data.shape == expected_shape, (
            f"Level {i} should have shape {expected_shape}, got {img.data.shape}"
        )

    # Verify data can be computed without errors
    for i, img in enumerate(result.images):
        # Compute a small slice to ensure lazy loading works
        slice_data = img.data[:4, :4, :].compute()
        assert slice_data.shape == (4, 4, 3), (
            f"Level {i} slice should have shape (4, 4, 3), got {slice_data.shape}"
        )
        assert slice_data.dtype == np.uint8, (
            f"Level {i} should preserve uint8 dtype, got {slice_data.dtype}"
        )


def test_pyramidal_grayscale_tiff():
    """Test that pyramidal grayscale TIFF files work correctly (no S-axis)."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "pyramid_gray.ome.tiff"

    # Create multi-resolution grayscale data
    level0 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    level1 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            level0,
            photometric="minisblack",
            metadata={"axes": "YX"},
            tile=(128, 128),
            subifds=1,  # Reserve space for 1 pyramid level
        )
        tif.write(
            level1,
            photometric="minisblack",
            metadata={"axes": "YX"},
            tile=(128, 128),
            subfiletype=1,  # Reduced resolution
        )

    results = tiff_file_to_ngff_images(tiff_path, reuse_existing_pyramids=True)

    assert len(results) == 1
    name, result = results[0]

    from ngff_zarr.multiscales import NgffMultiscales

    assert isinstance(result, NgffMultiscales)
    assert len(result.images) == 2

    # Check dimensions and shapes
    for i, img in enumerate(result.images):
        assert img.dims == ("y", "x"), (
            f"Level {i} should have dims ('y', 'x'), got {img.dims}"
        )
        expected_size = 256 // (2**i)
        expected_shape = (expected_size, expected_size)
        assert img.data.shape == expected_shape, (
            f"Level {i} should have shape {expected_shape}, got {img.data.shape}"
        )
