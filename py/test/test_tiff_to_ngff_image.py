# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test tiff_file_to_ngff_images function for multi-series TIFF support."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_tiff_file_to_ngff_images_all_series():
    """Test converting all series from a multi-series TIFF."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a multi-series TIFF
    # Note: Using random data is acceptable here as we're testing structure, not values
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "multi_series.ome.tiff"

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(
            np.random.rand(10, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        tif.write(
            np.random.rand(5, 50, 50).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    # Test converting all series
    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 2

    # Check first series
    name0, img0 = images[0]
    assert img0.data.shape == (10, 100, 100)
    assert img0.dims == ("z", "y", "x")

    # Check second series
    name1, img1 = images[1]
    assert img1.data.shape == (5, 50, 50)
    assert img1.dims == ("z", "y", "x")


def test_tiff_file_to_ngff_images_single_series_by_index():
    """Test converting a single series by index from a multi-series TIFF."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a multi-series TIFF
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "multi_series.ome.tiff"

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(
            np.random.rand(10, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        tif.write(
            np.random.rand(5, 50, 50).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    # Test converting series 0
    images = tiff_file_to_ngff_images(tiff_path, series=0)
    assert len(images) == 1
    name, img = images[0]
    assert img.data.shape == (10, 100, 100)

    # Test converting series 1
    images = tiff_file_to_ngff_images(tiff_path, series=1)
    assert len(images) == 1
    name, img = images[0]
    assert img.data.shape == (5, 50, 50)


def test_tiff_file_to_ngff_images_invalid_index():
    """Test that invalid series index raises IndexError."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a single-series TIFF
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "single_series.ome.tiff"

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(
            np.random.rand(10, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    # Test invalid index
    with pytest.raises(IndexError, match="Series index 5 is out of bounds"):
        tiff_file_to_ngff_images(tiff_path, series=5)


def test_tiff_file_to_ngff_images_series_all():
    """Test converting all series using 'all' keyword."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a multi-series TIFF
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "multi_series.ome.tiff"

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(
            np.random.rand(10, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        tif.write(
            np.random.rand(5, 50, 50).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    # Test with 'all' keyword
    images = tiff_file_to_ngff_images(tiff_path, series="all")
    assert len(images) == 2
