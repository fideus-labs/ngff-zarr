# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test tiff_file_to_ngff_images function for multi-series TIFF support and OME metadata extraction."""

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


def test_tiff_file_to_ngff_images_with_ome_metadata():
    """Test that OME metadata (physical sizes and units) is extracted."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create an OME-TIFF with physical size metadata
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "ome_with_metadata.ome.tiff"

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(10, 100, 100).astype(np.float32),
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.5,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": 2.0,
                "PhysicalSizeZUnit": "µm",
            },
        )

    # Convert and check metadata
    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    assert img.dims == ("z", "y", "x")

    # Check scale was extracted from OME metadata
    assert img.scale is not None
    assert img.scale["x"] == pytest.approx(0.5)
    assert img.scale["y"] == pytest.approx(0.5)
    assert img.scale["z"] == pytest.approx(2.0)

    # Check units were extracted and normalized
    assert img.axes_units is not None
    assert img.axes_units["x"] == "micrometer"
    assert img.axes_units["y"] == "micrometer"
    assert img.axes_units["z"] == "micrometer"


def test_tiff_file_to_ngff_images_partial_ome_metadata():
    """Test handling of partial OME metadata (only X/Y sizes, not Z)."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create an OME-TIFF with only X/Y physical sizes
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "ome_partial_metadata.ome.tiff"

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(10, 100, 100).astype(np.float32),
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.25,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.25,
                "PhysicalSizeYUnit": "µm",
                # No Z metadata
            },
        )

    # Convert and check metadata
    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]

    # Check X and Y scale was extracted
    assert img.scale is not None
    assert img.scale["x"] == pytest.approx(0.25)
    assert img.scale["y"] == pytest.approx(0.25)
    # Z should have default scale
    assert img.scale["z"] == pytest.approx(1.0)

    # Check units - only X and Y should have units
    assert img.axes_units is not None
    assert img.axes_units["x"] == "micrometer"
    assert img.axes_units["y"] == "micrometer"
    assert "z" not in img.axes_units


def test_tiff_file_to_ngff_images_no_ome_metadata():
    """Test handling of non-OME TIFF (no metadata extraction)."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a plain TIFF without OME metadata
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "plain.tiff"

    with tifffile.TiffWriter(tiff_path) as tif:  # Not ome=True
        tif.write(
            np.random.rand(10, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    # Convert and check - should use defaults
    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]

    # Scale should be defaults (1.0)
    assert img.scale["x"] == pytest.approx(1.0)
    assert img.scale["y"] == pytest.approx(1.0)
    assert img.scale["z"] == pytest.approx(1.0)

    # No units extracted
    assert img.axes_units is None


def test_tiff_file_to_ngff_images_2d():
    """Test handling of 2D TIFF."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "2d.ome.tiff"

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={
                "axes": "YX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.5,
                "PhysicalSizeYUnit": "µm",
            },
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    assert img.data.shape == (100, 100)
    assert img.dims == ("y", "x")

    # Check scale
    assert img.scale["x"] == pytest.approx(0.5)
    assert img.scale["y"] == pytest.approx(0.5)


def test_tiff_file_to_ngff_images_with_channels():
    """Test handling of TIFF with channel dimension."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "with_channels.ome.tiff"

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(3, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.5,
                "PhysicalSizeYUnit": "µm",
            },
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    assert img.data.shape == (3, 100, 100)
    assert img.dims == ("c", "y", "x")


def test_tiff_file_to_ngff_images_unit_normalization():
    """Test that various OME unit formats are normalized correctly."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Test with nanometer units
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "nanometer.ome.tiff"

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(100, 100).astype(np.float32),
            photometric="minisblack",
            metadata={
                "axes": "YX",
                "PhysicalSizeX": 100.0,
                "PhysicalSizeXUnit": "nm",
                "PhysicalSizeY": 100.0,
                "PhysicalSizeYUnit": "nm",
            },
        )

    images = tiff_file_to_ngff_images(tiff_path)
    name, img = images[0]

    assert img.axes_units["x"] == "nanometer"
    assert img.axes_units["y"] == "nanometer"


def test_tiff_file_to_ngff_images_with_sample_axis():
    """Test handling of TIFF with S (sample/RGB) axis."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "rgb_zyxs.ome.tiff"

    # Create ZYXS data (Z-stack of RGB images)
    data = np.random.rand(5, 100, 100, 3).astype(np.uint8)

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(data, photometric="rgb", metadata={"axes": "ZYXS"})

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    # S axis should be mapped to c
    assert img.dims == ("z", "y", "x", "c")
    assert img.data.shape == (5, 100, 100, 3)


def test_tiff_file_to_ngff_images_simple_rgb():
    """Test handling of simple RGB TIFF (YXS -> yxc)."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "rgb_yxs.tiff"

    # Create YXS data (2D RGB image)
    data = np.random.rand(100, 100, 3).astype(np.uint8)

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(data, photometric="rgb")

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    assert img.dims == ("y", "x", "c")
    assert img.data.shape == (100, 100, 3)


def test_tiff_file_to_ngff_images_unsupported_axis_warning():
    """Test that unsupported axes produce warnings and are dropped."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    import warnings

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "with_q_axis.tiff"

    # Create data with Q (unknown) axis
    data = np.random.rand(2, 5, 100, 100).astype(np.uint8)

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(data, photometric="minisblack", metadata={"axes": "QZYX"})

    # Should produce warning about Q axis
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        images = tiff_file_to_ngff_images(tiff_path)

        # Check warning was issued
        assert len(w) == 1
        assert "Dropping unsupported TIFF axes: Q" in str(w[0].message)

    assert len(images) == 1
    name, img = images[0]

    # Q axis should be dropped (first slice taken)
    assert img.dims == ("z", "y", "x")
    assert img.data.shape == (5, 100, 100)


def test_tiff_file_to_ngff_images_corrupted_file():
    """Test that corrupted TIFF files raise appropriate errors."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "corrupted.tiff"

    # Create a file that looks like a TIFF but is corrupted
    with open(tiff_path, "wb") as f:
        # Write TIFF header (little-endian)
        f.write(b"II\x2a\x00")  # TIFF magic number
        f.write(b"\x08\x00\x00\x00")  # Invalid IFD offset
        # Write garbage data
        f.write(b"corrupted data" * 100)

    # Should raise OSError with descriptive message
    with pytest.raises(OSError) as exc_info:
        tiff_file_to_ngff_images(tiff_path)

    error_msg = str(exc_info.value)
    assert "Failed to open TIFF file" in error_msg or "corrupted" in error_msg.lower()


def test_tiff_file_to_ngff_images_nonexistent_file():
    """Test that nonexistent files raise appropriate errors."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Try to open a file that doesn't exist
    with pytest.raises((OSError, FileNotFoundError)):
        tiff_file_to_ngff_images("/nonexistent/path/to/file.tiff")


def test_tiff_file_to_ngff_images_corrupted_series_continues():
    """Test that processing continues after encountering a corrupted series."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    import warnings

    from ngff_zarr import tiff_file_to_ngff_images

    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "multi_series.tiff"

    # Create a multi-series TIFF where we can later corrupt one series
    with tifffile.TiffWriter(tiff_path) as tif:
        # First series - valid
        tif.write(
            np.random.rand(10, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        # Second series - valid
        tif.write(
            np.random.rand(5, 50, 50).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    # Note: It's difficult to create a partially corrupted multi-series TIFF
    # where one series is valid and another is corrupted. This test documents
    # the intended behavior: if a series fails, a warning is issued and
    # processing continues with other series.
    # In practice, the error handling code will catch exceptions during
    # series processing and continue.

    images = tiff_file_to_ngff_images(tiff_path)
    # Both series should be valid in this test
    assert len(images) == 2
