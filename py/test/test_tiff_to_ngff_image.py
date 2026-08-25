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


def test_tiff_file_to_ngff_images_with_ome_translation():
    """Test that OME translation metadata is extracted."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import NgffMultiscales, tiff_file_to_ngff_images
    from ngff_zarr.tiff_to_ngff_image import _convert_unit_value

    # Create an OME-TIFF with translation metadata
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "ome_with_translation.ome.tiff"

    pixel_size = 1
    default_unit = "micrometer"

    size_x, size_y, size_z = 100, 100, 10
    position = {"x": 1.1, "y": 2.2, "z": 3.3}
    ome_unit = "µm"
    nplanes = size_z

    expected_position = {
        dim: _convert_unit_value(position[dim], ome_unit, default_unit) for dim in "xyz"
    }

    metadata_xyz = {
        "DimensionOrder": "XYZCT",
        "PhysicalSizeX": pixel_size,
        "PhysicalSizeXUnit": default_unit,
        "PhysicalSizeY": pixel_size,
        "PhysicalSizeYUnit": default_unit,
        "PhysicalSizeZ": pixel_size,
        "PhysicalSizeZUnit": default_unit,
        "Plane": {
            "PositionX": [position["x"]] * nplanes,
            "PositionXUnit": [ome_unit] * nplanes,
            "PositionY": [position["y"]] * nplanes,
            "PositionYUnit": [ome_unit] * nplanes,
            "PositionZ": [position["z"]] * nplanes,
            "PositionZUnit": [ome_unit] * nplanes,
        },
    }

    metadata_xy = {
        "DimensionOrder": "XYZCT",
        "PhysicalSizeX": pixel_size,
        "PhysicalSizeXUnit": default_unit,
        "PhysicalSizeY": pixel_size,
        "PhysicalSizeYUnit": default_unit,
        "Plane": {
            "PositionX": [position["x"]] * nplanes,
            "PositionXUnit": [ome_unit] * nplanes,
            "PositionY": [position["y"]] * nplanes,
            "PositionYUnit": [ome_unit] * nplanes,
        },
    }

    for metadata, expected_dims in zip([metadata_xy, metadata_xyz], ["xy", "xyz"]):
        with tifffile.TiffWriter(tiff_path, ome=True) as tif:
            if "z" in expected_dims:
                data = np.random.rand(size_z, size_y, size_x).astype(np.float32)
                data_downscaled = np.random.rand(
                    size_z // 2, size_y // 2, size_x // 2
                ).astype(np.float32)
            else:
                data = np.random.rand(size_y, size_x).astype(np.float32)
                data_downscaled = np.random.rand(size_y // 2, size_x // 2).astype(
                    np.float32
                )
            tif.write(
                data,
                photometric="minisblack",
                metadata=metadata,
                subifds=1,
            )
            tif.write(
                data_downscaled,
                photometric="minisblack",
                metadata=metadata,
                subfiletype=1,
            )

        # Convert and check metadata
        for reuse_existing_pyramids in [False, True]:
            images = tiff_file_to_ngff_images(
                tiff_path, reuse_existing_pyramids=reuse_existing_pyramids
            )
            assert len(images) == 1

            name, img = images[0]
            if isinstance(img, NgffMultiscales):
                img = img.images[0]  # Get the first NgffImage from the multiscales

            # Check translation was extracted from OME metadata
            assert img.translation is not None
            for dim in expected_dims:
                assert _convert_unit_value(
                    img.translation[dim], img.axes_units[dim], default_unit
                ) == pytest.approx(expected_position[dim])


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

        # Check the expected warning was issued. Filter for the relevant
        # warning rather than asserting on the total count, since unrelated
        # libraries (e.g. tifffile/numpy) may emit additional warnings on
        # some platforms.
        dropped_warnings = [
            warning
            for warning in w
            if "Dropping unsupported TIFF axes: Q" in str(warning.message)
        ]
        assert len(dropped_warnings) == 1

    assert len(images) == 1
    name, img = images[0]

    # Q axis should be dropped (first slice taken)
    assert img.dims == ("z", "y", "x")
    assert img.data.shape == (5, 100, 100)


def test_tiff_file_to_ngff_images_with_channel_names():
    """Test that channel names are extracted from OME-XML metadata."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create an OME-TIFF with channel names
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "with_channel_names.ome.tiff"

    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(3, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "Channel": {"Name": ["DAPI", "GFP", "RFP"]},
            },
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    assert img.data.shape == (3, 100, 100)
    assert img.dims == ("c", "y", "x")

    # Check that channel names were extracted
    assert img.channel_names is not None
    assert len(img.channel_names) == 3
    assert img.channel_names == ["DAPI", "GFP", "RFP"]


def test_tiff_file_to_ngff_images_with_partial_channel_names():
    """Test handling of OME-TIFF with some channels lacking names."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create an OME-TIFF where some channels have names and others don't
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "partial_channel_names.ome.tiff"

    # Note: tifffile's ome=True mode handles Channel metadata differently
    # We'll create a more explicit OME-XML for this test
    with tifffile.TiffWriter(tiff_path, ome=True) as tif:
        tif.write(
            np.random.rand(3, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "Channel": {"Name": ["DAPI", "", "RFP"]},
            },
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    # Channel names should be present even if some are empty
    assert img.channel_names is not None
    assert len(img.channel_names) == 3
    assert img.channel_names[0] == "DAPI"
    assert img.channel_names[1] == ""  # Empty string for unnamed channel
    assert img.channel_names[2] == "RFP"


def test_tiff_file_to_ngff_images_without_channel_names():
    """Test that non-OME TIFF or OME-TIFF without channel names has None."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    # Create a plain TIFF without OME metadata
    tmpdir = Path(tempfile.mkdtemp())
    tiff_path = tmpdir / "no_channel_names.tiff"

    with tifffile.TiffWriter(tiff_path) as tif:  # Not ome=True
        tif.write(
            np.random.rand(3, 100, 100).astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    name, img = images[0]
    # Should have None for channel names
    assert img.channel_names is None


def test_tiff_sample_axis_sets_rgb_channel_colors(tmp_path):
    """Test that TIFF images with S (sample) axis get RGB channel_colors set."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tiff_path = tmp_path / "rgb_yxs.tiff"

    # Create a 2D RGB image with YXS axes
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(data, photometric="rgb")

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    _, img = images[0]
    # S axis should be mapped to c with RGB channel_colors assigned
    assert img.dims == ("y", "x", "c")
    assert img.channel_colors is not None
    assert img.channel_colors == ["FF0000", "00FF00", "0000FF"]


def test_tiff_sample_axis_rgba_sets_four_channel_colors(tmp_path):
    """Test that a 4-sample (RGBA) TIFF gets 4 RGB+alpha channel_colors."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tiff_path = tmp_path / "rgba_yxs.tiff"

    # Create a 2D RGBA image with YXS axes (4 samples)
    data = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(data, photometric="rgb", extrasamples=["unassalpha"])

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    _, img = images[0]
    assert img.channel_colors is not None
    assert img.channel_colors == ["FF0000", "00FF00", "0000FF", "FFFFFF"]


def test_ome_tiff_channel_colors_from_xml(tmp_path):
    """Test that channel colors are extracted from OME-XML Color attributes."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tiff_path = tmp_path / "colored.ome.tiff"

    # Create OME-TIFF with channel color metadata
    # OME color integers (RGBA format as signed int32):
    #   Red   (FF FF 00 00): int32 = -65536       -> hex "FF0000"
    #   Green (FF 00 FF 00): int32 = -16711936     -> hex "00FF00"
    #   Blue  (FF 00 00 FF): int32 = -16776961     -> hex "0000FF"
    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="test">
    <Pixels ID="Pixels:0" Type="uint8" SizeX="100" SizeY="100" SizeC="3"
            SizeZ="1" SizeT="1" DimensionOrder="XYZCT"
            PhysicalSizeX="0.5" PhysicalSizeXUnit="um"
            PhysicalSizeY="0.5" PhysicalSizeYUnit="um">
      <Channel ID="Channel:0:0" Name="Red" Color="-65536"/>
      <Channel ID="Channel:0:1" Name="Green" Color="-16711936"/>
      <Channel ID="Channel:0:2" Name="Blue" Color="-16776961"/>
      <TiffData/>
    </Pixels>
  </Image>
</OME>"""

    data = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)

    with tifffile.TiffWriter(tiff_path, ome=False) as tif:
        tif.write(
            data,
            photometric="minisblack",
            description=ome_xml,
            metadata=None,
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    _, img = images[0]
    assert img.channel_colors is not None
    assert img.channel_colors[0] == "FF0000"
    assert img.channel_colors[1] == "00FF00"
    assert img.channel_colors[2] == "0000FF"


def test_non_rgb_multichannel_tiff_no_channel_colors(tmp_path):
    """Test that multi-channel TIFFs without S axis have no channel_colors."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tiff_path = tmp_path / "multichannel.tiff"

    # 3-channel fluorescence image with C axis (not S)
    data = np.random.randint(0, 65535, (3, 100, 100), dtype=np.uint16)

    with tifffile.TiffWriter(tiff_path) as tif:
        tif.write(data, photometric="minisblack", metadata={"axes": "CYX"})

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    _, img = images[0]
    # No S axis and no OME-XML colors -> channel_colors should be None
    assert img.channel_colors is None


def test_ome_tiff_rgb_s_axis_overrides_xml_colors(tmp_path):
    """Test that RGB images with S axis use canonical RGB colors, overriding OME-XML colors.

    This is a regression test for the issue where j2k OME-TIFF files with RGB data
    (S axis) but incorrect channel colors in OME-XML metadata would display with
    wrong colors. The S axis indicates RGB/RGBA data which must always use the
    canonical RGB color mapping, regardless of what OME-XML says.
    """
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tiff_path = tmp_path / "rgb_with_wrong_ome_colors.ome.tiff"

    # Create OME-TIFF with S axis (RGB data) but wrong channel colors in OME-XML.
    # This simulates what can happen with j2k-compressed OME-TIFF files where
    # the OME-XML might have been auto-generated with incorrect color assignments.
    # OME color integers (RGBA format as signed int32):
    #   Cyan    (FF 00 FF FF): int32 = -16711681
    #   Magenta (FF FF 00 FF): int32 = -65281
    #   Yellow  (FF FF FF 00): int32 = -256
    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="test_rgb">
    <Pixels ID="Pixels:0" Type="uint8" SizeX="100" SizeY="100" SizeC="3"
            SizeZ="1" SizeT="1" DimensionOrder="XYZCT">
      <Channel ID="Channel:0:0" Name="Sample 1" Color="-16711681"/>
      <Channel ID="Channel:0:1" Name="Sample 2" Color="-65281"/>
      <Channel ID="Channel:0:2" Name="Sample 3" Color="-256"/>
      <TiffData/>
    </Pixels>
  </Image>
</OME>"""

    # Create RGB data with S axis (YXS ordering - standard RGB interleaved)
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    with tifffile.TiffWriter(tiff_path, ome=False) as tif:
        tif.write(
            data,
            photometric="rgb",  # RGB photometric interpretation
            description=ome_xml,
            metadata={"axes": "YXS"},  # S axis indicates RGB samples
        )

    images = tiff_file_to_ngff_images(tiff_path)
    assert len(images) == 1

    _, img = images[0]

    # The S axis should force RGB colors, overriding the wrong OME-XML colors
    assert img.channel_colors is not None, "RGB images should have channel_colors"
    assert img.channel_colors == ["FF0000", "00FF00", "0000FF"], (
        f"RGB images with S axis must use canonical RGB colors, "
        f"got {img.channel_colors}"
    )


def test_tiff_file_to_ngff_images_corrupted_file(tmp_path):
    """A corrupted TIFF should raise an actionable OSError (gh-issue-343)."""
    try:
        import tifffile  # noqa: F401
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    tiff_path = tmp_path / "corrupted.tiff"

    # A file that looks like a TIFF (valid magic) but has an invalid IFD.
    with tiff_path.open("wb") as f:
        f.write(b"II\x2a\x00")  # little-endian TIFF magic
        f.write(b"\x08\x00\x00\x00")  # invalid IFD offset
        f.write(b"corrupted data" * 100)

    with pytest.raises(OSError) as exc_info:
        tiff_file_to_ngff_images(tiff_path)

    error_msg = str(exc_info.value)
    assert "Failed to open TIFF file" in error_msg
    assert "corrupted" in error_msg.lower()


def test_tiff_file_to_ngff_images_nonexistent_file(tmp_path):
    """A missing path keeps its specific FileNotFoundError, not relabeled.

    Path/permission errors should propagate unchanged rather than being
    re-raised as a generic "file may be corrupted" error.
    """
    try:
        import tifffile  # noqa: F401
    except ImportError:
        pytest.skip("tifffile not available")

    from ngff_zarr import tiff_file_to_ngff_images

    missing_path = tmp_path / "does_not_exist.tiff"
    with pytest.raises(FileNotFoundError) as exc_info:
        tiff_file_to_ngff_images(missing_path)

    # Should not be relabeled with the corruption message.
    assert "may be" not in str(exc_info.value).lower()
