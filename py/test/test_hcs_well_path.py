# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test HCS well path support in from_ngff_zarr."""

from pathlib import Path

import pytest
from ngff_zarr.from_ngff_zarr import from_ngff_zarr
from ngff_zarr.hcs import from_hcs_zarr
from ngff_zarr.parse_metadata import _parse_hcs_path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data" / "input"


@pytest.fixture
def hcs_data_path(test_data_dir):
    """Return path to HCS test data."""
    return test_data_dir / "hcs.ome.zarr"


def test_parse_hcs_path():
    """Test HCS path parsing function."""
    # Test basic plate path
    store, subpath = _parse_hcs_path("plate.zarr")
    assert store == "plate.zarr", f"Expected store 'plate.zarr', got '{store}'"
    assert subpath is None, "Expected no subpath for plate root"

    # Test well path
    store, subpath = _parse_hcs_path("plate.zarr/A/1")
    assert store == "plate.zarr", f"Expected store 'plate.zarr', got '{store}'"
    assert subpath == "A/1", f"Expected subpath 'A/1', got '{subpath}'"

    # Test well with field
    store, subpath = _parse_hcs_path("plate.zarr/A/1/0")
    assert store == "plate.zarr", f"Expected store 'plate.zarr', got '{store}'"
    assert subpath == "A/1/0", f"Expected subpath 'A/1/0', got '{subpath}'"

    # Test .ome.zarr extension
    store, subpath = _parse_hcs_path("/path/to/plate.ome.zarr/B/2/0")
    assert store == "/path/to/plate.ome.zarr", (
        f"Expected store '/path/to/plate.ome.zarr', got '{store}'"
    )
    assert subpath == "B/2/0", f"Expected subpath 'B/2/0', got '{subpath}'"

    # Edge case: multiple .zarr segments — deepest (last) store should be selected
    store, subpath = _parse_hcs_path("/a.zarr/b/plate.zarr/A/1/0")
    assert store == "/a.zarr/b/plate.zarr", (
        f"Expected deepest store '/a.zarr/b/plate.zarr', got '{store}'"
    )
    assert subpath == "A/1/0", f"Expected subpath 'A/1/0', got '{subpath}'"

    # Edge case: trailing slash on subpath should be stripped
    store, subpath = _parse_hcs_path("plate.zarr/A/1/0/")
    assert store == "plate.zarr", f"Expected store 'plate.zarr', got '{store}'"
    assert subpath == "A/1/0", f"Expected stripped subpath 'A/1/0', got '{subpath}'"

    # Edge case: no zarr extension — returned unchanged
    store, subpath = _parse_hcs_path("no-extension/path")
    assert store == "no-extension/path", f"Expected unchanged path, got '{store}'"
    assert subpath is None, "Expected no subpath for non-zarr path"


def test_from_ngff_zarr_hcs_plate_error(hcs_data_path):
    """Test that loading HCS plate root raises helpful error."""
    if not hcs_data_path.exists():
        pytest.skip("HCS test data not available")

    # Loading the plate root should raise an error with helpful message
    with pytest.raises(ValueError) as excinfo:
        from_ngff_zarr(str(hcs_data_path))

    error_msg = str(excinfo.value)
    assert "HCS" in error_msg or "plate" in error_msg.lower()
    # Should mention how many wells exist
    assert "wells" in error_msg.lower()


def test_from_ngff_zarr_hcs_well_image(hcs_data_path):
    """Test loading HCS well image directly via from_ngff_zarr."""
    if not hcs_data_path.exists():
        pytest.skip("HCS test data not available")

    # First, get the plate to find available wells
    plate = from_hcs_zarr(str(hcs_data_path))

    if not plate.metadata.wells:
        pytest.skip("No wells in test HCS data")

    # Get the first well
    first_well = plate.metadata.wells[0]
    well_path = first_well.path

    # Load the first field of the first well using sub-path syntax
    well_image_path = f"{hcs_data_path}/{well_path}/0"

    # This should work without error
    multiscales = from_ngff_zarr(str(well_image_path))

    # Validate the result
    assert multiscales is not None
    assert len(multiscales.images) > 0
    assert multiscales.images[0].data is not None

    # Compare with loading via HCS API
    well = plate.get_well(
        plate.metadata.rows[first_well.rowIndex].name,
        plate.metadata.columns[first_well.columnIndex].name,
    )
    image_via_hcs = well.get_image(0)

    # Both should have the same shape at full resolution
    assert multiscales.images[0].data.shape == image_via_hcs.images[0].data.shape


def test_from_ngff_zarr_hcs_invalid_subpath(hcs_data_path):
    """Test that invalid HCS sub-path raises appropriate error."""
    if not hcs_data_path.exists():
        pytest.skip("HCS test data not available")

    # Try to load a non-existent well
    invalid_path = f"{hcs_data_path}/Z/99/0"

    with pytest.raises(ValueError) as excinfo:
        from_ngff_zarr(str(invalid_path))

    error_msg = str(excinfo.value)
    assert "not found" in error_msg.lower() or "sub-path" in error_msg.lower()
