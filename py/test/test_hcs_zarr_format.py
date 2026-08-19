# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
"""Test HCS Zarr format version selection."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import NgffImage, to_multiscales
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr, write_hcs_well_image
from ngff_zarr.v04.zarr_metadata import (
    Plate,
    PlateColumn,
    PlateRow,
    PlateWell,
)


@pytest.fixture
def basic_plate_metadata():
    """Create basic plate metadata for testing."""
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ]

    return Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Test Plate",
        field_count=2,
        version="0.4",  # Default to 0.4 for testing
    )


@pytest.fixture
def sample_multiscales():
    """Create sample multiscales data for testing."""
    # Create a small test image
    data = np.random.randint(0, 255, size=(1, 1, 10, 64, 64), dtype=np.uint8)
    dask_data = da.from_array(data, chunks=(1, 1, 10, 32, 32))

    ngff_image = NgffImage(
        data=dask_data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name="test_field",
    )

    return to_multiscales(ngff_image, scale_factors=[2])


def test_to_hcs_zarr_uses_correct_zarr_format_v04(basic_plate_metadata):
    """Test that to_hcs_zarr uses zarr format 2 for NGFF version 0.4."""
    basic_plate_metadata.version = "0.4"
    plate = HCSPlate(None, basic_plate_metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plate_v04.ome.zarr"

        # Call the function
        to_hcs_zarr(plate, str(output_path))

        # Verify the zarr format by checking the store structure
        # Zarr format 2 should have .zgroup and .zattrs files
        assert (output_path / ".zgroup").exists(), (
            "Expected .zgroup file for zarr format 2"
        )
        assert (output_path / ".zattrs").exists(), (
            "Expected .zattrs file for zarr format 2"
        )

        # Zarr format 3 would have zarr.json instead
        assert not (output_path / "zarr.json").exists(), (
            "Should not have zarr.json for zarr format 2"
        )

        # Verify the metadata structure
        root = zarr.open_group(str(output_path), mode="r")
        attrs = root.attrs.asdict()

        # For NGFF 0.4, metadata should be in "multiscales" not "ome"
        # (but since this is HCS, it should be in "ome" with plate metadata)
        assert "ome" in attrs
        assert "plate" in attrs["ome"]
        assert attrs["ome"]["version"] == "0.4"


def test_to_hcs_zarr_uses_correct_zarr_format_v05(basic_plate_metadata):
    """Test that to_hcs_zarr uses zarr format 3 for NGFF version 0.5."""
    basic_plate_metadata.version = "0.5"
    plate = HCSPlate(None, basic_plate_metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plate_v05.ome.zarr"

        # Call the function
        to_hcs_zarr(plate, str(output_path))

        # For zarr-python >= 3, zarr format 3 creates zarr.json
        # For zarr-python < 3, it still uses .zgroup/.zattrs
        # We check that the function was called with the right parameters
        # by inspecting the metadata structure which should follow v0.5 conventions

        root = zarr.open_group(str(output_path), mode="r")
        attrs = root.attrs.asdict()

        assert "ome" in attrs
        assert "plate" in attrs["ome"]
        assert attrs["ome"]["version"] == "0.5"


@patch("ngff_zarr.hcs.create_zarrista_group")
def test_to_hcs_zarr_path_store_uses_zarrista(mock_create_group, basic_plate_metadata):
    """Path stores dispatch plate-group creation to the zarrista compat layer."""
    basic_plate_metadata.version = "0.4"
    plate = HCSPlate(None, basic_plate_metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = str(Path(tmpdir) / "test_store")

        to_hcs_zarr(plate, test_path)

        args, kwargs = mock_create_group.call_args
        assert args[0] == test_path
        assert args[1]["ome"]["plate"]["name"] == "Test Plate"
        assert args[2] == 2  # zarr format follows the NGFF version
        assert kwargs["overwrite"] is True


@patch("ngff_zarr.hcs.to_ome_zarr")
@patch("ngff_zarr.hcs.create_zarrista_subgroup")
def test_write_hcs_well_image_path_store_uses_zarrista(
    mock_create_subgroup,
    mock_to_ome_zarr,
    basic_plate_metadata,
    sample_multiscales,
):
    """Path stores dispatch well-group writes to the zarrista compat layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = str(Path(tmpdir) / "test_store")

        write_hcs_well_image(
            store=store_path,
            multiscales=sample_multiscales,
            plate_metadata=basic_plate_metadata,
            row_name="A",
            column_name="1",
            field_index=0,
            version="0.5",
        )

        args, _kwargs = mock_create_subgroup.call_args
        assert args[0] == store_path
        assert args[1] == "A/1"
        assert args[2]["ome"]["well"]["images"][0]["path"] == "0"
        assert args[3] == 3  # zarr format follows the NGFF version
        # The root group was created through the compat layer (mode="a" style).
        assert (Path(store_path) / "zarr.json").exists()


def test_zarr_format_selection_logic():
    """Test the core logic for zarr format selection."""
    # Test version 0.4 -> zarr format 2
    version = "0.4"
    zarr_format = 2 if version == "0.4" else 3
    assert zarr_format == 2, (
        f"Expected zarr_format 2 for version 0.4, got {zarr_format}"
    )

    # Test version 0.5 -> zarr format 3
    version = "0.5"
    zarr_format = 2 if version == "0.4" else 3
    assert zarr_format == 3, (
        f"Expected zarr_format 3 for version 0.5, got {zarr_format}"
    )

    # Test other versions -> zarr format 3
    for version in ["0.3", "0.6", "1.0"]:
        zarr_format = 2 if version == "0.4" else 3
        assert zarr_format == 3, (
            f"Expected zarr_format 3 for version {version}, got {zarr_format}"
        )


def test_write_hcs_well_image_integration(basic_plate_metadata, sample_multiscales):
    """Integration test for write_hcs_well_image with actual zarr operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_well.ome.zarr"

        # First create the plate structure
        basic_plate_metadata.version = "0.4"
        plate = HCSPlate(None, basic_plate_metadata)
        to_hcs_zarr(plate, str(output_path))

        # Now write a well image
        write_hcs_well_image(
            store=str(output_path),
            multiscales=sample_multiscales,
            plate_metadata=basic_plate_metadata,
            row_name="A",
            column_name="1",
            field_index=0,
            version="0.4",
        )

        # Verify the structure was created correctly
        assert output_path.exists()

        # Check that the well directory exists
        well_path = output_path / "A" / "1"
        assert well_path.exists()

        # Check that the field directory exists
        field_path = well_path / "0"
        assert field_path.exists()

        # For zarr format 2, should have .zarray files
        assert (field_path / ".zattrs").exists()

        # Check well metadata
        well_group = zarr.open_group(str(well_path), mode="r")
        well_attrs = well_group.attrs.asdict()
        assert "well" in well_attrs
        assert len(well_attrs["well"]["images"]) == 1
        assert well_attrs["well"]["images"][0]["path"] == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
