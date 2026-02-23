# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for write_store_to_zip public API"""

import zipfile
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr

from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr, write_store_to_zip
from ngff_zarr.hcs import to_hcs_zarr, HCSPlate, from_hcs_zarr
from ngff_zarr.rfc9_zip import read_ozx_version
from ngff_zarr.v04.zarr_metadata import (
    Plate,
    PlateColumn,
    PlateRow,
    PlateWell,
)

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

# RFC-9 requires zarr v3 (OME-Zarr 0.5)
pytestmark = pytest.mark.skipif(
    zarr_version_major < 3, reason="RFC-9 requires zarr-python >= 3.0.0"
)

# Output directory for test files
OUTPUT_DIR = Path(__file__).parent / "output"


def test_write_store_to_zip_with_path_string():
    """Test that users can call write_store_to_zip directly with a path string"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create a simple image and write it to a regular zarr store
    data = np.random.randint(0, 255, (2, 64, 64), dtype=np.uint8)
    image = to_ngff_image(
        data=data,
        dims=["c", "y", "x"],
        scale={"y": 0.5, "x": 0.5},
    )
    multiscales = to_multiscales(image)

    zarr_path = OUTPUT_DIR / "test_image.ome.zarr"
    ozx_path = OUTPUT_DIR / "test_image_converted.ozx"

    # Write to regular zarr store
    to_ngff_zarr(str(zarr_path), multiscales, version="0.5")

    # Now convert to .ozx using write_store_to_zip with path string
    write_store_to_zip(str(zarr_path), str(ozx_path), version="0.5")

    # Verify the .ozx file was created and is valid
    assert ozx_path.exists()
    assert zipfile.is_zipfile(ozx_path)

    # Check ZIP structure
    with zipfile.ZipFile(ozx_path, "r") as zf:
        files = zf.namelist()
        assert "zarr.json" in files
        assert files[0] == "zarr.json"  # Root zarr.json must be first

    # Verify version in ZIP comment
    version_from_zip = read_ozx_version(ozx_path)
    assert version_from_zip == "0.5"


def test_write_store_to_zip_hcs_plate():
    """Test converting an HCS plate to .ozx after writing"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create plate metadata
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    ]
    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Test Plate",
        field_count=1,
        version="0.5",
    )

    # Create and write plate to regular zarr
    plate_path = OUTPUT_DIR / "test_plate.ome.zarr"
    plate = HCSPlate(store=str(plate_path), plate_metadata=plate_metadata)
    to_hcs_zarr(plate, str(plate_path))

    # Add a well image
    from ngff_zarr.hcs import write_hcs_well_image

    data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data=data, dims=["c", "y", "x"])
    multiscales = to_multiscales(image)

    write_hcs_well_image(
        store=str(plate_path),
        multiscales=multiscales,
        plate_metadata=plate_metadata,
        row_name="A",
        column_name="1",
        field_index=0,
        version="0.5",
    )

    # Convert to .ozx
    ozx_path = OUTPUT_DIR / "test_plate_converted.ozx"
    write_store_to_zip(str(plate_path), str(ozx_path), version="0.5")

    # Verify the .ozx file
    assert ozx_path.exists()
    assert zipfile.is_zipfile(ozx_path)

    # Read it back
    plate_read = from_hcs_zarr(str(ozx_path))
    assert plate_read.name == "Test Plate"
    assert len(plate_read.wells) == 2

    # Verify we can access well data
    well = plate_read.get_well("A", "1")
    assert well is not None
    assert len(well.images) == 1


def test_write_store_to_zip_with_path_object():
    """Test that write_store_to_zip works with Path objects too"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create a simple image
    data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data=data, dims=["c", "y", "x"])
    multiscales = to_multiscales(image)

    zarr_path = OUTPUT_DIR / "test_path_obj.ome.zarr"
    ozx_path = OUTPUT_DIR / "test_path_obj.ozx"

    # Write to regular zarr store
    to_ngff_zarr(zarr_path, multiscales, version="0.5")

    # Convert using Path objects (not strings)
    write_store_to_zip(zarr_path, ozx_path, version="0.5")

    # Verify
    assert ozx_path.exists()
    assert zipfile.is_zipfile(ozx_path)
