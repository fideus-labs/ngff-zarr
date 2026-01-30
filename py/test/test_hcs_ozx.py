# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for HCS support with RFC-9: Zipped OME-Zarr (.ozx) format"""

import json
import zipfile
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr

from ngff_zarr import to_multiscales, to_ngff_image
from ngff_zarr.hcs import HCSPlateWriter, from_hcs_zarr
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


def test_write_hcs_well_image_to_ozx():
    """Test writing HCS well images to .ozx format using HCSPlateWriter"""
    # Create plate metadata
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    ]
    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Test Plate OZX",
        field_count=2,
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_hcs_plate.ozx"

    # Remove old file if exists
    if ozx_path.exists():
        ozx_path.unlink()

    # Create test images for wells
    data1 = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    image1 = to_ngff_image(
        data=data1,
        dims=["c", "y", "x"],
        scale={"y": 0.65, "x": 0.65},
        translation={"c": 0.0, "y": 0.0, "x": 0.0},
        name="Field 0",
    )
    multiscales1 = to_multiscales(image1, [2])

    data2 = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    image2 = to_ngff_image(
        data=data2,
        dims=["c", "y", "x"],
        scale={"y": 0.65, "x": 0.65},
        translation={"c": 0.0, "y": 0.0, "x": 0.0},
        name="Field 1",
    )
    multiscales2 = to_multiscales(image2, [2])

    # Write individual well images using HCSPlateWriter
    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        writer.write_well_image(
            multiscales=multiscales1,
            row_name="A",
            column_name="1",
            field_index=0,
            acquisition_id=0,
        )

        writer.write_well_image(
            multiscales=multiscales2,
            row_name="A",
            column_name="1",
            field_index=1,
            acquisition_id=0,
        )

    # Verify file was created and is a valid ZIP
    assert ozx_path.exists()
    assert zipfile.is_zipfile(ozx_path)

    # Verify ZIP structure
    with zipfile.ZipFile(ozx_path, "r") as zf:
        files = zf.namelist()

        # Root zarr.json should be first
        assert files[0] == "zarr.json"

        # Should contain well metadata
        well_zarr_json = "A/1/zarr.json"
        assert well_zarr_json in files

        # Check for OME-Zarr version comment
        version_from_zip = read_ozx_version(ozx_path)
        assert version_from_zip == "0.5"

    # Read back and verify
    plate_read = from_hcs_zarr(str(ozx_path))

    assert plate_read.name == "Test Plate OZX"
    assert len(plate_read.wells) == 2
    assert plate_read.field_count == 2

    # Get well and verify images
    well = plate_read.get_well("A", "1")
    assert well is not None
    assert len(well.images) == 2

    # Verify field images can be loaded
    field0 = well.get_image(0)
    assert field0 is not None
    assert len(field0.images) > 0  # Has scale levels

    field1 = well.get_image(1)
    assert field1 is not None
    assert len(field1.images) > 0


def test_write_hcs_ozx_requires_version_05():
    """Test that HCS .ozx files require OME-Zarr version 0.5"""
    columns = [PlateColumn(name="1")]
    rows = [PlateRow(name="A")]
    wells = [PlateWell(path="A/1", rowIndex=0, columnIndex=0)]
    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Test Plate",
        version="0.4",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_hcs_version_check.ozx"

    # Should raise error for version 0.4 with .ozx
    with pytest.raises(ValueError, match="RFC-9.*requires.*0.5"):
        with HCSPlateWriter(str(ozx_path), plate_metadata, version="0.4"):
            pass


def test_hcs_ozx_multiple_wells():
    """Test writing multiple wells to .ozx HCS plate"""
    # Create 2x2 plate
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ]
    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Multi-Well Plate",
        field_count=1,
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_hcs_multi_well.ozx"

    # Remove old file if exists
    if ozx_path.exists():
        ozx_path.unlink()

    # Write one image to each well using HCSPlateWriter
    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        for well in wells:
            row_name, col_name = well.path.split("/")
            data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
            image = to_ngff_image(
                data=data,
                dims=["c", "y", "x"],
                name=f"Well {well.path}",
            )
            multiscales = to_multiscales(image)

            writer.write_well_image(
                multiscales=multiscales,
                row_name=row_name,
                column_name=col_name,
                field_index=0,
            )

    # Verify all wells can be read
    plate_read = from_hcs_zarr(str(ozx_path))

    assert len(plate_read.wells) == 4

    for well_meta in plate_read.wells:
        row_name = plate_read.rows[well_meta.rowIndex].name
        col_name = plate_read.columns[well_meta.columnIndex].name
        well = plate_read.get_well(row_name, col_name)

        assert well is not None
        assert len(well.images) == 1

        # Verify image can be loaded
        image = well.get_image(0)
        assert image is not None


def test_hcs_ozx_zip_comment():
    """Test that HCS .ozx files have proper ZIP comment"""
    columns = [PlateColumn(name="1")]
    rows = [PlateRow(name="A")]
    wells = [PlateWell(path="A/1", rowIndex=0, columnIndex=0)]
    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_hcs_comment.ozx"

    if ozx_path.exists():
        ozx_path.unlink()

    # Create and write test image using HCSPlateWriter
    data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data=data, dims=["c", "y", "x"])
    multiscales = to_multiscales(image)

    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        writer.write_well_image(
            multiscales=multiscales,
            row_name="A",
            column_name="1",
            field_index=0,
        )

    # Check ZIP comment
    with zipfile.ZipFile(ozx_path, "r") as zf:
        comment = zf.comment.decode("utf-8").rstrip("\0")
        comment_dict = json.loads(comment)

        assert "ome" in comment_dict
        assert "version" in comment_dict["ome"]
        assert comment_dict["ome"]["version"] == "0.5"
