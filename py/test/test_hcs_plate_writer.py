# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for HCSPlateWriter context manager with .ozx support"""

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
    zarr_version_major < 3,
    reason="HCSPlateWriter with .ozx requires zarr-python >= 3.0.0",
)

# Output directory for test files
OUTPUT_DIR = Path(__file__).parent / "output"


def test_hcs_plate_writer_ozx_basic():
    """Test basic HCSPlateWriter with .ozx format"""
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
        name="Test Plate Writer",
        field_count=2,
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_plate_writer.ozx"

    # Remove old file if exists
    if ozx_path.exists():
        ozx_path.unlink()

    # Create test images
    data1 = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    image1 = to_ngff_image(
        data=data1,
        dims=["c", "y", "x"],
        scale={"y": 0.65, "x": 0.65},
    )
    multiscales1 = to_multiscales(image1, [2])

    data2 = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    image2 = to_ngff_image(
        data=data2,
        dims=["c", "y", "x"],
        scale={"y": 0.65, "x": 0.65},
    )
    multiscales2 = to_multiscales(image2, [2])

    # Use context manager to write wells
    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        writer.write_well_image(
            multiscales=multiscales1,
            row_name="A",
            column_name="1",
            field_index=0,
        )
        writer.write_well_image(
            multiscales=multiscales2,
            row_name="A",
            column_name="2",
            field_index=0,
        )

    # Verify file was created and is a valid ZIP
    assert ozx_path.exists()
    assert zipfile.is_zipfile(ozx_path)

    # Verify ZIP structure
    with zipfile.ZipFile(ozx_path, "r") as zf:
        files = zf.namelist()
        assert files[0] == "zarr.json"

        # Check for version
        version_from_zip = read_ozx_version(ozx_path)
        assert version_from_zip == "0.5"

    # Read back and verify
    plate_read = from_hcs_zarr(str(ozx_path))
    assert plate_read.name == "Test Plate Writer"
    assert len(plate_read.wells) == 2

    # Verify wells
    well1 = plate_read.get_well("A", "1")
    assert well1 is not None
    assert len(well1.images) == 1

    well2 = plate_read.get_well("A", "2")
    assert well2 is not None
    assert len(well2.images) == 1


def test_hcs_plate_writer_regular_zarr():
    """Test HCSPlateWriter with regular .ome.zarr directory"""
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
    zarr_path = OUTPUT_DIR / "test_plate_writer_regular.ome.zarr"

    # Remove old directory if exists
    if zarr_path.exists():
        import shutil

        shutil.rmtree(zarr_path)

    # Create test image
    data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data=data, dims=["c", "y", "x"])
    multiscales = to_multiscales(image)

    # Use context manager
    with HCSPlateWriter(str(zarr_path), plate_metadata) as writer:
        writer.write_well_image(
            multiscales=multiscales,
            row_name="A",
            column_name="1",
            field_index=0,
        )

    # Verify directory was created
    assert zarr_path.exists()
    assert zarr_path.is_dir()

    # Read back
    plate_read = from_hcs_zarr(str(zarr_path))
    well = plate_read.get_well("A", "1")
    assert well is not None


def test_hcs_plate_writer_multiple_fields():
    """Test writing multiple fields to same well"""
    columns = [PlateColumn(name="1")]
    rows = [PlateRow(name="A")]
    wells = [PlateWell(path="A/1", rowIndex=0, columnIndex=0)]
    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        field_count=3,
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_multiple_fields.ozx"

    if ozx_path.exists():
        ozx_path.unlink()

    # Create context manager and write multiple fields
    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        for field_idx in range(3):
            data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
            image = to_ngff_image(data=data, dims=["c", "y", "x"])
            multiscales = to_multiscales(image)

            writer.write_well_image(
                multiscales=multiscales,
                row_name="A",
                column_name="1",
                field_index=field_idx,
            )

    # Verify
    plate_read = from_hcs_zarr(str(ozx_path))
    well = plate_read.get_well("A", "1")
    assert well is not None
    assert len(well.images) == 3


def test_hcs_plate_writer_parallel_writes():
    """Test parallel writing using ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor

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
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_parallel.ozx"

    if ozx_path.exists():
        ozx_path.unlink()

    # Prepare well data
    well_data = []
    for well in wells:
        row_name, col_name = well.path.split("/")
        data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
        image = to_ngff_image(data=data, dims=["c", "y", "x"])
        multiscales = to_multiscales(image)
        well_data.append((multiscales, row_name, col_name))

    # Write in parallel
    def write_well(args):
        writer, multiscales, row, col = args
        writer.write_well_image(
            multiscales=multiscales,
            row_name=row,
            column_name=col,
            field_index=0,
        )

    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(write_well, [(writer, *wd) for wd in well_data])

    # Verify all wells were written
    plate_read = from_hcs_zarr(str(ozx_path))
    assert len(plate_read.wells) == 4

    for well_meta in plate_read.wells:
        row_name = plate_read.rows[well_meta.rowIndex].name
        col_name = plate_read.columns[well_meta.columnIndex].name
        well = plate_read.get_well(row_name, col_name)
        assert well is not None
        assert len(well.images) == 1


def test_hcs_plate_writer_version_validation():
    """Test that .ozx requires version 0.5"""
    plate_metadata = Plate(
        columns=[PlateColumn(name="1")],
        rows=[PlateRow(name="A")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        version="0.4",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_version_error.ozx"

    # Should raise error for version 0.4 with .ozx
    with pytest.raises(ValueError, match="RFC-9.*requires.*0.5"):
        with HCSPlateWriter(str(ozx_path), plate_metadata, version="0.4"):
            pass


def test_hcs_plate_writer_exception_handling():
    """Test that .ozx is not created if exception occurs"""
    plate_metadata = Plate(
        columns=[PlateColumn(name="1")],
        rows=[PlateRow(name="A")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_exception.ozx"

    if ozx_path.exists():
        ozx_path.unlink()

    # Trigger exception inside context
    try:
        with HCSPlateWriter(str(ozx_path), plate_metadata) as _writer:
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass

    # .ozx file should not be created due to exception
    assert not ozx_path.exists()


def test_hcs_plate_writer_default_version():
    """Test that default version is 0.5"""
    plate_metadata = Plate(
        columns=[PlateColumn(name="1")],
        rows=[PlateRow(name="A")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_default_version.ozx"

    if ozx_path.exists():
        ozx_path.unlink()

    # Don't specify version - should default to 0.5
    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
        image = to_ngff_image(data=data, dims=["c", "y", "x"])
        multiscales = to_multiscales(image)

        writer.write_well_image(
            multiscales=multiscales,
            row_name="A",
            column_name="1",
            field_index=0,
        )

    # Verify version is 0.5
    version = read_ozx_version(ozx_path)
    assert version == "0.5"


def test_hcs_plate_writer_large_plate():
    """Test writing a larger plate with multiple wells and fields"""
    # Create 3x3 plate
    columns = [PlateColumn(name=str(i)) for i in range(1, 4)]
    rows = [PlateRow(name=chr(65 + i)) for i in range(3)]  # A, B, C
    wells = []
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(columns):
            wells.append(
                PlateWell(
                    path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx
                )
            )

    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Large Test Plate",
        field_count=2,
        version="0.5",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_large_plate.ozx"

    if ozx_path.exists():
        ozx_path.unlink()

    # Write all wells with 2 fields each
    with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
        for well in wells:
            row_name, col_name = well.path.split("/")
            for field_idx in range(2):
                data = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
                image = to_ngff_image(data=data, dims=["c", "y", "x"])
                multiscales = to_multiscales(image)

                writer.write_well_image(
                    multiscales=multiscales,
                    row_name=row_name,
                    column_name=col_name,
                    field_index=field_idx,
                )

    # Verify all wells and fields
    plate_read = from_hcs_zarr(str(ozx_path))
    assert plate_read.name == "Large Test Plate"
    assert len(plate_read.wells) == 9  # 3x3

    # Check each well has 2 fields
    for well_meta in plate_read.wells:
        row_name = plate_read.rows[well_meta.rowIndex].name
        col_name = plate_read.columns[well_meta.columnIndex].name
        well = plate_read.get_well(row_name, col_name)
        assert well is not None
        assert len(well.images) == 2
