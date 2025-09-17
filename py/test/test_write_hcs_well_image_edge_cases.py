# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
"""Test for edge cases in write_hcs_well_image function."""

import pytest
import tempfile
from pathlib import Path
import numpy as np

import ngff_zarr as nz
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr, write_hcs_well_image
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell


def test_write_hcs_well_image_edge_cases():
    """Test write_hcs_well_image with edge cases that might cause index errors.

    This test covers scenarios where rowIndex/columnIndex in PlateWell metadata
    might not align with expectations, and validates that the function properly
    handles these cases without index out of range errors.
    """
    # Create a plate with gaps in the row/column layout
    columns = [
        PlateColumn(name="1"),
        PlateColumn(name="3"),  # Skip column "2"
        PlateColumn(name="5"),  # Skip column "4"
    ]
    rows = [
        PlateRow(name="A"),
        PlateRow(name="C"),  # Skip row "B"
        PlateRow(name="E"),  # Skip row "D"
    ]

    # Create wells where the logical rowIndex/columnIndex might not
    # correspond to array positions
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/3", rowIndex=0, columnIndex=2),
        PlateWell(path="C/1", rowIndex=2, columnIndex=0),
        PlateWell(path="C/5", rowIndex=2, columnIndex=4),
        PlateWell(path="E/3", rowIndex=4, columnIndex=2),
    ]

    plate_metadata = Plate(
        columns=columns, rows=rows, wells=wells, name="Sparse Test Plate", field_count=2
    )

    # Create sample image data
    data = np.random.randint(0, 255, size=(1, 1, 5, 128, 128), dtype=np.uint8)
    ngff_image = nz.NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = nz.to_multiscales(ngff_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "edge_case_plate.ome.zarr"

        # Create plate structure
        hcs_plate = HCSPlate(None, plate_metadata)
        to_hcs_zarr(hcs_plate, str(output_path))

        # Test writing to each well - should not raise any errors
        test_cases = [
            ("A", "1"),  # Normal case
            ("A", "3"),  # Gap in columns
            ("C", "1"),  # Gap in rows
            ("C", "5"),  # Gaps in both
            ("E", "3"),  # Later row with middle column
        ]

        for row_name, column_name in test_cases:
            # This should work without errors
            write_hcs_well_image(
                store=str(output_path),
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name=row_name,
                column_name=column_name,
                field_index=0,
            )

            # Verify the data was written correctly
            well_path = output_path / row_name / column_name / "0"
            assert (
                well_path.exists()
            ), f"Well {row_name}/{column_name}/0 was not created"
            assert (
                well_path / ".zattrs"
            ).exists(), f"Well {row_name}/{column_name}/0 metadata missing"


def test_write_hcs_well_image_invalid_indices():
    """Test that write_hcs_well_image properly handles invalid row/column names."""
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ]

    plate_metadata = Plate(
        columns=columns, rows=rows, wells=wells, name="Test Plate", field_count=1
    )

    # Create sample image data
    data = np.random.randint(0, 255, size=(1, 1, 5, 128, 128), dtype=np.uint8)
    ngff_image = nz.NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = nz.to_multiscales(ngff_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plate.ome.zarr"

        # Create plate structure
        hcs_plate = HCSPlate(None, plate_metadata)
        to_hcs_zarr(hcs_plate, str(output_path))

        # Test invalid row name
        with pytest.raises(ValueError, match="Row 'Z' not found in plate metadata"):
            write_hcs_well_image(
                store=str(output_path),
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="Z",  # Invalid row
                column_name="1",
                field_index=0,
            )

        # Test invalid column name
        with pytest.raises(ValueError, match="Column '9' not found in plate metadata"):
            write_hcs_well_image(
                store=str(output_path),
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="A",
                column_name="9",  # Invalid column
                field_index=0,
            )

        # Test valid row/column but no well defined
        with pytest.raises(ValueError, match="Well 'A/3' not found in plate metadata"):
            # Add column "3" but no well for it
            plate_metadata.columns.append(PlateColumn(name="3"))
            write_hcs_well_image(
                store=str(output_path),
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="A",
                column_name="3",  # Valid column but no well defined
                field_index=0,
            )


def test_write_hcs_well_image_memory_store():
    """Test write_hcs_well_image with non-file store (MemoryStore)."""
    import zarr
    from packaging import version

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
        name="Memory Store Test Plate",
        field_count=1,
    )

    # Create sample image data
    data = np.random.randint(0, 255, size=(1, 1, 5, 64, 64), dtype=np.uint8)
    ngff_image = nz.NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = nz.to_multiscales(ngff_image)

    # Test with memory store (non-file store)
    try:
        from zarr.storage import MemoryStore

        memory_store = MemoryStore()
    except (AttributeError, ImportError):
        # MemoryStore not available in this zarr version
        pytest.skip("MemoryStore not available in this zarr version")

    # Create plate structure
    hcs_plate = HCSPlate(memory_store, plate_metadata)
    to_hcs_zarr(hcs_plate, memory_store)

    # Check zarr version to determine expected behavior
    zarr_version = version.parse(zarr.__version__)

    if zarr_version.major >= 3:
        # Zarr 3.x should work with StorePath
        try:
            # Write well image - should work with fixed non-file store handling
            write_hcs_well_image(
                store=memory_store,
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="A",
                column_name="1",
                field_index=0,
            )

            # Verify the data was written to the correct location
            root = zarr.group(memory_store)
            assert "A" in root, "Row group 'A' not found"
            assert "1" in root["A"], "Column group '1' not found in row 'A'"
            assert "0" in root["A/1"], "Field group '0' not found in well 'A/1'"

            # Check that the multiscales metadata exists
            field_group = root["A/1/0"]
            assert "multiscales" in field_group.attrs, "Multiscales metadata not found"

            # Just check that the field group exists (data was written successfully)
            assert field_group is not None, "Field group should exist after writing"

        except ImportError:
            # StorePath not available even in zarr 3.x
            pytest.skip("StorePath not available in this zarr 3.x version")
    else:
        # Zarr 2.x should raise NotImplementedError
        with pytest.raises(
            NotImplementedError,
            match="Non-file stores with zarr-python 2.x are not fully supported",
        ):
            write_hcs_well_image(
                store=memory_store,
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="A",
                column_name="1",
                field_index=0,
            )


def test_write_hcs_well_image_multiple_fields():
    """Test writing multiple fields to the same well."""
    columns = [PlateColumn(name="1")]
    rows = [PlateRow(name="A")]
    wells = [PlateWell(path="A/1", rowIndex=0, columnIndex=0)]

    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name="Multi-field Test Plate",
        field_count=3,
    )

    # Create sample image data
    data = np.random.randint(0, 255, size=(1, 1, 3, 32, 32), dtype=np.uint8)
    ngff_image = nz.NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = nz.to_multiscales(ngff_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "multi_field_plate.ome.zarr"

        # Create plate structure
        hcs_plate = HCSPlate(None, plate_metadata)
        to_hcs_zarr(hcs_plate, str(output_path))

        # Write multiple fields to the same well
        for field_idx in range(3):
            write_hcs_well_image(
                store=str(output_path),
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="A",
                column_name="1",
                field_index=field_idx,
            )

            # Verify each field was created
            field_path = output_path / "A" / "1" / str(field_idx)
            assert field_path.exists(), f"Field {field_idx} was not created"

        # Verify well metadata includes all fields
        import zarr

        root = zarr.open_group(output_path)
        well_group = root["A/1"]
        well_attrs = well_group.attrs["well"]

        assert len(well_attrs["images"]) == 3, "Well metadata should include 3 images"

        for i, img in enumerate(well_attrs["images"]):
            assert img["path"] == str(i), f"Image {i} path should be '{i}'"
            assert img["acquisition"] == 0, f"Image {i} acquisition should be 0"
