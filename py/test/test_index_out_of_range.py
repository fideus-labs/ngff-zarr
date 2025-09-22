"""Test for index out of range issue in write_hcs_well_image function."""

import pytest
import tempfile
from packaging import version
from pathlib import Path

import numpy as np
import dask.array as da
import zarr

from ngff_zarr import NgffImage, to_multiscales
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr, write_hcs_well_image
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell

zarr_version = version.parse(zarr.__version__)


def test_write_hcs_well_image_index_out_of_range():
    """Test that write_hcs_well_image handles sparse plates correctly.

    This tests a use case where the rowIndex/columnIndex in PlateWell metadata
    don't correspond directly to array indices in the rows/columns arrays.
    For example, a plate with rows ["A", "C"] where well "C/2" has rowIndex=1,
    but the actual position of "C" in the rows array is index 1.
    """
    # Create a sparse plate layout - missing row "B" and some columns
    columns = [
        PlateColumn(name="1"),
        PlateColumn(name="3"),  # Skip column "2"
        PlateColumn(name="5"),  # Skip column "4"
    ]
    rows = [
        PlateRow(name="A"),
        PlateRow(name="C"),  # Skip row "B"
    ]

    # Create wells with logical indices that might not match array positions
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(
            path="A/3", rowIndex=0, columnIndex=2
        ),  # columnIndex=2, but array index is 1
        PlateWell(
            path="C/1", rowIndex=2, columnIndex=0
        ),  # rowIndex=2, but array index is 1
        PlateWell(
            path="C/5", rowIndex=2, columnIndex=4
        ),  # both indices don't match array positions
    ]

    plate_metadata = Plate(
        columns=columns, rows=rows, wells=wells, name="Sparse Test Plate", field_count=1
    )

    # Create sample image data
    data = da.random.randint(
        0, 255, size=(1, 1, 10, 256, 256), dtype=np.uint8, chunks=(1, 1, 5, 128, 128)
    )
    ngff_image = NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = to_multiscales(ngff_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "sparse_plate.ome.zarr"

        # Create plate structure
        hcs_plate = HCSPlate(None, plate_metadata)
        to_hcs_zarr(hcs_plate, str(output_path))

        # Test writing to each well - this should work without errors
        test_cases = [
            ("A", "1"),
            ("A", "3"),
            ("C", "1"),
            ("C", "5"),
        ]

        for row_name, column_name in test_cases:
            # This should work correctly without index errors
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

            # Check that the multiscales structure exists
            assert (
                well_path / ".zattrs"
            ).exists(), f"Multiscales metadata not found for {row_name}/{column_name}"

            # Check that the first scale level exists
            scale_path = well_path / "scale0"
            assert (
                scale_path.exists()
            ), f"Scale 0 data not found for {row_name}/{column_name}"

    # Test error case: try to write to a well that doesn't exist in the metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "error_plate.ome.zarr"
        hcs_plate = HCSPlate(None, plate_metadata)
        to_hcs_zarr(hcs_plate, str(output_path))

        # This should raise an error for a well not in the metadata
        with pytest.raises((ValueError, KeyError)):
            write_hcs_well_image(
                store=str(output_path),
                multiscales=multiscales,
                plate_metadata=plate_metadata,
                row_name="A",
                column_name="2",  # This column doesn't exist in our sparse layout
                field_index=0,
            )


@pytest.mark.skipif(zarr_version < version.parse("3"), reason="zarr version < 3")
def test_write_hcs_well_image_non_file_store():
    """Test write_hcs_well_image with non-file store types."""

    from zarr.storage import MemoryStore

    # Create a simple plate layout
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
    data = da.random.randint(
        0, 255, size=(1, 1, 5, 128, 128), dtype=np.uint8, chunks=(1, 1, 5, 128, 128)
    )
    ngff_image = NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = to_multiscales(ngff_image)

    # Test with memory store
    memory_store = MemoryStore()

    # Create plate structure
    hcs_plate = HCSPlate(memory_store, plate_metadata)
    to_hcs_zarr(hcs_plate, memory_store)

    # This should work without errors - testing the main functionality
    # The previous version would fail with non-file stores
    try:
        write_hcs_well_image(
            store=memory_store,
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name="A",
            column_name="1",
            field_index=0,
        )
        # If we get here, the function completed successfully
        assert True, "Function completed successfully with memory store"
    except Exception as e:
        pytest.fail(f"write_hcs_well_image failed with memory store: {e}")
