# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test high content screening (HCS) functionality."""

import pytest
import tempfile
from pathlib import Path
from packaging import version

import zarr
from ngff_zarr.hcs import from_hcs_zarr, to_hcs_zarr, HCSPlate, HCSWell
from ngff_zarr.v04.zarr_metadata import (
    Plate,
    PlateAcquisition,
    PlateColumn,
    PlateRow,
    PlateWell,
    Well,
    WellImage,
)

zarr_version = version.parse(zarr.__version__)


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data" / "input"


@pytest.fixture
def hcs_data_path(test_data_dir):
    """Return path to HCS test data."""
    return test_data_dir / "hcs.ome.zarr"


def test_hcs_data_exists(hcs_data_path):
    """Test that HCS test data exists."""
    assert hcs_data_path.exists()
    assert hcs_data_path.is_dir()
    assert (hcs_data_path / ".zattrs").exists()


def test_from_hcs_zarr_basic(hcs_data_path):
    """Test basic loading of HCS data."""
    plate = from_hcs_zarr(hcs_data_path)

    assert isinstance(plate, HCSPlate)
    assert plate.name == "Plate Name 0"

    # Check plate structure
    assert len(plate.rows) == 2  # A, B
    assert len(plate.columns) == 2  # 1, 2
    assert len(plate.wells) == 4  # A/1, A/2, B/1, B/2

    # Check row names
    row_names = [row.name for row in plate.rows]
    assert row_names == ["A", "B"]

    # Check column names
    column_names = [col.name for col in plate.columns]
    assert column_names == ["1", "2"]

    # Check wells
    well_paths = [well.path for well in plate.wells]
    assert "A/1" in well_paths
    assert "A/2" in well_paths
    assert "B/1" in well_paths
    assert "B/2" in well_paths


def test_from_hcs_zarr_acquisitions(hcs_data_path):
    """Test acquisition metadata loading."""
    plate = from_hcs_zarr(hcs_data_path)

    assert plate.acquisitions is not None
    assert len(plate.acquisitions) == 1

    acq = plate.acquisitions[0]
    assert acq.id == 0


def test_from_hcs_zarr_field_count(hcs_data_path):
    """Test field count metadata."""
    plate = from_hcs_zarr(hcs_data_path)

    assert plate.field_count == 2


def test_get_well_by_name(hcs_data_path):
    """Test getting well by row and column name."""
    plate = from_hcs_zarr(hcs_data_path)

    well = plate.get_well("A", "1")
    assert well is not None
    assert isinstance(well, HCSWell)
    assert well.path == "A/1"
    assert well.row_index == 0
    assert well.column_index == 0

    # Test non-existent well
    none_well = plate.get_well("Z", "99")
    assert none_well is None


def test_get_well_by_indices(hcs_data_path):
    """Test getting well by indices."""
    plate = from_hcs_zarr(hcs_data_path)

    well = plate.get_well_by_indices(0, 0)  # A/1
    assert well is not None
    assert well.path == "A/1"

    well = plate.get_well_by_indices(1, 1)  # B/2
    assert well is not None
    assert well.path == "B/2"

    # Test out of bounds
    none_well = plate.get_well_by_indices(99, 99)
    assert none_well is None


def test_well_images(hcs_data_path):
    """Test well image access."""
    plate = from_hcs_zarr(hcs_data_path)
    well = plate.get_well("A", "1")

    assert well is not None
    assert len(well.images) == 2  # Two fields of view

    # Check image paths
    image_paths = [img.path for img in well.images]
    assert "0" in image_paths
    assert "1" in image_paths

    # Check acquisitions
    for img in well.images:
        assert img.acquisition == 0


def test_get_image_from_well(hcs_data_path):
    """Test getting multiscales image from well."""
    plate = from_hcs_zarr(hcs_data_path)
    well = plate.get_well("A", "1")

    assert well is not None

    # Get first field of view
    image = well.get_image(0)
    assert image is not None

    # Check that it's a proper multiscales object
    assert hasattr(image, "images")
    assert len(image.images) >= 1  # Should have at least one scale level

    # Test invalid field index
    none_image = well.get_image(99)
    assert none_image is None


def test_get_image_by_acquisition(hcs_data_path):
    """Test getting image by acquisition ID."""
    plate = from_hcs_zarr(hcs_data_path)
    well = plate.get_well("A", "1")

    assert well is not None

    # Get image from acquisition 0
    image = well.get_image_by_acquisition(0, 0)
    assert image is not None

    # Test non-existent acquisition
    none_image = well.get_image_by_acquisition(99, 0)
    assert none_image is None


def test_plate_metadata_creation():
    """Test creating plate metadata objects."""
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ]
    acquisitions = [PlateAcquisition(id=0, name="Test Acquisition")]

    plate = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        acquisitions=acquisitions,
        field_count=2,
        name="Test Plate",
    )

    assert plate.name == "Test Plate"
    assert len(plate.columns) == 2
    assert len(plate.rows) == 2
    assert len(plate.wells) == 4
    assert len(plate.acquisitions) == 1


def test_well_metadata_creation():
    """Test creating well metadata objects."""
    images = [
        WellImage(path="0", acquisition=0),
        WellImage(path="1", acquisition=0),
    ]

    well = Well(images=images, version="0.4")

    assert len(well.images) == 2
    assert well.version == "0.4"
    assert well.images[0].path == "0"
    assert well.images[0].acquisition == 0


def test_to_hcs_zarr_basic():
    """Test basic HCS writing functionality."""
    # Create a simple plate structure
    columns = [PlateColumn(name="1")]
    rows = [PlateRow(name="A")]
    wells = [PlateWell(path="A/1", rowIndex=0, columnIndex=0)]

    plate_metadata = Plate(
        columns=columns, rows=rows, wells=wells, name="Test Plate", field_count=1
    )

    plate = HCSPlate(None, plate_metadata)

    # Write to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plate.ome.zarr"
        to_hcs_zarr(plate, str(output_path))

        # Check that the structure was created
        assert output_path.exists()

        # Check metadata
        root = zarr.open_group(str(output_path), mode="r")
        attrs = root.attrs.asdict()

        assert "ome" in attrs
        assert "plate" in attrs["ome"]

        plate_attrs = attrs["ome"]["plate"]
        assert plate_attrs["name"] == "Test Plate"
        assert len(plate_attrs["wells"]) == 1
        assert len(plate_attrs["rows"]) == 1
        assert len(plate_attrs["columns"]) == 1


def test_validate_hcs_metadata(hcs_data_path):
    """Test HCS metadata validation."""
    # This should not raise an exception
    plate = from_hcs_zarr(hcs_data_path, validate=True)
    assert plate is not None


def test_round_trip_basic():
    """Test basic round trip: create -> save -> load."""
    # Create simple plate metadata
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    ]
    acquisitions = [PlateAcquisition(id=0, name="Test Acq", maximumfieldcount=1)]

    original_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        acquisitions=acquisitions,
        field_count=1,
        name="Round Trip Test",
        version="0.4",
    )

    original_plate = HCSPlate(None, original_metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "roundtrip.ome.zarr"

        # Save
        to_hcs_zarr(original_plate, str(output_path))

        # Load
        loaded_plate = from_hcs_zarr(str(output_path))

        # Compare metadata
        assert loaded_plate.name == original_plate.name
        assert len(loaded_plate.rows) == len(original_plate.rows)
        assert len(loaded_plate.columns) == len(original_plate.columns)
        assert len(loaded_plate.wells) == len(original_plate.wells)
        assert loaded_plate.field_count == original_plate.field_count

        # Compare acquisitions
        assert loaded_plate.acquisitions is not None
        assert original_plate.acquisitions is not None
        assert len(loaded_plate.acquisitions) == len(original_plate.acquisitions)

        loaded_acq = loaded_plate.acquisitions[0]
        original_acq = original_plate.acquisitions[0]
        assert loaded_acq.id == original_acq.id
        assert loaded_acq.name == original_acq.name
        assert loaded_acq.maximumfieldcount == original_acq.maximumfieldcount


def test_hcs_image_axes_and_dims(hcs_data_path):
    """Test that HCS images have correct axes and dimensions."""
    plate = from_hcs_zarr(hcs_data_path)
    well = plate.get_well("A", "1")
    image = well.get_image(0)

    assert image is not None

    # Check the first scale level
    ngff_image = image.images[0]

    # Should have t, c, z, y, x dimensions based on the test data
    expected_dims = ("t", "c", "z", "y", "x")
    assert ngff_image.dims == expected_dims

    # Check that the data has the right number of dimensions
    assert len(ngff_image.data.shape) == 5


@pytest.mark.skipif(zarr_version < version.parse("3"), reason="zarr version < 3")
def test_write_hcs_v05_complete():
    """Test complete HCS v0.5 writing workflow with plates, wells, and images."""
    import numpy as np
    from ngff_zarr import NgffImage, to_multiscales
    from ngff_zarr.hcs import write_hcs_well_image

    # Create plate metadata for v0.5
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ]
    acquisitions = [
        PlateAcquisition(id=0, name="Time 0", maximumfieldcount=2),
        PlateAcquisition(id=1, name="Time 1", maximumfieldcount=2),
    ]

    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        acquisitions=acquisitions,
        field_count=2,
        name="Test v0.5 Plate",
        version="0.5",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_v05.ome.zarr"

        # First, create the plate structure
        plate = HCSPlate(None, plate_metadata)
        to_hcs_zarr(plate, str(output_path))

        # Verify plate structure was created with correct version
        root = zarr.open_group(str(output_path), mode="r")
        attrs = root.attrs.asdict()

        assert "ome" in attrs
        assert attrs["ome"]["version"] == "0.5"
        assert "plate" in attrs["ome"]

        plate_attrs = attrs["ome"]["plate"]
        assert plate_attrs["name"] == "Test v0.5 Plate"
        # Version should NOT be in plate dict for v0.5
        assert "version" not in plate_attrs
        assert len(plate_attrs["wells"]) == 4
        assert len(plate_attrs["acquisitions"]) == 2

        # Create synthetic image data
        data = np.random.randint(0, 255, size=(1, 2, 10, 256, 256), dtype=np.uint8)
        ngff_image = NgffImage(
            data=data,
            dims=["t", "c", "z", "y", "x"],
            scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
            translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        )
        multiscales = to_multiscales(ngff_image, scale_factors=[2])

        # Write images to wells using v0.5
        for well in wells[:2]:  # Write to first two wells for testing
            row_name = rows[well.rowIndex].name
            col_name = columns[well.columnIndex].name

            # Write two fields per well
            for field_idx in range(2):
                write_hcs_well_image(
                    store=str(output_path),
                    multiscales=multiscales,
                    plate_metadata=plate_metadata,
                    row_name=row_name,
                    column_name=col_name,
                    field_index=field_idx,
                    acquisition_id=0,
                    version="0.5",
                )

        # Reopen to see the updates (the previous root was opened in read mode before writes)
        root = zarr.open_group(str(output_path), mode="r")

        # Verify well metadata structure for v0.5
        well_path = "A/1"
        well_group = root[well_path]
        well_attrs = well_group.attrs.asdict()

        # v0.5 should have ome wrapper with version at top level
        assert "ome" in well_attrs
        assert "version" in well_attrs["ome"]
        assert well_attrs["ome"]["version"] == "0.5"
        assert "well" in well_attrs["ome"]

        well_data = well_attrs["ome"]["well"]
        # Version should NOT be in well dict for v0.5
        assert "version" not in well_data
        assert "images" in well_data
        assert len(well_data["images"]) == 2

        # Verify images were written correctly
        for field_idx in range(2):
            field_path = f"{well_path}/{field_idx}"
            assert field_path in root
            field_group = root[field_path]
            field_attrs = field_group.attrs.asdict()

            # Field should have ome wrapper for v0.5
            assert "ome" in field_attrs
            assert "version" in field_attrs["ome"]
            assert field_attrs["ome"]["version"] == "0.5"

        # Round-trip test: read back and verify
        loaded_plate = from_hcs_zarr(str(output_path))
        assert loaded_plate.metadata.version == "0.5"
        assert loaded_plate.name == "Test v0.5 Plate"
        assert len(loaded_plate.wells) == 4

        # Verify well data
        loaded_well = loaded_plate.get_well("A", "1")
        assert loaded_well is not None
        assert len(loaded_well.images) == 2

        # Verify image data
        loaded_image = loaded_well.get_image(0)
        assert loaded_image is not None
        assert len(loaded_image.images) >= 1


@pytest.mark.skipif(zarr_version < version.parse("3"), reason="zarr version < 3")
def test_write_hcs_v05_vs_v04_metadata_structure():
    """Test that v0.5 and v0.4 have correct metadata structure differences."""
    import numpy as np
    from ngff_zarr import NgffImage, to_multiscales
    from ngff_zarr.hcs import write_hcs_well_image

    columns = [PlateColumn(name="1")]
    rows = [PlateRow(name="A")]
    wells = [PlateWell(path="A/1", rowIndex=0, columnIndex=0)]

    # Create synthetic image
    data = np.random.randint(0, 255, size=(1, 1, 5, 128, 128), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test v0.4
        v04_path = Path(tmpdir) / "v04.ome.zarr"
        plate_v04 = Plate(
            columns=columns,
            rows=rows,
            wells=wells,
            name="v0.4 Plate",
            version="0.4",
        )
        hcs_plate_v04 = HCSPlate(None, plate_v04)
        to_hcs_zarr(hcs_plate_v04, str(v04_path))

        write_hcs_well_image(
            store=str(v04_path),
            multiscales=multiscales,
            plate_metadata=plate_v04,
            row_name="A",
            column_name="1",
            field_index=0,
            version="0.4",
        )

        # Test v0.5
        v05_path = Path(tmpdir) / "v05.ome.zarr"
        plate_v05 = Plate(
            columns=columns,
            rows=rows,
            wells=wells,
            name="v0.5 Plate",
            version="0.5",
        )
        hcs_plate_v05 = HCSPlate(None, plate_v05)
        to_hcs_zarr(hcs_plate_v05, str(v05_path))

        write_hcs_well_image(
            store=str(v05_path),
            multiscales=multiscales,
            plate_metadata=plate_v05,
            row_name="A",
            column_name="1",
            field_index=0,
            version="0.5",
        )

        # Compare metadata structures
        root_v04 = zarr.open_group(str(v04_path), mode="r")
        root_v05 = zarr.open_group(str(v05_path), mode="r")

        # Plate level
        attrs_v04 = root_v04.attrs.asdict()
        attrs_v05 = root_v05.attrs.asdict()

        # Both should have ome wrapper at plate level
        assert "ome" in attrs_v04
        assert "ome" in attrs_v05

        # v0.4 has version in plate dict
        assert "version" in attrs_v04["ome"]["plate"]
        # v0.5 should NOT have version in plate dict
        assert "version" not in attrs_v05["ome"]["plate"]

        # Well level
        well_v04 = root_v04["A/1"]
        well_v05 = root_v05["A/1"]

        well_attrs_v04 = well_v04.attrs.asdict()
        well_attrs_v05 = well_v05.attrs.asdict()

        # v0.4 has well dict at top level
        assert "well" in well_attrs_v04
        assert "version" in well_attrs_v04["well"]

        # v0.5 has ome wrapper
        assert "ome" in well_attrs_v05
        assert "well" in well_attrs_v05["ome"]
        assert "version" in well_attrs_v05["ome"]
        # Version should NOT be in well dict for v0.5
        assert "version" not in well_attrs_v05["ome"]["well"]


if __name__ == "__main__":
    pytest.main([__file__])
