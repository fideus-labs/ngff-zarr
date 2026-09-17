# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""HCS plates and RFC-9 ``.ozx`` archives at OME-Zarr 0.6 (GH #748).

From 0.5 the plate and well documents live under a single ``ome`` namespace
beside ``ome.version``; 0.6 keeps that layout and only the version tag
changes. Every HCS writer path is exercised at each Zarr v3 version the
writers accept, and the written documents are checked against the bundled
plate and well schemas of that version.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image
from ngff_zarr.hcs import (
    HCSPlate,
    HCSPlateWriter,
    from_hcs_zarr,
    to_hcs_zarr,
    write_hcs_well_image,
)
from ngff_zarr.rfc9_zip import read_ozx_version
from ngff_zarr.v04.zarr_metadata import (
    Plate,
    PlateAcquisition,
    PlateColumn,
    PlateRow,
    PlateWell,
)

pytest.importorskip("jsonschema", reason="schema checks need the validate extra")

from jsonschema import ValidationError
from ngff_zarr.validate import validate as validate_ngff

# Every plate here is a Zarr v3 store, which the zarr-python 2 legs of the
# test matrix cannot read back.
pytestmark = pytest.mark.skipif(
    packaging.version.parse(zarr.__version__).major < 3,
    reason="OME-Zarr 0.5+ plates require zarr-python >= 3.0.0",
)

# The versions stored in Zarr v3 that the HCS writers accept: the 0.5 layout
# under the released 0.5 and 0.6 tags, plus the opt-in 0.9.dev1 development
# version. Every one of them can be zipped into an RFC-9 .ozx archive.
ZARR_V3_VERSIONS = ("0.5", "0.6", "0.9.dev1")


def _plate_metadata(version: str) -> Plate:
    columns = [PlateColumn(name="1"), PlateColumn(name="2")]
    rows = [PlateRow(name="A"), PlateRow(name="B")]
    wells = [
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ]
    return Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        name=f"Plate v{version}",
        field_count=2,
        version=version,
    )


def _field_image(seed: int):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 255, (2, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data=data, dims=["c", "y", "x"], scale={"y": 0.65, "x": 0.65})
    return data, to_multiscales(image, [2])


def _group_attrs(group_dir: Path) -> dict:
    """The attributes document of a Zarr v3 group."""
    return json.loads((group_dir / "zarr.json").read_text())["attributes"]


def _write_group_attrs(group_dir: Path, attrs: dict) -> None:
    doc = json.loads((group_dir / "zarr.json").read_text())
    doc["attributes"] = attrs
    (group_dir / "zarr.json").write_text(json.dumps(doc))


@pytest.mark.parametrize("version", ZARR_V3_VERSIONS)
def test_to_hcs_zarr_tags_the_plate_with_its_version(tmp_path, version):
    """The plate root is a Zarr v3 group whose ``ome.version`` is the plate's.

    The version is recorded once, on the ``ome`` namespace, not repeated in
    the plate document as at 0.4, and the document satisfies the plate schema
    bundled for that version.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata(version)
    to_hcs_zarr(HCSPlate(store=str(store), plate_metadata=plate_metadata), str(store))

    assert (store / "zarr.json").exists()
    assert not (store / ".zattrs").exists()
    root_attrs = _group_attrs(store)
    assert root_attrs["ome"]["version"] == version
    plate_doc = root_attrs["ome"]["plate"]
    assert "version" not in plate_doc
    assert [row["name"] for row in plate_doc["rows"]] == ["A", "B"]
    assert plate_doc["field_count"] == 2

    validate_ngff(root_attrs, version=version, model="plate")


@pytest.mark.parametrize("version", ZARR_V3_VERSIONS)
def test_write_hcs_well_image_round_trip(tmp_path, version):
    """Fields written at a Zarr v3 version read back through every entry point.

    The well and field documents carry the same version tag as the plate,
    satisfy the bundled well schema, and the pixel data survives a round trip
    through ``from_hcs_zarr`` as well as the ``plate/A/1/0`` sub-path form of
    ``from_ome_zarr``.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata(version)
    to_hcs_zarr(HCSPlate(store=str(store), plate_metadata=plate_metadata), str(store))

    fields = {("A", "1", 0): 0, ("A", "1", 1): 1, ("B", "2", 0): 2}
    expected = {}
    for (row, column, field_index), seed in fields.items():
        data, multiscales = _field_image(seed)
        expected[(row, column, field_index)] = data
        write_hcs_well_image(
            store=str(store),
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name=row,
            column_name=column,
            field_index=field_index,
            version=version,
        )

    well_attrs = _group_attrs(store / "A" / "1")
    assert well_attrs["ome"]["version"] == version
    assert "version" not in well_attrs["ome"]["well"]
    assert [img["path"] for img in well_attrs["ome"]["well"]["images"]] == ["0", "1"]
    validate_ngff(well_attrs, version=version, model="well")

    field_attrs = _group_attrs(store / "A" / "1" / "0")
    assert field_attrs["ome"]["version"] == version

    plate = from_hcs_zarr(str(store), validate=True)
    assert plate.metadata.version == version
    assert plate.name == plate_metadata.name
    for (row, column, field_index), data in expected.items():
        well = plate.get_well(row, column)
        assert well is not None
        assert well.metadata.version == version
        image = well.get_image(field_index)
        assert image is not None
        np.testing.assert_array_equal(np.asarray(image.images[0].data), data)
    assert len(plate.get_well("A", "1").images) == 2

    sub_path = from_ome_zarr(str(store / "A" / "1" / "0"), validate=True)
    np.testing.assert_array_equal(
        np.asarray(sub_path.images[0].data), expected[("A", "1", 0)]
    )


@pytest.mark.parametrize("version", ZARR_V3_VERSIONS)
def test_hcs_plate_writer_ozx_round_trip(tmp_path, version):
    """An HCS plate at any Zarr v3 version zips into an RFC-9 ``.ozx`` archive.

    The archive leads with the root ``zarr.json``, its ZIP comment records
    the plate's version, and ``from_hcs_zarr`` reads it back validated.
    """
    ozx_path = tmp_path / "plate.ozx"
    plate_metadata = _plate_metadata(version)
    data, multiscales = _field_image(7)

    with HCSPlateWriter(str(ozx_path), plate_metadata, version=version) as writer:
        writer.write_well_image(
            multiscales=multiscales, row_name="A", column_name="1", field_index=0
        )

    assert zipfile.is_zipfile(ozx_path)
    with zipfile.ZipFile(ozx_path) as zf:
        assert zf.namelist()[0] == "zarr.json"
    assert read_ozx_version(ozx_path) == version

    plate = from_hcs_zarr(str(ozx_path), validate=True)
    assert plate.metadata.version == version
    image = plate.get_well("A", "1").get_image(0)
    np.testing.assert_array_equal(np.asarray(image.images[0].data), data)


def test_hcs_plate_writer_ozx_rejects_zarr_v2_version(tmp_path):
    """RFC-9 is defined on Zarr v3, so a 0.4 plate cannot be zipped."""
    plate_metadata = _plate_metadata("0.4")
    with pytest.raises(ValueError, match=r"RFC-9.*requires.*0\.5 or later"):
        HCSPlateWriter(str(tmp_path / "plate.ozx"), plate_metadata, version="0.4")


def test_hcs_writers_reject_a_version_to_ome_zarr_does_not_write(tmp_path):
    """The HCS writers refuse 0.3 up front, before touching the store.

    Each field is written through ``to_ome_zarr``, which writes 0.4 and later
    only, so the plate structure must not be created for a version whose
    fields could never follow.
    """
    plate_metadata = _plate_metadata("0.3")
    store = tmp_path / "plate.ome.zarr"
    _, multiscales = _field_image(0)

    with pytest.raises(ValueError, match="Unsupported OME-Zarr version: 0.3"):
        to_hcs_zarr(
            HCSPlate(store=str(store), plate_metadata=plate_metadata), str(store)
        )
    assert not store.exists()

    with pytest.raises(ValueError, match="Unsupported OME-Zarr version: 0.3"):
        write_hcs_well_image(
            store=str(store),
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name="A",
            column_name="1",
            version="0.3",
        )
    assert not store.exists()

    with pytest.raises(ValueError, match="Unsupported OME-Zarr version: 0.3"):
        HCSPlateWriter(str(store), plate_metadata, version="0.3")


def test_hcs_plate_writer_follows_the_plate_version(tmp_path):
    """Without an explicit ``version`` the writer uses the plate's.

    A 0.6 plate is then written at 0.6 throughout -- root, well and field --
    rather than at a fixed writer default that the plate root would not share.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata("0.6")
    _, multiscales = _field_image(3)

    with HCSPlateWriter(str(store), plate_metadata) as writer:
        assert writer.version == "0.6"
        writer.write_well_image(
            multiscales=multiscales, row_name="A", column_name="1", field_index=0
        )

    for group in (store, store / "A" / "1", store / "A" / "1" / "0"):
        assert _group_attrs(group)["ome"]["version"] == "0.6"


def test_write_hcs_well_image_without_an_acquisition(tmp_path):
    """``acquisition_id=None`` records no ``acquisition`` key for the image.

    The key is optional in the well document when the plate declares at most
    one acquisition. Its absence survives a later write to the same well --
    it is not turned into a made-up ``0`` -- and re-writing a field updates
    its one entry rather than adding a second at the same path.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata("0.6")
    to_hcs_zarr(HCSPlate(store=str(store), plate_metadata=plate_metadata), str(store))

    def write(field_index, acquisition_id):
        _, multiscales = _field_image(field_index)
        write_hcs_well_image(
            store=str(store),
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name="A",
            column_name="1",
            field_index=field_index,
            acquisition_id=acquisition_id,
            version="0.6",
        )

    def images():
        return _group_attrs(store / "A" / "1")["ome"]["well"]["images"]

    write(0, None)
    assert images() == [{"path": "0"}]

    write(1, 0)
    assert images() == [{"path": "0"}, {"path": "1", "acquisition": 0}]

    write(1, None)
    assert images() == [{"path": "0"}, {"path": "1"}]

    plate = from_hcs_zarr(str(store), validate=True)
    well = plate.get_well("A", "1")
    assert [img.acquisition for img in well.images] == [None, None]
    assert well.get_image(1) is not None


def test_write_hcs_well_image_requires_an_acquisition_when_plate_declares_several(
    tmp_path,
):
    """A plate with several acquisitions refuses an image that names none.

    The specification requires the reference then (the reader's
    ``well-acquisition-missing`` rule); the writer refuses up front rather
    than write a well document the reader would reject.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata("0.6")
    plate_metadata.acquisitions = [
        PlateAcquisition(id=0, name="Baseline"),
        PlateAcquisition(id=1, name="Post-treatment"),
    ]
    to_hcs_zarr(HCSPlate(store=str(store), plate_metadata=plate_metadata), str(store))
    _, multiscales = _field_image(0)

    def write(acquisition_id):
        write_hcs_well_image(
            store=str(store),
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name="A",
            column_name="1",
            field_index=0,
            acquisition_id=acquisition_id,
            version="0.6",
        )

    with pytest.raises(ValueError, match="declares 2 acquisitions.*acquisition_id"):
        write(None)
    assert not (store / "A").exists()

    write(1)
    well = from_hcs_zarr(str(store), validate=True).get_well("A", "1")
    assert [img.acquisition for img in well.images] == [1]


def test_hcs_writers_reject_a_version_the_plate_was_not_created_with(tmp_path):
    """A ``version`` other than ``plate_metadata.version`` is refused up front.

    The plate root is written at the plate's version and the wells at the
    given one; letting them differ would leave a plate whose wells are not
    readable with its root. Nothing is written before the refusal.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata("0.6")
    _, multiscales = _field_image(0)
    mismatch = r"version '0\.5' does not match the plate's version '0\.6'"

    with pytest.raises(ValueError, match=mismatch):
        HCSPlateWriter(str(store), plate_metadata, version="0.5")
    assert not store.exists()

    with pytest.raises(ValueError, match=mismatch):
        write_hcs_well_image(
            store=str(store),
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name="A",
            column_name="1",
            version="0.5",
        )
    assert not store.exists()


@pytest.mark.parametrize("version", ZARR_V3_VERSIONS)
def test_from_hcs_zarr_validates_against_the_stored_version(tmp_path, version):
    """``validate=True`` checks the plate document at the version it records.

    From 0.5 the plate document lives under ``ome``, where the 0.4 schema --
    the reader's former fixed choice -- never looks; validating at the stored
    version means a broken 0.6 plate is reported rather than accepted.
    """
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = _plate_metadata(version)
    to_hcs_zarr(HCSPlate(store=str(store), plate_metadata=plate_metadata), str(store))

    plate = from_hcs_zarr(str(store), validate=True)
    assert plate.metadata.version == version

    root_attrs = _group_attrs(store)
    del root_attrs["ome"]["plate"]["rows"]
    _write_group_attrs(store, root_attrs)

    with pytest.raises(ValidationError, match="'rows' is a required property"):
        from_hcs_zarr(str(store), validate=True)
    # Without validation the reader is lenient, as before.
    assert from_hcs_zarr(str(store)).metadata.version == version
