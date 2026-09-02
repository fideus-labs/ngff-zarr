# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 HCS attributes: plate, well and acquisition."""

from pathlib import Path

import pytest
from ngff_zarr import (
    Collection,
    Node,
    OmePath,
    Reference,
    from_collection_zarr,
    to_collection_zarr,
)
from ngff_zarr.rfc8 import (
    PlateAttribute,
    PlateEntry,
    WellAttribute,
    acquisition,
    plate,
    plate_collection_from_hcs,
    set_acquisition,
    set_plate,
    set_well,
    validate_collection,
    well,
)
from ngff_zarr.structural_validation import ValidationError

HCS_STORE = Path(__file__).parent / "data" / "input" / "hcs.ome.zarr"


def _wide_plate() -> Collection:
    root = Collection("hcs-plate-001", nodes=[])
    set_plate(
        root,
        PlateAttribute(
            columns=[PlateEntry("1", name="1")],
            rows=[PlateEntry("A", name="A")],
            acquisitions=[PlateEntry("acq_0", name="Acquisition Round 1")],
        ),
    )
    well_node = Collection("well A01", nodes=[])
    set_well(
        well_node,
        WellAttribute(column=Reference("1"), row=Reference("A")),
    )
    image = Node(
        type="multiscale",
        name="A01_0",
        id="A01_0",
        path=OmePath("zarr", "./A/01/001.img"),
    )
    set_acquisition(image, Reference("acq_0"))
    well_node.nodes.append(image)
    root.nodes.append(well_node)
    return root


def test_plate_attributes_round_trip():
    root = _wide_plate()
    serialized = root.attributes["plate"]
    assert serialized["columns"] == [{"id": "1", "name": "1"}]
    assert serialized["acquisitions"][0]["id"] == "acq_0"

    parsed = plate(root)
    assert parsed.columns == [PlateEntry("1", name="1")]
    assert parsed.rows == [PlateEntry("A", name="A")]

    well_node = root.nodes[0]
    parsed_well = well(well_node)
    assert parsed_well.column == Reference("1")
    assert parsed_well.row == Reference("A")

    image = well_node.nodes[0]
    assert acquisition(image) == Reference("acq_0")

    set_acquisition(image, None)
    assert acquisition(image) is None
    set_well(well_node, None)
    assert well(well_node) is None
    set_plate(root, None)
    assert plate(root) is None


def test_wide_plate_validates_and_round_trips(tmp_path):
    root = _wide_plate()
    validate_collection(root, version="0.9.dev3")

    store = tmp_path / "plate.ome.zarr"
    to_collection_zarr(store, root)
    read_back = from_collection_zarr(store, validate=True)
    assert plate(read_back.root).columns[0].id == "1"
    assert well(read_back.root.nodes[0]).row.id == "A"


def test_tall_layout_acquisition_on_a_sub_collection():
    root = _wide_plate()
    well_node = root.nodes[0]
    image = well_node.nodes[0]
    set_acquisition(image, None)
    sub = Collection("A01_acq0", nodes=[image])
    set_acquisition(sub, Reference("acq_0"))
    well_node.nodes = [sub]
    validate_collection(root, version="0.9.dev3")


def test_hcs_rules_fire():
    root = _wide_plate()
    root.attributes["plate"].pop("columns")
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(root, version="0.9.dev3")
    assert exc_info.value.rule.value == "plate-columns-rows-required"

    root = _wide_plate()
    root.nodes[0].attributes["well"]["column"] = {"id": "2"}
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(root, version="0.9.dev3")
    assert exc_info.value.rule.value == "well-reference-resolves"

    root = _wide_plate()
    root.nodes[0].nodes[0].attributes["acquisition"] = {"id": "acq_9"}
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(root, version="0.9.dev3")
    assert exc_info.value.rule.value == "acquisition-reference-resolves"

    orphan = Collection("orphan", nodes=[])
    set_well(orphan, WellAttribute(column=Reference("1"), row=Reference("A")))
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(orphan, version="0.9.dev3")
    assert exc_info.value.rule.value == "well-reference-resolves"


def test_plate_entry_ids_join_the_document_namespace():
    root = _wide_plate()
    root.id = "A"  # collides with the row id
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(root, version="0.9.dev3")
    assert exc_info.value.rule.value == "node-id-unique"


def test_plate_collection_from_hcs(tmp_path):
    if not HCS_STORE.exists():
        pytest.skip("HCS test data not downloaded")
    from ngff_zarr import NgffMultiscales, from_hcs_zarr

    hcs_plate = from_hcs_zarr(str(HCS_STORE))
    root = plate_collection_from_hcs(hcs_plate)

    parsed = plate(root)
    assert {entry.id for entry in parsed.rows} == {
        row.name for row in hcs_plate.metadata.rows
    }
    assert len(root.nodes) == len(hcs_plate.metadata.wells)
    validate_collection(root, version="0.9.dev3")

    # Written beside the source store, the collection is the RFC-8 view of
    # the same data: loading an image child reads the well image.
    import shutil

    plate_store = tmp_path / "plate.ome.zarr"
    shutil.copytree(HCS_STORE, plate_store)
    to_collection_zarr(plate_store, root)
    collection = from_collection_zarr(plate_store, validate=True)
    first_well = collection.root.nodes[0]
    image_node = first_well.nodes[0]
    loaded = collection.load(image_node)
    assert isinstance(loaded, NgffMultiscales)
    assert loaded.images[0].data.size > 0
