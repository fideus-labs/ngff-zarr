# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""OME-Zarr 0.9.dev3 multiscale node stores: write, read, upgrade, convert."""

import json

import numpy as np
import pytest
from ngff_zarr import (
    Collection,
    NgffMultiscales,
    Node,
    OmePath,
    from_collection_zarr,
    from_ome_zarr,
    to_collection_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
    upgrade_ome_zarr,
)
from ngff_zarr.rfc8.multiscale import (
    multiscale_ome_dict,
    multiscales_entry_from_ome,
)


def _multiscales():
    image = to_ngff_image(
        np.arange(64, dtype=np.uint16).reshape(8, 8),
        dims=("y", "x"),
        scale={"y": 2.0, "x": 2.0},
        translation={"y": 1.0, "x": 1.0},
    )
    return to_multiscales(image, scale_factors=[2])


def test_dev3_store_shape(tmp_path):
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, _multiscales(), version="0.9.dev3")
    ome = json.loads((store / "zarr.json").read_text())["attributes"]["ome"]

    assert ome["version"] == "0.9.dev3"
    assert ome["type"] == "multiscale"
    assert "multiscales" not in ome
    systems = ome["attributes"]["coordinateSystems"]
    assert all("id" in system for system in systems)
    for node in ome["nodes"]:
        assert node["type"] == "singlescale"
        assert node["path"]["type"] == "zarr"
        assert node["path"]["path"].startswith("./")
        (transform,) = node["attributes"]["coordinateTransformations"]
        assert transform["input"] == {"id": node["id"]}
        assert transform["output"] == {"id": systems[0]["id"]}


def test_dev3_round_trip(tmp_path):
    multiscales = _multiscales()
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, multiscales, version="0.9.dev3")

    read_back = from_ome_zarr(store, validate=True)
    assert read_back.images[0].dims == ["y", "x"]
    assert read_back.images[0].scale == {"y": 2.0, "x": 2.0}
    assert read_back.images[0].translation == {"y": 1.0, "x": 1.0}
    assert len(read_back.images) == len(multiscales.images)
    assert np.array_equal(
        np.asarray(read_back.images[0].data),
        np.asarray(multiscales.images[0].data),
    )

    # The read-back model writes again, at dev3 and downgraded.
    to_ome_zarr(tmp_path / "again.ome.zarr", read_back, version="0.9.dev3")
    again = from_ome_zarr(tmp_path / "again.ome.zarr")
    assert again.images[0].scale == {"y": 2.0, "x": 2.0}
    to_ome_zarr(tmp_path / "down.ome.zarr", read_back, version="0.6")
    down = from_ome_zarr(tmp_path / "down.ome.zarr")
    assert down.images[0].scale == {"y": 2.0, "x": 2.0}


def test_distributed_singlescale_documents_read(tmp_path):
    # The RFC-8 distributed layout: the multiscale document's singlescale
    # entries carry no inline transformations; each level's own zarr.json
    # holds its singlescale document.
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, _multiscales(), version="0.9.dev3")
    root_doc = json.loads((store / "zarr.json").read_text())
    ome = root_doc["attributes"]["ome"]
    for node in ome["nodes"]:
        attributes = node.pop("attributes")
        level_path = store / node["path"]["path"][2:]
        level_doc = json.loads((level_path / "zarr.json").read_text())
        level_doc.setdefault("attributes", {})["ome"] = {
            "version": "0.9.dev3",
            "type": "singlescale",
            "id": node["id"],
            "name": node["name"],
            "attributes": attributes,
        }
        (level_path / "zarr.json").write_text(json.dumps(level_doc))
    (store / "zarr.json").write_text(json.dumps(root_doc))

    read_back = from_ome_zarr(store)
    assert read_back.images[0].scale == {"y": 2.0, "x": 2.0}
    assert read_back.images[0].translation == {"y": 1.0, "x": 1.0}


def test_upgrade_dev1_to_dev3_in_place_keeps_chunks(tmp_path):
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, _multiscales(), version="0.9.dev1")
    dev1_ome = json.loads((store / "zarr.json").read_text())["attributes"]["ome"]
    level = store / dev1_ome["multiscales"][0]["datasets"][0]["path"]
    level_doc = (level / "zarr.json").read_bytes()

    upgrade_ome_zarr(store, version="0.9.dev3")
    ome = json.loads((store / "zarr.json").read_text())["attributes"]["ome"]
    assert ome["version"] == "0.9.dev3"
    assert ome["type"] == "multiscale"
    assert (level / "zarr.json").read_bytes() == level_doc
    upgraded = from_ome_zarr(store)
    assert upgraded.images[0].scale == {"y": 2.0, "x": 2.0}


def test_downgrade_dev3_to_dev1(tmp_path):
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, _multiscales(), version="0.9.dev3")
    upgrade_ome_zarr(store, tmp_path / "out.ome.zarr", version="0.9.dev1")
    ome = json.loads((tmp_path / "out.ome.zarr" / "zarr.json").read_text())[
        "attributes"
    ]["ome"]
    assert ome["version"] == "0.9.dev1"
    assert "multiscales" in ome


def test_collection_loads_a_dev3_image_child(tmp_path):
    root_store = tmp_path / "study.ome.zarr"
    collection = Collection(
        "study",
        nodes=[
            Node(
                type="multiscale",
                name="img",
                id="img",
                path=OmePath("zarr", "./img.ome.zarr"),
            )
        ],
    )
    to_collection_zarr(root_store, collection)
    to_ome_zarr(root_store / "img.ome.zarr", _multiscales(), version="0.9.dev3")

    loaded = from_collection_zarr(root_store).load("img")
    assert isinstance(loaded, NgffMultiscales)
    assert loaded.images[0].scale == {"y": 2.0, "x": 2.0}


def test_omero_and_method_metadata_round_trip(tmp_path):
    multiscales = _multiscales()
    from ngff_zarr import compute_omero_from_multiscales

    multiscales.metadata.omero = compute_omero_from_multiscales(multiscales)
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, multiscales, version="0.9.dev3")
    ome = json.loads((store / "zarr.json").read_text())["attributes"]["ome"]
    assert "omero" in ome["attributes"]

    read_back = from_ome_zarr(store)
    assert read_back.metadata.omero is not None
    assert len(read_back.metadata.omero.channels) == 1


def test_entry_conversion_is_lossless_for_names_and_ids():
    entry = {
        "name": "image",
        "coordinateSystems": [
            {
                "name": "physical",
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
            }
        ],
        "datasets": [
            {
                "path": "scale0/image",
                "coordinateTransformations": [
                    {
                        "type": "sequence",
                        "name": "scale0_to_physical",
                        "input": {"path": "scale0/image"},
                        "output": {"name": "physical"},
                        "transformations": [{"type": "scale", "scale": [2.0, 2.0]}],
                    }
                ],
            }
        ],
        "@type": "ngff:Image",
    }
    ome = multiscale_ome_dict(dict(entry), "0.9.dev3")
    # A nested dataset path sanitizes into the id production.
    assert ome["nodes"][0]["id"] == "scale0-image"
    assert ome["attributes"]["@type"] == "ngff:Image"

    rebuilt, omero = multiscales_entry_from_ome(ome)
    assert omero is None
    assert rebuilt["datasets"][0]["path"] == "scale0/image"
    transform = rebuilt["datasets"][0]["coordinateTransformations"][0]
    assert transform["input"]["id"] == "scale0-image"
    assert transform["input"]["path"] == "scale0/image"
    assert transform["output"]["name"] == "physical"
    assert rebuilt["@type"] == "ngff:Image"


def test_dev3_writes_rfc3_axis_models_the_gate_refuses_below(tmp_path):
    zarr = pytest.importorskip("zarr")
    source = tmp_path / "src.ome.zarr"
    group = zarr.open_group(str(source), mode="w")
    shape = (2,) * 6
    # zarr-python 2 spells this create_dataset; 3 spells it create_array.
    create = getattr(group, "create_array", None) or group.create_dataset
    array = create("0", shape=shape, dtype="uint8", chunks=shape)
    array[...] = np.zeros(shape, dtype="uint8")
    group.attrs["multiscales"] = [
        {
            "axes": [{"name": name, "type": "space"} for name in "abcdef"],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0] * 6}
                    ],
                }
            ],
            "version": "0.4",
        }
    ]
    multiscales = from_ome_zarr(source)

    with pytest.raises(ValueError, match="0.9 development series"):
        to_ome_zarr(tmp_path / "img.ome.zarr", multiscales, version="0.4")
    to_ome_zarr(tmp_path / "img.ome.zarr", multiscales, version="0.9.dev3")
    read_back = from_ome_zarr(tmp_path / "img.ome.zarr")
    assert read_back.images[0].dims == ["a", "b", "c", "d", "e", "f"]


def test_dev3_write_accepts_id_only_transform_references(tmp_path):
    from ngff_zarr.v06.zarr_metadata import CoordinateSystemIdentifier, Scale

    array = np.random.random((4, 8, 8)).astype("float32")
    multiscales = to_multiscales(array, [2])
    # An id distinct from every name: resolving it as a name would fail.
    multiscales.metadata.intrinsic_coordinate_system.id = "pixel-id"
    multiscales.metadata.coordinateTransformations = [
        Scale(
            scale=[2.0, 2.0, 2.0],
            input=CoordinateSystemIdentifier(id="pixel-id"),
            output=CoordinateSystemIdentifier(id="pixel-id"),
        )
    ]

    # 0.6 still requires names on both references.
    with pytest.raises(ValueError, match="names no input coordinate system"):
        to_ome_zarr(tmp_path / "refused.ome.zarr", multiscales, version="0.6")

    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, multiscales, version="0.9.dev3")
    document = json.loads((store / "zarr.json").read_text())
    attributes = document["attributes"]["ome"]["attributes"]
    assert attributes["coordinateSystems"][0]["id"] == "pixel-id"
    transform = attributes["coordinateTransformations"][0]
    assert transform["input"] == {"id": "pixel-id"}
    assert transform["output"] == {"id": "pixel-id"}
    read_back = from_ome_zarr(store, validate=True)
    assert read_back.metadata.coordinateSystems[0].id == "pixel-id"


def test_corrupt_distributed_singlescale_document_raises(tmp_path):
    store = tmp_path / "img.ome.zarr"
    to_ome_zarr(store, _multiscales(), version="0.9.dev3")
    root_doc = json.loads((store / "zarr.json").read_text())
    ome = root_doc["attributes"]["ome"]
    level_path = None
    for node in ome["nodes"]:
        node.pop("attributes")
        level_path = store / node["path"]["path"][2:]
    (store / "zarr.json").write_text(json.dumps(root_doc))
    (level_path / "zarr.json").write_text("{not json")

    with pytest.raises(Exception) as exc_info:
        from_ome_zarr(store)
    assert not isinstance(exc_info.value, KeyError)


def test_nameless_systems_refuse_name_based_writes(tmp_path):
    from ngff_zarr.v06.zarr_metadata import CoordinateSystem

    multiscales = _multiscales()
    axes = multiscales.metadata.coordinateSystems[0].axes
    multiscales.metadata.coordinateSystems = [
        CoordinateSystem(name=None, axes=axes, id="pixel")
    ]
    with pytest.raises(ValueError, match="carry a name"):
        to_ome_zarr(tmp_path / "img.ome.zarr", multiscales, version="0.9.dev1")


def test_project_axis_transforms_validate_with_a_version(tmp_path):
    from ngff_zarr.v06.zarr_metadata import (
        CoordinateSystem,
        CoordinateSystemIdentifier,
        ProjectAxis,
    )

    multiscales = _multiscales()
    intrinsic = multiscales.metadata.intrinsic_coordinate_system
    multiscales.metadata.coordinateSystems = [
        intrinsic,
        CoordinateSystem(name="line", axes=[intrinsic.axes[0]]),
    ]
    multiscales.metadata.coordinateTransformations = [
        ProjectAxis(
            droppedInputs=[1],
            input=CoordinateSystemIdentifier(name=intrinsic.name),
            output=CoordinateSystemIdentifier(name="line"),
        )
    ]
    to_ome_zarr(tmp_path / "img.ome.zarr", multiscales, version="0.9.dev1")
