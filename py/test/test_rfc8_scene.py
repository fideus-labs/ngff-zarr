# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Scene metadata: the RFC-5 store shape and the RFC-8 attribute shape."""

import json
from pathlib import Path

import pytest
from ngff_zarr import Collection, to_collection_zarr
from ngff_zarr.rfc8 import (
    Scene,
    node_from_ome_dict,
    read_scene,
    scene,
    set_scene,
    validate_collection,
    write_scene,
)
from ngff_zarr.structural_validation import ValidationError
from ngff_zarr.v06.zarr_metadata import Bijection
from ngff_zarr.v09.zarr_metadata import (
    Affine,
    Axis,
    CoordinateSystem,
    Translation,
)
from ngff_zarr.v09.zarr_metadata import (
    CoordinateSystemIdentifier as Identifier,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rfc8"


def _tiles_scene() -> Scene:
    return Scene(
        coordinateSystems=[
            CoordinateSystem(
                name="world",
                id="world",
                axes=[
                    Axis(name="y", type="space", unit="micrometer"),
                    Axis(name="x", type="space", unit="micrometer"),
                ],
            )
        ],
        coordinateTransformations=[
            Translation(
                translation=[0.0, 100.0],
                input=Identifier(id="physical", path="./tile_0.zarr"),
                output=Identifier(id="world"),
            ),
            Translation(
                translation=[100.0, 0.0],
                input=Identifier(id="physical", path="./tile_1.zarr"),
                output=Identifier(id="world"),
            ),
        ],
    )


def test_scene_attribute_round_trips():
    collection = Collection("tiles", nodes=[])
    set_scene(collection, _tiles_scene())
    serialized = collection.attributes["scene"]
    assert serialized["coordinateSystems"][0]["id"] == "world"
    assert serialized["coordinateTransformations"][0]["input"] == {
        "path": "./tile_0.zarr",
        "id": "physical",
    }

    parsed = scene(collection)
    assert parsed.coordinateSystems[0].id == "world"
    assert parsed.coordinateTransformations[0].input.id == "physical"
    assert parsed.coordinateTransformations[0].input.path == "./tile_0.zarr"
    assert parsed.coordinateTransformations[1].translation == [100.0, 0.0]

    set_scene(collection, None)
    assert scene(collection) is None


def test_valid_scene_passes_validation():
    collection = Collection("tiles", nodes=[])
    set_scene(collection, _tiles_scene())
    validate_collection(collection, version="0.9.dev3")


def test_scene_validation_rules():
    collection = Collection("tiles", nodes=[])
    collection.attributes = {"scene": {}}
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(collection, version="0.9.dev3")
    assert exc_info.value.rule.value == "scene-transformations-required"

    # A name-based reference in a dev3 scene is refused.
    collection.attributes = {
        "scene": {
            "coordinateTransformations": [
                {
                    "type": "identity",
                    "input": {"name": "physical"},
                    "output": {"id": "world"},
                }
            ]
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(collection, version="0.9.dev3")
    assert exc_info.value.rule.value == "reference-id-required"

    # An unresolved scene reference without a path is refused.
    collection.attributes = {
        "scene": {
            "coordinateTransformations": [
                {
                    "type": "identity",
                    "input": {"id": "physical"},
                    "output": {"id": "world"},
                }
            ]
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(collection, version="0.9.dev3")
    assert exc_info.value.rule.value == "reference-path-required"


def test_write_and_read_scene_at_0_6_family(tmp_path):
    for version, expected_tag in (("0.6", "0.6rc0"), ("0.9.dev1", "0.9.dev1")):
        store = tmp_path / f"scene-{version}.ome.zarr"
        write_scene(store, _tiles_scene(), version=version)
        document = json.loads((store / "zarr.json").read_text())
        ome = document["attributes"]["ome"]
        assert ome["version"] == expected_tag
        assert "scene" in ome

        parsed = read_scene(store)
        assert parsed.coordinateSystems[0].name == "world"
        assert parsed.coordinateTransformations[0].input.path == "./tile_0.zarr"


def test_write_and_read_scene_at_dev3(tmp_path):
    store = tmp_path / "tiles.ome.zarr"
    with pytest.raises(ValueError, match="to_collection_zarr"):
        write_scene(store, _tiles_scene(), version="0.9.dev3")

    to_collection_zarr(store, Collection("tiles", nodes=[]))
    write_scene(store, _tiles_scene(), version="0.9.dev3")
    document = json.loads((store / "zarr.json").read_text())
    ome = document["attributes"]["ome"]
    assert ome["type"] == "collection"
    assert "scene" in ome["attributes"]

    parsed = read_scene(store)
    assert parsed.coordinateTransformations[1].input.path == "./tile_1.zarr"

    # The document still parses as a collection whose scene validates.
    node, _ = node_from_ome_dict(ome)
    validate_collection(node, version="0.9.dev3")


def test_upstream_scene_examples_parse():
    # Byte-identical copies of ome/ngff-spec@0.9dev examples/scene/*.json.
    from ngff_zarr.rfc8.scene import _parse_scene

    for name in ("scene_registration.json", "scene_stitching.json"):
        document = json.loads((FIXTURES / "scene" / name).read_text())
        ome = document["attributes"]["ome"]
        parsed = _parse_scene(ome["scene"], ome.get("version"))
        assert parsed.coordinateTransformations
        first = parsed.coordinateTransformations[0]
        assert first.input is not None
        assert first.input.name is not None


def test_scene_read_returns_none_without_scene(tmp_path):
    store = tmp_path / "plain.ome.zarr"
    to_collection_zarr(store, Collection("c", nodes=[]))
    assert read_scene(store) is None


def test_bijection_scene_transform_round_trips():
    value = Scene(
        coordinateTransformations=[
            Bijection(
                forward=Affine(affine=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                inverse=Affine(affine=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                input=Identifier(id="physical", path="JRC2018F"),
                output=Identifier(id="physical2", path="FCWB"),
            )
        ]
    )
    collection = Collection("templates", nodes=[])
    set_scene(collection, value)
    parsed = scene(collection)
    assert parsed.coordinateTransformations[0].input.path == "JRC2018F"
    assert parsed.coordinateTransformations[0].forward.type == "affine"
