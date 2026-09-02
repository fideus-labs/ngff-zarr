# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 coordinate attribute views and id-based references."""

import pytest
from ngff_zarr import Collection, Node
from ngff_zarr.rfc8 import (
    coordinate_systems,
    coordinate_transformations,
    node_from_ome_dict,
    node_to_ome_dict,
    set_coordinate_systems,
    set_coordinate_transformations,
    validate_collection,
)
from ngff_zarr.structural_validation import ValidationError
from ngff_zarr.v09.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Scale,
    Translation,
)


def _multiscale_node() -> Node:
    node = Node(type="multiscale", name="img", id="img")
    set_coordinate_systems(
        node,
        [
            CoordinateSystem(
                name="physical",
                id="physical",
                axes=[
                    Axis(name="y", type="space", unit="micrometer"),
                    Axis(name="x", type="space", unit="micrometer"),
                ],
            )
        ],
    )
    set_coordinate_transformations(
        node,
        [
            Scale(
                scale=[1.0, 2.0],
                input=CoordinateSystemIdentifier(id="s0"),
                output=CoordinateSystemIdentifier(id="physical"),
            )
        ],
    )
    return node


def test_coordinate_attributes_round_trip():
    node = _multiscale_node()
    document = node_to_ome_dict(node, version="0.9.dev3")
    assert document["attributes"]["coordinateSystems"][0]["id"] == "physical"
    serialized = document["attributes"]["coordinateTransformations"][0]
    assert serialized["input"] == {"id": "s0"}
    assert serialized["output"] == {"id": "physical"}
    assert "name" not in serialized["input"]

    parsed, _ = node_from_ome_dict(document)
    systems = coordinate_systems(parsed)
    assert systems[0].id == "physical"
    assert systems[0].axes[0].unit == "micrometer"
    transforms = coordinate_transformations(parsed)
    assert transforms[0].scale == [1.0, 2.0]
    assert transforms[0].input.id == "s0"
    assert transforms[0].output.axis_count(systems) == 2


def test_nameless_coordinate_system_round_trips():
    node = Node(type="multiscale", name="img")
    set_coordinate_systems(
        node,
        [
            CoordinateSystem(
                name=None, id="physical", axes=[Axis(name="x", type="space")]
            )
        ],
    )
    assert "name" not in node.attributes["coordinateSystems"][0]
    systems = coordinate_systems(node)
    assert systems[0].name is None
    assert systems[0].id == "physical"


def test_identifier_resolves_by_id_before_name():
    systems = [
        CoordinateSystem(name="a", id="b", axes=[Axis(name="x", type="space")]),
        CoordinateSystem(
            name="b",
            id="a",
            axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
        ),
    ]
    assert CoordinateSystemIdentifier(id="a").axis_count(systems) == 2
    assert CoordinateSystemIdentifier(name="a").axis_count(systems) == 1
    assert CoordinateSystemIdentifier(id="missing").axis_count(systems) is None


def test_sequence_transformations_parse_from_attributes():
    node = Node(type="singlescale", name="s0", id="s0")
    node.attributes = {
        "coordinateTransformations": [
            {
                "type": "sequence",
                "transformations": [
                    {"type": "scale", "scale": [1.0, 1.0]},
                    {"type": "translation", "translation": [0.0, 5.0]},
                ],
                "input": {"id": "s0"},
                "output": {"id": "physical"},
            }
        ]
    }
    systems = [
        CoordinateSystem(
            name="physical",
            id="physical",
            axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
        )
    ]
    transforms = coordinate_transformations(node, systems)
    sequence = transforms[0]
    assert sequence.input.id == "s0"
    assert sequence.output.id == "physical"
    assert isinstance(sequence.transformations[0], Scale)
    assert isinstance(sequence.transformations[1], Translation)


def test_validation_covers_coordinate_attributes():
    document = {
        "type": "multiscale",
        "name": "img",
        "nodes": [],
        "attributes": {
            "coordinateSystems": [
                {"name": "physical", "axes": [{"name": "x", "type": "space"}]}
            ]
        },
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(document, version="0.9.dev3")
    assert exc_info.value.rule.value == "coordinate-system-id-required"

    # A parsed tree is judged identically.
    parsed, _ = node_from_ome_dict(document)
    with pytest.raises(ValidationError):
        validate_collection(parsed, version="0.9.dev3")


def test_collection_with_coordinate_attributes_validates():
    collection = Collection("tiles", nodes=[_multiscale_node()])
    # The transformation input references the singlescale id "s0", which is
    # not declared in this document, so a path is required.
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(collection, version="0.9.dev3")
    assert exc_info.value.rule.value == "reference-path-required"

    collection.nodes[0].nodes = None
    collection.nodes[0].id = "img"
    node = Node(type="singlescale", name="s0", id="s0")
    collection.nodes.append(node)
    validate_collection(collection, version="0.9.dev3")


def test_external_reference_typed_paths_round_trip():
    node = Node(type="multiscale", name="img", id="img")
    typed_path = {"type": "zarr", "path": "./tile_0.zarr"}
    set_coordinate_systems(
        node,
        [
            CoordinateSystem(
                name="physical",
                id="physical",
                axes=[Axis(name="x", type="space")],
            )
        ],
    )
    set_coordinate_transformations(
        node,
        [
            Scale(
                scale=[2.0],
                input=CoordinateSystemIdentifier(id="physical"),
                output=CoordinateSystemIdentifier(id="elsewhere", path=typed_path),
            )
        ],
    )
    parsed = coordinate_transformations(node)
    assert parsed[0].output.path == typed_path
    serialized = node.attributes["coordinateTransformations"][0]
    assert serialized["output"]["path"] == typed_path


def test_reference_paths_pass_through_path_type_known():
    node = Node(type="multiscale", name="img", id="img")
    node.attributes = {
        "coordinateSystems": [
            {"id": "physical", "axes": [{"name": "x", "type": "space"}]}
        ],
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [1.0],
                "input": {"id": "physical"},
                "output": {
                    "id": "elsewhere",
                    "path": {"type": "google-sheet", "path": "./sheet"},
                },
            }
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(node, version="0.9.dev3")
    assert exc_info.value.rule.value == "path-type-known"


def test_id_only_serialized_systems_parse(tmp_path):
    import json

    import numpy as np
    from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr

    store = tmp_path / "img.ome.zarr"
    image = to_ngff_image(np.zeros((4, 4), dtype=np.uint8), dims=("y", "x"))
    to_ome_zarr(store, to_multiscales(image, scale_factors=[]), version="0.9.dev1")
    document = json.loads((store / "zarr.json").read_text())
    entry = document["attributes"]["ome"]["multiscales"][0]
    for system in entry["coordinateSystems"]:
        system["id"] = system.pop("name")
    for dataset in entry["datasets"]:
        for transform in dataset["coordinateTransformations"]:
            transform["output"] = {"id": transform["output"]["name"]}
    (store / "zarr.json").write_text(json.dumps(document))

    parsed = from_ome_zarr(store)
    assert parsed.metadata.coordinateSystems[0].name is None
    system_id = parsed.metadata.coordinateSystems[0].id
    assert system_id is not None
    sequence = parsed.metadata.datasets[0].coordinateTransformations[0]
    assert sequence.output.id == system_id
    assert sequence.transformations[0].scale == [1.0, 1.0]
    assert parsed.images[0].scale == {"y": 1.0, "x": 1.0}
