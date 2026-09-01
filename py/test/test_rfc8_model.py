# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 node model: construction, parsing and serialization round trips."""

import json
from pathlib import Path

import pytest
from ngff_zarr import Collection, Node, OmePath, Reference
from ngff_zarr.rfc8 import (
    RFC8_ID_PATTERN,
    is_collection_node,
    is_prefixed_identifier,
    node_from_ome_dict,
    node_to_ome_dict,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rfc8"

#: Every checked-in example document; ``webknossos_inline_multiscale.json``
#: is a full ``zarr.json`` (the Zarr-group storage medium), the others are
#: standalone documents with a root ``ome`` key.
FIXTURE_FILES = [
    "base_multiscale_collection.json",
    "nested_collection.json",
    "external_collection/parent_collection.json",
    "external_collection/child_collection.json",
    "external_collection/resolved_collection.json",
    "webknossos_inline_multiscale.json",
    "mobie_image_labels_table.json",
]


def _ome_value(document: dict) -> dict:
    if "attributes" in document:
        return document["attributes"]["ome"]
    return document["ome"]


def test_collection_constructor_shape():
    collection = Collection("study", id="c1")
    assert collection.type == "collection"
    assert collection.name == "study"
    assert collection.nodes is None
    assert collection.path is None
    assert is_collection_node(collection)
    # ``type`` is pinned by the class, not an init parameter.
    with pytest.raises(TypeError):
        Collection(type="collection", name="study")


def test_node_requires_type_and_name():
    node = Node(type="multiscale", name="raw")
    assert not is_collection_node(node)
    with pytest.raises(TypeError):
        Node(name="raw")


def test_reference_and_path_are_frozen_values():
    path = OmePath("zarr", "./raw")
    assert Reference("raw", path) == Reference("raw", OmePath("zarr", "./raw"))
    with pytest.raises(AttributeError):
        path.path = "./other"


def test_id_pattern_matches_the_rfc_production():
    for accepted in ("s0", "A-1_b.2", "7fa2f064-62c5-4d55-9e6c-25acc1aa3a31"):
        assert RFC8_ID_PATTERN.match(accepted)
    for rejected in ("a b", "a/b", "", "é"):
        assert not RFC8_ID_PATTERN.match(rejected)


def test_prefixed_identifier_detection():
    assert is_prefixed_identifier("myorg:zip")
    assert is_prefixed_identifier("webknossos:settings")
    assert not is_prefixed_identifier("zip")
    assert not is_prefixed_identifier(":zip")
    assert not is_prefixed_identifier("myorg:")


def test_collection_parses_to_collection_class():
    node, version = node_from_ome_dict(
        {"version": "0.9.dev3", "type": "collection", "name": "c", "nodes": []}
    )
    assert isinstance(node, Collection)
    assert version == "0.9.dev3"
    assert node.nodes == []


def test_unknown_node_type_parses_as_generic_node():
    node, _ = node_from_ome_dict({"type": "mobie:spots_table", "name": "spots"})
    assert type(node) is Node
    assert node.type == "mobie:spots_table"


def test_unknown_keys_round_trip_through_extra():
    document = {
        "type": "collection",
        "name": "c",
        "nodes": [],
        "future-key": {"nested": [1, None, "x"]},
    }
    node, _ = node_from_ome_dict(document)
    assert node.extra == {"future-key": {"nested": [1, None, "x"]}}
    assert node_to_ome_dict(node) == document


def test_attributes_are_opaque_and_deep_copied():
    attributes = {"mobie:grid": "true", "nested": {"value": None}}
    node, _ = node_from_ome_dict(
        {"type": "collection", "name": "c", "nodes": [], "attributes": attributes}
    )
    assert node.attributes == attributes
    assert node.attributes is not attributes
    node.attributes["nested"]["value"] = 1
    assert attributes["nested"]["value"] is None
    out = node_to_ome_dict(node)
    out["attributes"]["nested"]["value"] = 2
    assert node.attributes["nested"]["value"] == 1


def test_version_is_not_stamped_on_children():
    root = Collection("c", nodes=[Collection("sub", nodes=[])])
    document = node_to_ome_dict(root, version="0.9.dev3")
    assert document["version"] == "0.9.dev3"
    assert "version" not in document["nodes"][0]


def test_parser_reports_malformed_core_keys():
    with pytest.raises(ValueError, match="'type'"):
        node_from_ome_dict({"name": "c"})
    with pytest.raises(ValueError, match="'name'"):
        node_from_ome_dict({"type": "collection"})
    with pytest.raises(ValueError, match="'id'"):
        node_from_ome_dict({"type": "collection", "name": "c", "id": 7})
    with pytest.raises(ValueError, match="ome.nodes\\[0\\]"):
        node_from_ome_dict(
            {"type": "collection", "name": "c", "nodes": [{"name": "x"}]}
        )
    with pytest.raises(ValueError, match="'path'"):
        node_from_ome_dict({"type": "collection", "name": "c", "path": "./x.json"})
    with pytest.raises(ValueError, match="'version'"):
        node_from_ome_dict({"type": "collection", "name": "c", "version": 9})


@pytest.mark.parametrize("fixture", FIXTURE_FILES)
def test_fixture_documents_round_trip(fixture):
    document = json.loads((FIXTURES / fixture).read_text())
    ome = _ome_value(document)
    node, version = node_from_ome_dict(ome)
    assert version == "0.9.dev3"
    assert node_to_ome_dict(node, version=version) == ome
