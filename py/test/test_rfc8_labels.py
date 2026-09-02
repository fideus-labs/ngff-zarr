# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 labels attribute: model, accessors, round trips and rules."""

import pytest
from ngff_zarr import Collection, Node, OmePath, Reference
from ngff_zarr.rfc8 import (
    LabelAttributes,
    Labels,
    is_label_map,
    labels,
    node_from_ome_dict,
    node_to_ome_dict,
    set_labels,
    validate_collection,
)
from ngff_zarr.structural_validation import ValidationError


def _label_map() -> Node:
    node = Node(
        type="multiscale",
        name="nuclei",
        path=OmePath("zarr", "./nuclei"),
    )
    set_labels(
        node,
        Labels(
            labelAttributes=[
                LabelAttributes(labelValue=1, color=[255, 0, 0, 255]),
                LabelAttributes(labelValue=2, extra={"ngio:class": "nucleus"}),
            ],
            source=[Reference("raw")],
        ),
    )
    return node


def test_labels_round_trip_through_the_document():
    node = _label_map()
    serialized = node.attributes["labels"]
    assert serialized["labelAttributes"][0] == {
        "labelValue": 1,
        "color": [255, 0, 0, 255],
    }
    assert serialized["labelAttributes"][1] == {
        "labelValue": 2,
        "ngio:class": "nucleus",
    }
    assert serialized["source"] == [{"id": "raw"}]

    parsed, _ = node_from_ome_dict(node_to_ome_dict(node))
    assert is_label_map(parsed)
    value = labels(parsed)
    assert value.labelAttributes[0].color == [255, 0, 0, 255]
    assert value.labelAttributes[1].extra == {"ngio:class": "nucleus"}
    assert value.source == [Reference("raw")]


def test_empty_labels_object_denotes_a_label_map():
    node = Node(type="multiscale", name="seg", path=OmePath("zarr", "./seg"))
    assert not is_label_map(node)
    assert labels(node) is None
    set_labels(node, Labels())
    assert node.attributes["labels"] == {}
    assert is_label_map(node)
    value = labels(node)
    assert value == Labels()

    set_labels(node, None)
    assert not is_label_map(node)


def test_external_source_reference_round_trips():
    node = Node(type="multiscale", name="seg", path=OmePath("zarr", "./seg"))
    set_labels(
        node,
        Labels(
            source=[Reference("em", OmePath("zarr", "https://example.com/em.zarr"))]
        ),
    )
    value = labels(node)
    assert value.source[0].path.path == "https://example.com/em.zarr"


def test_m_n_label_relations_validate():
    # Two label maps annotating the same two images: m:n via id references.
    images = [
        Node(
            type="multiscale",
            name=f"img{i}",
            id=f"img{i}",
            path=OmePath("zarr", f"./img{i}"),
        )
        for i in (0, 1)
    ]
    label_maps = []
    for i in (0, 1):
        node = Node(
            type="multiscale",
            name=f"seg{i}",
            id=f"seg{i}",
            path=OmePath("zarr", f"./seg{i}"),
        )
        set_labels(node, Labels(source=[Reference("img0"), Reference("img1")]))
        label_maps.append(node)
    collection = Collection("study", nodes=[*images, *label_maps])
    validate_collection(collection, version="0.9.dev3")


def test_label_rules_fire():
    node = Node(type="multiscale", name="seg", path=OmePath("zarr", "./seg"))
    node.attributes = {"labels": {"labelAttributes": [{"color": [1, 2, 3, 4]}]}}
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(node, version="0.9.dev3")
    assert exc_info.value.rule.value == "label-value-required"

    node.attributes = {
        "labels": {"labelAttributes": [{"labelValue": 1, "color": [1, 2, 3]}]}
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(node, version="0.9.dev3")
    assert exc_info.value.rule.value == "label-color-format"

    node.attributes = {"labels": {"source": [{"id": "elsewhere"}]}}
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(node, version="0.9.dev3")
    assert exc_info.value.rule.value == "reference-path-required"


def test_a_null_labels_value_is_rejected():
    node = Node(type="multiscale", name="nuclei", nodes=[])
    node.attributes = {"labels": None}
    assert is_label_map(node)
    with pytest.raises(ValueError, match="must be an object"):
        labels(node)


def test_extra_keys_do_not_override_typed_fields():
    node = Node(type="multiscale", name="nuclei", nodes=[])
    set_labels(
        node,
        Labels(
            labelAttributes=[
                LabelAttributes(
                    labelValue=1,
                    color=[255, 0, 0, 255],
                    extra={"labelValue": 99, "myorg:note": "kept"},
                )
            ]
        ),
    )
    serialized = node.attributes["labels"]["labelAttributes"][0]
    assert serialized["labelValue"] == 1
    assert serialized["color"] == [255, 0, 0, 255]
    assert serialized["myorg:note"] == "kept"
