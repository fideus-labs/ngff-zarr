# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 collection reading, writing, path resolution and lazy loading."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ngff_zarr import (
    Collection,
    NgffCollection,
    NgffMultiscales,
    Node,
    OmePath,
    Reference,
    from_collection_json,
    from_collection_zarr,
    from_ome_zarr,
    to_collection_json,
    to_collection_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
    upgrade_ome_zarr,
)
from ngff_zarr.structural_validation import ValidationError

FIXTURES = Path(__file__).parent / "fixtures" / "rfc8"


def _image_multiscales():
    image = to_ngff_image(np.arange(16, dtype=np.uint8).reshape(4, 4), dims=("y", "x"))
    return to_multiscales(image, scale_factors=[])


def _study_collection() -> Collection:
    return Collection(
        "study",
        id="study",
        attributes={"mobie:grid": "true"},
        nodes=[
            Node(
                type="multiscale",
                name="raw",
                id="raw",
                path=OmePath("zarr", "./raw"),
            ),
            Collection(
                "nested",
                path=OmePath("json", "./nested_collection.json"),
            ),
        ],
    )


def test_zarr_round_trip(tmp_path):
    store = tmp_path / "collection.ome.zarr"
    root = _study_collection()
    to_collection_zarr(store, root, root_attributes={"lab": "fideus"})

    document = json.loads((store / "zarr.json").read_text())
    assert document["zarr_format"] == 3
    assert document["node_type"] == "group"
    assert document["attributes"]["ome"]["version"] == "0.9.dev3"
    assert document["attributes"]["lab"] == "fideus"

    collection = from_collection_zarr(store, validate=True)
    assert collection.root == root
    assert collection.version == "0.9.dev3"
    assert collection.base == str(store.resolve())
    assert collection.root_attributes == {"lab": "fideus"}


def test_overwrite_never_removes_the_tree_beneath_the_root(tmp_path):
    store = tmp_path / "collection.ome.zarr"
    to_collection_zarr(store, _study_collection())
    to_ome_zarr(store / "raw", _image_multiscales(), version="0.9.dev1")

    to_collection_zarr(store, _study_collection(), overwrite=True)
    assert (store / "raw" / "zarr.json").exists()
    loaded = from_collection_zarr(store).load("raw")
    assert isinstance(loaded, NgffMultiscales)


def test_collection_writer_accepts_only_dev3(tmp_path):
    with pytest.raises(ValueError, match="0.9.dev3"):
        to_collection_zarr(
            tmp_path / "c.ome.zarr", _study_collection(), version="0.9.dev1"
        )


def test_collection_writer_validates_by_default(tmp_path):
    invalid = Collection("c")  # neither nodes nor path
    with pytest.raises(ValidationError):
        to_collection_zarr(tmp_path / "c.ome.zarr", invalid)
    to_collection_zarr(tmp_path / "c.ome.zarr", invalid, validate=False)


def test_json_round_trip(tmp_path):
    root = _study_collection()
    document = to_collection_json(root)
    assert document["ome"]["version"] == "0.9.dev3"

    path = tmp_path / "collection.json"
    to_collection_json(root, path)
    collection = from_collection_json(path, validate=True)
    assert collection.root == root
    assert collection.base == str(tmp_path.resolve())

    from_mapping = from_collection_json(document)
    assert from_mapping.root == root
    assert from_mapping.base is None


def test_from_ome_zarr_names_the_collection_reader(tmp_path):
    store = tmp_path / "collection.ome.zarr"
    to_collection_zarr(store, _study_collection())
    with pytest.raises(ValueError, match="from_collection_zarr"):
        from_ome_zarr(store)


def test_from_collection_zarr_names_the_image_reader(tmp_path):
    store = tmp_path / "image.ome.zarr"
    to_ome_zarr(store, _image_multiscales(), version="0.5")
    with pytest.raises(ValueError, match="from_ome_zarr"):
        from_collection_zarr(store)


def test_to_ome_zarr_refuses_dev3_with_guidance(tmp_path):
    with pytest.raises(ValueError, match="to_collection_zarr"):
        to_ome_zarr(
            tmp_path / "image.ome.zarr", _image_multiscales(), version="0.9.dev3"
        )


def test_upgrade_does_not_offer_dev3(tmp_path):
    store = tmp_path / "image.ome.zarr"
    to_ome_zarr(store, _image_multiscales(), version="0.5")
    with pytest.raises(ValueError):
        upgrade_ome_zarr(store, tmp_path / "out.ome.zarr", version="0.9.dev3")


def test_load_resolves_inline_and_referenced_nodes(tmp_path):
    store = tmp_path / "collection.ome.zarr"
    root = _study_collection()
    to_collection_zarr(store, root)
    to_ome_zarr(store / "raw", _image_multiscales(), version="0.9.dev1")
    to_collection_json(Collection("nested", nodes=[]), store / "nested_collection.json")

    collection = from_collection_zarr(store)

    image = collection.load("raw")
    assert isinstance(image, NgffMultiscales)
    assert image.images[0].data.shape == (4, 4)
    # Loaded documents are cached per resolved location.
    assert collection.load("raw") is image
    assert collection.load(Reference("raw")) is image

    nested = collection.load("nested")
    assert isinstance(nested, NgffCollection)
    assert nested.base == str((store / "nested_collection.json").parent.resolve())


def test_external_collection_fixture_layout(tmp_path):
    # The upstream external_collection example: a parent document whose
    # sub-collection lives in a sibling JSON file.
    for name in ("parent_collection.json", "child_collection.json"):
        shutil.copy(FIXTURES / "external_collection" / name, tmp_path / name)

    parent = from_collection_json(tmp_path / "parent_collection.json")
    child = parent.load("sub-collection")
    assert isinstance(child, NgffCollection)
    assert child.root.name == "sub-collection"
    assert child.root.nodes[0].name == "image-label"
    # The child collection resolves its own relative paths from its file.
    assert child.base == str(tmp_path.resolve())


def test_external_reference_with_undeclared_id_raises(tmp_path):
    to_collection_json(
        Collection("child", nodes=[Node(type="multiscale", name="img", id="img")]),
        tmp_path / "child.json",
    )
    parent = NgffCollection(root=Collection("parent", nodes=[]), base=str(tmp_path))
    reference = Reference("missing", OmePath("json", "./child.json"))
    with pytest.raises(KeyError, match="missing"):
        parent.load(reference)


def test_in_memory_collection_cannot_resolve_relative_paths():
    collection = from_collection_json(
        {
            "ome": {
                "version": "0.9.dev3",
                "type": "collection",
                "name": "c",
                "nodes": [
                    {
                        "type": "collection",
                        "name": "n",
                        "path": {"type": "json", "path": "./other.json"},
                    }
                ],
            }
        }
    )
    with pytest.raises(ValueError, match="resolution base"):
        collection.load("n")


def test_lru_cache_requires_a_positive_size():
    from ngff_zarr._lru_cache import LRUCache

    with pytest.raises(ValueError, match="positive"):
        LRUCache(0)


def test_node_lookup_prefers_ids(tmp_path):
    collection = NgffCollection(
        root=Collection(
            "c",
            nodes=[
                Node(type="multiscale", name="a", id="b"),
                Node(type="multiscale", name="b", id="c"),
            ],
        )
    )
    assert collection.node("b").name == "a"
    assert collection.node("a").name == "a"
    with pytest.raises(KeyError):
        collection.node("missing")


def test_collection_reader_never_uses_the_versioned_schema_pass(tmp_path, monkeypatch):
    # The vendored spec/0.9 tree enumerates only 0.9.dev1, so routing a
    # 0.9.dev3 document through ngff_zarr.validate would fail on the version
    # enum with a misleading error; the RFC-8 reader must not call it.
    import importlib

    validate_module = importlib.import_module("ngff_zarr.validate")

    def _forbidden(*args, **kwargs):
        raise AssertionError("validate_ngff must not see RFC-8 documents")

    monkeypatch.setattr(validate_module, "validate", _forbidden)
    store = tmp_path / "collection.ome.zarr"
    to_collection_zarr(store, _study_collection())
    pytest.importorskip("jsonschema")
    from_collection_zarr(store, validate=True)


def test_validate_true_runs_the_bundled_node_schema(tmp_path):
    pytest.importorskip("jsonschema")
    import jsonschema

    store = tmp_path / "collection.ome.zarr"
    to_collection_zarr(store, _study_collection())
    # Corrupt the document below the structural rules' radar: a non-object
    # ``attributes`` passes no structural rule but fails the schema.
    document = json.loads((store / "zarr.json").read_text())
    document["attributes"]["ome"]["attributes"] = "not-an-object"
    (store / "zarr.json").write_text(json.dumps(document))
    with pytest.raises((jsonschema.ValidationError, ValueError)):
        from_collection_zarr(store, validate=True)
