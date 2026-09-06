# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 structural rules, driven by the shared cross-language case table.

``rfc8_collection_cases.json`` pins the rule, message and location bytes both
ports must produce; ``ts/test/rfc8_validation_test.ts`` drives the identical
file.
"""

import json
from pathlib import Path

import pytest
from ngff_zarr import (
    Collection,
    Node,
    OmePath,
    ValidateOptions,
    ValidationLevel,
    validate_collection,
)
from ngff_zarr.structural_validation import ValidationError, is_rfc8_node_model

CASES_PATH = Path(__file__).parent / "rfc8_collection_cases.json"
CASES = json.loads(CASES_PATH.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_shared_case(case):
    version = case.get("version")
    if case.get("ok"):
        validate_collection(case["ome"], version=version)
        return
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(case["ome"], version=version)
    error = exc_info.value
    assert error.rule.value == case["rule"]
    assert str(error) == case["message"]
    assert error.location == case["location"]


def test_every_rule_is_exercised_by_the_shared_cases():
    exercised = {case["rule"] for case in CASES if "rule" in case}
    assert exercised == {
        "node-type-required",
        "node-name-required",
        "node-name-unique",
        "node-id-format",
        "node-id-unique",
        "node-nodes-xor-path",
        "path-type-known",
        "coordinate-system-id-required",
        "reference-id-required",
        "reference-path-required",
        "label-value-required",
        "label-color-format",
        "scene-transformations-required",
        "plate-columns-rows-required",
        "well-reference-resolves",
        "acquisition-reference-resolves",
    }


def test_is_rfc8_node_model_is_a_dispatch_predicate():
    # Unlike the RFC-4 gate, ``None`` is not node-shaped: the predicate says
    # which document shape a version stores, and the orchestrator handles
    # the no-version strictness default itself.
    assert is_rfc8_node_model("0.9.dev3")
    assert not is_rfc8_node_model("0.9.dev1")
    assert not is_rfc8_node_model("0.6")
    assert not is_rfc8_node_model(None)


def test_validate_collection_accepts_a_parsed_node_tree():
    root = Collection(
        "c",
        nodes=[
            Node(type="multiscale", name="img"),
            Node(type="multiscale", name="img"),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(root, version="0.9.dev3")
    assert exc_info.value.rule.value == "node-name-unique"
    assert exc_info.value.location == "ome.nodes[1]"


def test_schema_only_level_skips_the_rules():
    root = Collection("c", nodes=None, path=None)
    validate_collection(
        root, options=ValidateOptions(level=ValidationLevel.SCHEMA_ONLY)
    )
    with pytest.raises(ValidationError):
        validate_collection(root)


def test_prefixed_path_types_pass_and_core_types_pass():
    for path_type in ("zarr", "json", "myorg:zip"):
        validate_collection(Collection("c", path=OmePath(path_type, "./somewhere")))


def test_mobie_fixture_fails_on_its_unprefixed_extension_path_type():
    # The upstream example uses the unprefixed extension identifier
    # ``google-sheet``, which the RFC's naming scheme reserves for the core
    # specification; the rule reports it (see fixtures/rfc8/README.md).
    document = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "rfc8"
            / "mobie_image_labels_table.json"
        ).read_text()
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(document["ome"], version="0.9.dev3")
    assert exc_info.value.rule.value == "path-type-known"


def test_remaining_fixture_documents_validate():
    fixtures = Path(__file__).parent / "fixtures" / "rfc8"
    for name in (
        "base_multiscale_collection.json",
        "nested_collection.json",
        "external_collection/parent_collection.json",
        "external_collection/child_collection.json",
        "external_collection/resolved_collection.json",
    ):
        document = json.loads((fixtures / name).read_text())
        validate_collection(document["ome"], version="0.9.dev3")


def test_webknossos_fixture_fails_on_its_draft_reference_shape():
    # The upstream example predates the RFC's v1 Reference interface: its
    # transformation inputs and outputs use a {"type", "$ref"} shape instead
    # of an id reference, which the rule reports (see fixtures/rfc8/README.md).
    webknossos = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "rfc8"
            / "webknossos_inline_multiscale.json"
        ).read_text()
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_collection(webknossos["attributes"]["ome"], version="0.9.dev3")
    assert exc_info.value.rule.value == "reference-id-required"
