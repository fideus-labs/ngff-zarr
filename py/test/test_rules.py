# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The rule table, ngff_zarr/rules.json, which both ports read."""

import pytest
from ngff_zarr.rules import RULES
from ngff_zarr.validate import group_problems, schema_problems

EXAMPLES = [
    (code, example)
    for code, rule in sorted(RULES.items())
    for example in rule["examples"]
]


@pytest.mark.parametrize(("code", "example"), EXAMPLES)
def test_a_rule_fires_on_its_examples(code, example):
    version = example["ome"]["version"]
    assert schema_problems(example, version) == [], "an example must pass the schema"
    assert code in {problem.code for problem in group_problems(example, version)}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("coordinateSystems", None),
        ("coordinateSystems", [{}]),
        ("coordinateSystems", [None]),
        ("coordinateTransformations", None),
        ("coordinateTransformations", [None]),
        ("coordinateTransformations", [{"type": "sequence", "transformations": None}]),
        ("datasets", None),
        ("datasets", [None]),
    ],
)
def test_what_the_schema_leaves_open_does_not_break_the_rules(key, value):
    # ``ome`` is open to properties the schema does not define; the rules check
    # only what has the shape they expect, and still report the scene's finding.
    example = RULES["NGFF-CT-015"]["examples"][0]
    attributes = {"ome": {**example["ome"], key: value}}
    assert schema_problems(attributes, "0.6") == []
    assert [problem.code for problem in group_problems(attributes, "0.6")] == [
        "NGFF-CT-015"
    ]


@pytest.mark.parametrize(
    ("attributes", "version"),
    [({"series": 42}, "0.4"), ({"ome": {"version": "0.5", "series": 42}}, "0.5")],
)
def test_series_metadata_is_checked_against_its_schema(attributes, version):
    assert schema_problems(attributes, version)


@pytest.mark.parametrize("model", ["plate", "scene"])
def test_a_group_read_as_a_kind_is_checked_against_its_schema(model):
    attributes = {"ome": {"version": "0.6"}}
    assert schema_problems(attributes, "0.6") == []
    assert schema_problems(attributes, "0.6", model=model)


def test_a_kind_its_version_does_not_define_is_a_problem():
    assert schema_problems({"ome": {"version": "0.5"}}, "0.5", model="scene")
