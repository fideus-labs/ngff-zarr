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


@pytest.mark.parametrize("model", ["plate", "scene"])
def test_a_group_read_as_a_kind_is_checked_against_its_schema(model):
    attributes = {"ome": {"version": "0.6"}}
    assert schema_problems(attributes, "0.6") == []
    assert schema_problems(attributes, "0.6", model=model)


def test_a_kind_its_version_does_not_define_is_a_problem():
    assert schema_problems({"ome": {"version": "0.5"}}, "0.5", model="scene")
