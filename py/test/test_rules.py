# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The rule table, ngff_zarr/rules.json, which both ports read."""

import pytest
from ngff_zarr._supported_versions import SUPPORTED_VERSIONS
from ngff_zarr.rules import RULES
from ngff_zarr.validate import group_problems, schema_problems


@pytest.mark.parametrize("code", sorted(RULES))
def test_a_rule_fires_on_its_example(code):
    rule = RULES[code]
    example = rule["example"]
    version = example["ome"]["version"]
    assert schema_problems(example, version) == [], "the example must pass the schema"
    assert code in {problem.code for problem in group_problems(example, version)}


@pytest.mark.parametrize("code", sorted(RULES))
def test_a_rule_is_well_formed(code):
    rule = RULES[code]
    assert rule["code"] == code
    assert rule["level"] in ("MUST", "SHOULD")
    assert set(rule["versions"]) <= {str(v) for v in SUPPORTED_VERSIONS}
    assert rule["spec"].startswith("https://")


def test_a_rule_reaches_the_transformations_of_a_by_dimension():
    example = RULES["NGFF-CT-027"]["example"]
    scene = example["ome"]["scene"]
    rotation = scene["coordinateTransformations"][0]
    nested = {
        "ome": {
            **example["ome"],
            "scene": {
                **scene,
                "coordinateTransformations": [
                    {
                        "input": rotation["input"],
                        "output": rotation["output"],
                        "type": "byDimension",
                        "transformations": [
                            {
                                "inputAxes": [0, 1],
                                "outputAxes": [0, 1],
                                "transformation": {
                                    "type": "rotation",
                                    "rotation": rotation["rotation"],
                                },
                            }
                        ],
                    }
                ],
            },
        }
    }
    assert schema_problems(nested, "0.6") == []
    assert "NGFF-CT-027" in {problem.code for problem in group_problems(nested, "0.6")}


@pytest.mark.parametrize("model", ["plate", "scene"])
def test_a_group_read_as_a_kind_is_checked_against_its_schema(model):
    attributes = {"ome": {"version": "0.6"}}
    assert schema_problems(attributes, "0.6") == []
    assert schema_problems(attributes, "0.6", model=model)
