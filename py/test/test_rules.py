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
