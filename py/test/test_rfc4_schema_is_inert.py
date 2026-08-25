# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The bundled RFC 4 orientation schema reports nothing.

``spec/rfc/4/orientation.schema.json`` has no root entry point: every
definition sits in ``$defs`` and nothing references them, so the closing
``validator.validate`` in :func:`ngff_zarr.rfc4_validation.validate_rfc4_orientation`
accepts any JSON object. Upstream ome/ngff#585 and #586 report the same.

Every RFC 4 rule that fires is the hand-written Python in that function. These
tests pin that split so those checks are not later removed as redundant with a
schema pass that does not run, and so the day upstream publishes a schema with
a root entry point is visible as a failure here rather than a silent change.
"""

import pytest
from ngff_zarr.rfc4_validation import load_rfc4_orientation_schema

jsonschema = pytest.importorskip("jsonschema")


def _validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schema = load_rfc4_orientation_schema()
    registry = Registry().with_resource(
        "https://w3id.org/ome/ngff", resource=Resource.from_contents(schema)
    )
    return Draft202012Validator(schema, registry=registry)


def test_the_schema_has_no_root_entry_point():
    schema = load_rfc4_orientation_schema()

    assert "$defs" in schema
    assert "properties" not in schema
    assert "$ref" not in schema
    assert schema.get("additionalProperties") is True


@pytest.mark.parametrize(
    "document",
    [
        {"axes": ["not an object", {"name": 42, "type": "nonsense"}]},
        {"axes": "not even a list"},
        {"axes": [{"orientation": {"type": "bogus", "value": "sideways"}}]},
        {"axes": [{"name": "x", "type": "space", "orientation": []}]},
    ],
    ids=[
        "axis-is-a-string",
        "axes-not-a-list",
        "orientation-off-vocabulary",
        "orientation-is-a-list",
    ],
)
def test_the_bundled_rfc4_schema_checks_nothing(document):
    assert list(_validator().iter_errors(document)) == []


def test_the_hand_written_checks_are_what_reject_a_bad_orientation():
    """The same document the schema accepts is refused by the Python rules."""
    from ngff_zarr.rfc4_validation import validate_rfc4_orientation

    axes = [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "bogus", "value": "left-to-right"},
        },
    ]

    assert list(_validator().iter_errors({"axes": axes})) == []
    with pytest.raises(ValueError, match="must be anatomical"):
        validate_rfc4_orientation(axes)
