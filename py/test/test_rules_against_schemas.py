# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Where the structural rules and the bundled JSON Schemas overlap.

:mod:`ngff_zarr.structural_validation` carries the spec MUSTs no JSON Schema
states. Three of its rules do restate, or look like they restate, something a
schema declares, and the module docstring says why each is kept. These tests
measure that relationship instead of trusting the prose: if upstream tightens a
schema, or the bounds in a rule drift from it, the claim fails here rather than
quietly becoming false.
"""

import json

import pytest
from ngff_zarr.structural_validation import (
    SpecRule,
    ValidationError,
    validate_axis_names_unique,
    validate_structural,
)
from ngff_zarr.v04.zarr_metadata import Axis, Dataset, Metadata, Scale
from ngff_zarr.validate import _schemas_dir

jsonschema = pytest.importorskip("jsonschema")

#: The versions whose schemas keep the axes definition inside ``image.schema``.
FLAT_VERSIONS = ["0.4", "0.5"]


def _axes_constraints(version: str) -> dict:
    """The keywords the bundled schema puts on the axes array."""
    image = json.loads(_schemas_dir(version).joinpath("image.schema").read_text())
    return image["$defs"]["axes"]


def _validate(document: dict, version: str = "0.4") -> list:
    from jsonschema import Draft202012Validator
    from ngff_zarr.validate import _schema_registry, load_schema

    validator = Draft202012Validator(
        load_schema(version=version), registry=_schema_registry(version)
    )
    return list(validator.iter_errors(document))


def _image_document(axes: list[dict], version: str = "0.4") -> dict:
    return {
        "multiscales": [
            {
                "version": version,
                "name": "image",
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0] * len(axes)}
                        ],
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize("version", FLAT_VERSIONS)
def test_axis_count_restates_the_schema_bounds_at_0_4_and_0_5(version):
    """The rule's 2..5 is the schema's own ``minItems``/``maxItems``."""
    constraints = _axes_constraints(version)

    assert constraints["minItems"] == 2
    assert constraints["maxItems"] == 5


def test_axis_count_is_stricter_than_the_0_6_schema_at_the_low_end():
    """0.6 relaxes the floor to one axis; the rule holds it at two."""
    axes = json.loads(_schemas_dir("0.6").joinpath("axes.schema").read_text())

    assert axes["minItems"] == 1


@pytest.mark.parametrize("version", FLAT_VERSIONS)
def test_the_axes_array_carries_unique_items(version):
    assert _axes_constraints(version)["uniqueItems"] is True


def test_unique_items_does_not_catch_two_axes_sharing_a_name():
    """``uniqueItems`` compares whole objects, so a shared name gets through."""
    axes = [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "millimeter"},
    ]

    assert _validate(_image_document(axes)) == []


def test_axis_names_unique_refuses_what_unique_items_accepts():
    """The same document the schema accepts is refused by the rule."""
    metadata = Metadata(
        axes=[
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="y", type="space", unit="millimeter"),
        ],
        datasets=[Dataset(path="0", coordinateTransformations=[Scale([1.0, 1.0])])],
        coordinateTransformations=None,
    )

    with pytest.raises(ValidationError) as excinfo:
        validate_axis_names_unique(metadata)
    assert excinfo.value.rule == SpecRule.AXIS_NAMES_UNIQUE


def test_a_channel_and_a_space_axis_may_not_share_a_name():
    """The case no schema and no other rule covers, at any version.

    Every other axis rule passes: one channel axis, the space names are the
    canonical ``(y, x)`` suffix, and the type order is channel before space.
    """
    axes = [
        {"name": "y", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    assert _validate(_image_document(axes), "0.4") == []

    metadata = Metadata(
        axes=[
            Axis(name="y", type="channel"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ],
        datasets=[
            Dataset(path="0", coordinateTransformations=[Scale([1.0, 1.0, 1.0])])
        ],
        coordinateTransformations=None,
    )

    with pytest.raises(ValidationError) as excinfo:
        validate_structural(metadata)
    assert excinfo.value.rule == SpecRule.AXIS_NAMES_UNIQUE
    assert excinfo.value.location == "multiscales[0].axes[1]"


def _omero_channel_properties(version: str) -> dict:
    """The channel object the bundled schema declares, wherever it lives."""
    image = json.loads(_schemas_dir(version).joinpath("image.schema").read_text())

    found = []

    def walk(node):
        if isinstance(node, dict):
            channels = node.get("channels")
            if isinstance(channels, dict) and isinstance(channels.get("items"), dict):
                found.append(channels["items"].get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(image)
    assert found, f"no omero channels definition in the {version} image schema"
    return found[0]


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_no_schema_constrains_the_omero_channel_color_format(version):
    """`omero-channel-color-format` has no schema counterpart to defer to."""
    color = _omero_channel_properties(version).get("color")

    assert color is not None, f"the {version} schema declares no channel color"
    assert "pattern" not in color, color
    assert "enum" not in color, color
