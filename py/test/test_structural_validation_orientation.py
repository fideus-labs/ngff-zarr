# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Unit tests for the RFC 4 orientation rule in :mod:`ngff_zarr.structural_validation`.

Pass/fail tests for
:func:`~ngff_zarr.structural_validation.validate_axis_orientation`, the wrapper
that surfaces the package's RFC 4 anatomical-orientation checks through the
unified :class:`~ngff_zarr.structural_validation.ValidationError` channel.

Coverage:

* no spatial axis carries orientation -> no-op (the heavy ``jsonschema`` import is
  never triggered);
* a fully specified, consistent orientation passes for *both* stored shapes -- a
  raw ``dict`` (image read path) and an
  :class:`~ngff_zarr.rfc4.AnatomicalOrientation` dataclass (write path);
* a non-anatomical orientation ``type`` ->
  :attr:`SpecRule.AXIS_ORIENTATION_ANATOMICAL_TYPE`;
* orientation on only some spatial axes -> accepted (RFC 4 makes orientation
  optional per axis).

Each failing case asserts the expected :class:`SpecRule` and the
``multiscales[0].axes`` location.
"""

import pytest
from ngff_zarr.rfc4 import AnatomicalOrientation, AnatomicalOrientationValues
from ngff_zarr.structural_validation import (
    SpecRule,
    ValidationError,
    validate_axis_orientation,
)
from ngff_zarr.v04.zarr_metadata import Axis, Dataset, Metadata, Scale


def _metadata_with_axes(axes: list[Axis]) -> Metadata:
    """Wrap the given axes in an otherwise-valid single-level metadata.

    :func:`validate_axis_orientation` inspects only the axes, so the dataset's
    ``scale`` is a placeholder sized to the axis count.
    """
    return Metadata(
        axes=axes,
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[Scale([1.0] * len(axes))],
            )
        ],
        coordinateTransformations=None,
    )


def _orientation_dict(value: str) -> dict:
    """Render an anatomical orientation as the raw dict an image read produces."""
    return {"type": "anatomical", "value": value}


# Canonical (z, y, x) LPS orientation values, in axis order.
_LPS_VALUES = ("inferior-to-superior", "anterior-to-posterior", "right-to-left")
_LPS_ENUM = (
    AnatomicalOrientationValues.inferior_to_superior,
    AnatomicalOrientationValues.anterior_to_posterior,
    AnatomicalOrientationValues.right_to_left,
)


def test_axis_orientation_no_orientation_is_noop():
    # No spatial axis declares orientation: the rule is a no-op (no raise).
    metadata = _metadata_with_axes(
        [
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ]
    )
    validate_axis_orientation(metadata)


@pytest.mark.parametrize("shape", ["dict", "dataclass"])
def test_axis_orientation_valid_consistent(shape):
    if shape == "dict":
        orientations = [_orientation_dict(v) for v in _LPS_VALUES]
    else:
        orientations = [AnatomicalOrientation(value=v) for v in _LPS_ENUM]
    metadata = _metadata_with_axes(
        [
            Axis(name="z", type="space", orientation=orientations[0]),
            Axis(name="y", type="space", orientation=orientations[1]),
            Axis(name="x", type="space", orientation=orientations[2]),
        ]
    )
    validate_axis_orientation(metadata)


def test_axis_orientation_non_anatomical_type():
    # The "y" axis declares a different orientation type than its siblings.
    metadata = _metadata_with_axes(
        [
            Axis(name="z", type="space", orientation=_orientation_dict(_LPS_VALUES[0])),
            Axis(
                name="y",
                type="space",
                orientation={"type": "other", "value": _LPS_VALUES[1]},
            ),
            Axis(name="x", type="space", orientation=_orientation_dict(_LPS_VALUES[2])),
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_axis_orientation(metadata)
    assert exc_info.value.rule == SpecRule.AXIS_ORIENTATION_ANATOMICAL_TYPE
    assert exc_info.value.location == "multiscales[0].axes"


def test_axis_orientation_empty_object_is_undefined():
    """An empty ``orientation`` object is undefined, whatever the axis type.

    Guards the Axis-to-dict rendering: emitting ``{}`` as a populated
    ``{"type": None, "value": None}`` dict would read back as a real orientation
    and trip the non-space or anatomical-type rule.
    """
    # On a non-spatial axis: not a use of orientation, so not a violation.
    validate_axis_orientation(
        _metadata_with_axes(
            [
                Axis(name="t", type="time", orientation={}),
                Axis(name="y", type="space"),
                Axis(name="x", type="space"),
            ]
        )
    )
    # On a spatial axis: that axis is simply left undefined.
    validate_axis_orientation(
        _metadata_with_axes(
            [
                Axis(name="y", type="space", orientation={}),
                Axis(name="x", type="space"),
            ]
        )
    )


def test_axis_orientation_partial_is_accepted():
    # Orientation is defined for "z" and "y" and left undefined on "x". RFC 4
    # makes orientation optional per spatial axis, so this is not a violation.
    metadata = _metadata_with_axes(
        [
            Axis(name="z", type="space", orientation=_orientation_dict(_LPS_VALUES[0])),
            Axis(name="y", type="space", orientation=_orientation_dict(_LPS_VALUES[1])),
            Axis(name="x", type="space"),
        ]
    )
    validate_axis_orientation(metadata)


def test_axis_orientation_on_non_space_axis():
    # Orientation is (illegally) declared on the non-spatial "t" axis. The
    # spatial axes carry it too, so the RFC 4 check is actually reached.
    metadata = _metadata_with_axes(
        [
            Axis(
                name="t",
                type="time",
                orientation=_orientation_dict("inferior-to-superior"),
            ),
            Axis(name="z", type="space", orientation=_orientation_dict(_LPS_VALUES[0])),
            Axis(name="y", type="space", orientation=_orientation_dict(_LPS_VALUES[1])),
            Axis(name="x", type="space", orientation=_orientation_dict(_LPS_VALUES[2])),
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_axis_orientation(metadata)
    assert exc_info.value.rule == SpecRule.AXIS_ORIENTATION_ON_NON_SPACE
    assert exc_info.value.location == "multiscales[0].axes"


def test_axis_orientation_duplicate_anatomical_axis():
    # "z" and "x" both lie on the left-right anatomical axis (mutually exclusive).
    metadata = _metadata_with_axes(
        [
            Axis(
                name="z", type="space", orientation=_orientation_dict("left-to-right")
            ),
            Axis(name="y", type="space", orientation=_orientation_dict(_LPS_VALUES[1])),
            Axis(
                name="x", type="space", orientation=_orientation_dict("right-to-left")
            ),
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_axis_orientation(metadata)
    assert exc_info.value.rule == SpecRule.AXIS_ORIENTATION_UNIQUE_AXIS
    assert exc_info.value.location == "multiscales[0].axes"


def test_rfc4_orientation_messages_carry_mapping_markers():
    """Pin the substrings :func:`validate_axis_orientation` maps onto rules.

    The wrapper distinguishes the RFC 4 orientation failures by searching
    :func:`~ngff_zarr.rfc4_validation.validate_rfc4_orientation`'s message text
    for ``"must be anatomical"``, ``"non-space axes"`` and
    ``"same anatomical axis"``. Those markers are a load-bearing contract: if the
    wording is edited away, the wrapper silently falls through to a bare
    ``ValueError`` instead of raising the mapped :class:`ValidationError`.
    Asserting the markers here makes such an edit fail CI loudly at the source
    rather than silently break the mapping.
    """
    from ngff_zarr.rfc4_validation import validate_rfc4_orientation

    # A non-anatomical orientation type -> the marker the type rule maps.
    with pytest.raises(ValueError, match="must be anatomical"):
        validate_rfc4_orientation(
            [
                {
                    "name": "y",
                    "type": "space",
                    "orientation": {
                        "type": "anatomical",
                        "value": "anterior-to-posterior",
                    },
                },
                {
                    "name": "x",
                    "type": "space",
                    "orientation": {"type": "other", "value": "right-to-left"},
                },
            ]
        )

    # Orientation on a non-spatial axis -> the on-non-space marker.
    with pytest.raises(ValueError, match="non-space axes"):
        validate_rfc4_orientation(
            [
                {
                    "name": "t",
                    "type": "time",
                    "orientation": {
                        "type": "anatomical",
                        "value": "inferior-to-superior",
                    },
                },
            ]
        )

    # Two spatial axes on one anatomical axis -> the unique-axis marker.
    with pytest.raises(ValueError, match="same anatomical axis"):
        validate_rfc4_orientation(
            [
                {
                    "name": "y",
                    "type": "space",
                    "orientation": {"type": "anatomical", "value": "left-to-right"},
                },
                {
                    "name": "x",
                    "type": "space",
                    "orientation": {"type": "anatomical", "value": "right-to-left"},
                },
            ]
        )
