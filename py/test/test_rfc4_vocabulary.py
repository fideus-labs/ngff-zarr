# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The RFC 4 vocabulary is written down more than once; pin the copies together.

Three independent transcriptions of the 24 anatomical orientation values ship
in the package: the public :class:`AnatomicalOrientationValues` enum, the
antonym-pair table the validator checks against, and the enum inside the
bundled RFC 4 schema. Nothing derives one from another, so these tests are what
keeps them from drifting: a value added to one and not the others would let
``to_ngff_zarr`` write an orientation that ``from_ngff_zarr(validate=True)``
then rejects.
"""

from ngff_zarr.rfc4 import AnatomicalOrientationValues
from ngff_zarr.rfc4_validation import _ANATOMICAL_AXIS_OF, load_rfc4_orientation_schema

_EXPECTED_COUNT = 24


def _schema_values() -> set[str]:
    schema = load_rfc4_orientation_schema()
    return set(schema["$defs"]["AnatomicalOrientationValues"]["enum"])


def test_the_three_copies_hold_the_same_values():
    enum_values = {member.value for member in AnatomicalOrientationValues}
    assert enum_values == set(_ANATOMICAL_AXIS_OF)
    assert enum_values == _schema_values()


def test_the_vocabulary_has_not_changed_size():
    """A term added upstream must be added deliberately, in every copy."""
    assert len(AnatomicalOrientationValues) == _EXPECTED_COUNT
    assert len(_ANATOMICAL_AXIS_OF) == _EXPECTED_COUNT
    assert len(_schema_values()) == _EXPECTED_COUNT


def test_every_anatomical_axis_carries_exactly_two_directions():
    """The table collapses antonyms in pairs, which the mutual-exclusion rule
    relies on: an axis key reached by one value only would make that value
    impossible to conflict with, and one reached by three would reject a legal
    pair of axes."""
    per_axis: dict[str, list[str]] = {}
    for value, axis in _ANATOMICAL_AXIS_OF.items():
        per_axis.setdefault(axis, []).append(value)
    assert {axis: len(values) for axis, values in per_axis.items() if len(values) != 2} == {}
    assert len(per_axis) == _EXPECTED_COUNT // 2
