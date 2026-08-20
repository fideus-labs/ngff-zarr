# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the single place that locates the axes in raw metadata.

``_raw_axes`` is what every reader and the conformance report use to find the
axes of a ``multiscales`` entry, so the version-dependent location is decided
once. The v0.6 case is the one that regressed: the axes moved into the
intrinsic coordinate system, and code that looked only for a flat ``axes`` key
silently found none.
"""

from ngff_zarr.parse_metadata import _raw_axes

_AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]


def test_flat_axes_are_returned():
    assert _raw_axes({"axes": _AXES}) == _AXES


def test_v06_axes_come_from_the_intrinsic_coordinate_system():
    entry = {"coordinateSystems": [{"name": "intrinsic", "axes": _AXES}]}
    assert _raw_axes(entry) == _AXES


def test_a_flat_list_wins_over_coordinate_systems():
    """A document carrying both is read as the flat, pre-v0.6 shape."""
    entry = {
        "axes": _AXES,
        "coordinateSystems": [{"name": "other", "axes": [{"name": "t"}]}],
    }
    assert _raw_axes(entry) == _AXES


def test_entries_without_axes_yield_an_empty_list():
    assert _raw_axes({"datasets": [{"path": "0"}]}) == []
    assert _raw_axes({"coordinateSystems": []}) == []
    assert _raw_axes({"coordinateSystems": [{"name": "intrinsic"}]}) == []
    assert _raw_axes({"axes": "not a list"}) == []
    assert _raw_axes("not an entry") == []


def test_axis_entries_are_returned_verbatim():
    """Malformed entries are passed through for the caller to classify."""
    entry = {"axes": ["not an object", {"name": 42}]}
    assert _raw_axes(entry) == ["not an object", {"name": 42}]


def test_the_intrinsic_system_is_resolved_by_name_not_position():
    """Nothing requires the intrinsic system to be listed first.

    The datasets name the system they map into through their transformation
    ``output``; reading ``coordinateSystems[0]`` would report another system's
    axes, so an invalid intrinsic orientation could pass and an unrelated one
    be reported instead.
    """
    entry = {
        "coordinateSystems": [
            {"name": "physical", "axes": [{"name": "t", "type": "time"}]},
            {"name": "intrinsic", "axes": _AXES},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {
                        "input": {"path": "0"},
                        "output": {"name": "intrinsic"},
                        "type": "scale",
                        "scale": [1.0, 1.0],
                    }
                ],
            }
        ],
    }
    assert _raw_axes(entry) == _AXES


def test_an_unresolvable_output_falls_back_to_the_first_system():
    """Both ports write the intrinsic system first, so it is the safe default."""
    entry = {
        "coordinateSystems": [{"name": "intrinsic", "axes": _AXES}],
        "datasets": [{"path": "0", "coordinateTransformations": []}],
    }
    assert _raw_axes(entry) == _AXES

    named_elsewhere = {
        "coordinateSystems": [{"name": "intrinsic", "axes": _AXES}],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [{"output": {"name": "absent"}}],
            }
        ],
    }
    assert _raw_axes(named_elsewhere) == _AXES
