# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Unit tests for generalized removal of ``None`` metadata fields (issue #496)."""

from ngff_zarr.to_ngff_zarr import _pop_metadata_optionals, _remove_none_values


def test_remove_none_values_top_level():
    """A top-level ``None`` value is dropped while other keys are kept."""
    assert _remove_none_values({"a": 1, "b": None}) == {"a": 1}


def test_remove_none_values_nested_dict():
    """``None`` values are stripped recursively from nested dicts."""
    obj = {"a": {"b": None, "c": 2}, "d": None}
    assert _remove_none_values(obj) == {"a": {"c": 2}}


def test_remove_none_values_inside_list():
    """``None`` values are stripped from dicts nested inside lists."""
    obj = {"axes": [{"name": "x", "unit": None}, {"name": "c", "unit": "second"}]}
    assert _remove_none_values(obj) == {
        "axes": [{"name": "x"}, {"name": "c", "unit": "second"}]
    }


def test_remove_none_values_preserves_falsy_but_valid():
    """Falsy-but-valid values (``0``, ``""``, ``False``, empties) are kept."""
    # Zero scales/translations and other falsy values must survive.
    obj = {
        "zero_int": 0,
        "zero_float": 0.0,
        "empty_str": "",
        "false": False,
        "empty_list": [],
        "empty_dict": {},
        "scale": [0.0, 1.0, 0.0],
        "drop_me": None,
    }
    result = _remove_none_values(obj)
    assert "drop_me" not in result
    assert result == {
        "zero_int": 0,
        "zero_float": 0.0,
        "empty_str": "",
        "false": False,
        "empty_list": [],
        "empty_dict": {},
        "scale": [0.0, 1.0, 0.0],
    }


def test_remove_none_values_returns_scalars_unchanged():
    """Scalars, including ``None`` itself, pass through unchanged."""
    assert _remove_none_values(5) == 5
    assert _remove_none_values("x") == "x"
    assert _remove_none_values(None) is None


def _sample_metadata_dict():
    """Metadata dict mixing populated, ``None``, and orientation fields."""
    return {
        "axes": [
            {
                "name": "x",
                "type": "space",
                "unit": None,
                "orientation": {"type": "anatomical", "value": "left-to-right"},
            },
            {"name": "c", "type": "channel", "unit": None, "orientation": None},
        ],
        "datasets": [
            {
                "path": "scale0/image",
                "coordinateTransformations": [{"scale": [1.0, 0.0], "type": "scale"}],
            }
        ],
        "coordinateTransformations": None,
        "omero": None,
        "name": "image",
        "version": "0.4",
        "type": None,
        "metadata": None,
    }


def test_pop_optionals_rfc4_disabled_drops_populated_orientation():
    """With RFC 4 disabled, orientation and all ``None`` fields are removed."""
    result = _pop_metadata_optionals(_sample_metadata_dict(), enabled_rfcs=None)
    for axis in result["axes"]:
        assert "orientation" not in axis
        assert "unit" not in axis  # None units are stripped too
    # All None-valued top-level fields are gone.
    assert "coordinateTransformations" not in result
    assert "omero" not in result
    assert "type" not in result
    assert "metadata" not in result
    # Valid data, including a zero scale value, is preserved.
    assert result["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 0.0]


def test_pop_optionals_rfc4_enabled_keeps_populated_drops_null_orientation():
    """With RFC 4 enabled, populated orientation stays but ``None`` is culled."""
    result = _pop_metadata_optionals(_sample_metadata_dict(), enabled_rfcs=[4])
    by_name = {axis["name"]: axis for axis in result["axes"]}
    # Populated orientation survives when RFC 4 is enabled.
    assert by_name["x"]["orientation"] == {
        "type": "anatomical",
        "value": "left-to-right",
    }
    # A None orientation on a non-spatial axis is culled, not serialized.
    assert "orientation" not in by_name["c"]
