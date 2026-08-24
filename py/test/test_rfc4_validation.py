# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test RFC 4 validation in from_ngff_zarr."""

import pytest
import zarr
from jsonschema import ValidationError
from ngff_zarr import from_ngff_zarr
from ngff_zarr.rfc4_validation import (
    has_rfc4_orientation_metadata,
    validate_rfc4_orientation,
)


def test_has_rfc4_orientation_metadata():
    """Test detection of RFC 4 orientation metadata."""
    # No orientation metadata
    axes_no_orientation = [
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"},
    ]
    assert not has_rfc4_orientation_metadata(axes_no_orientation)

    # With orientation metadata
    axes_with_orientation = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
    ]
    assert has_rfc4_orientation_metadata(axes_with_orientation)


def test_validate_rfc4_orientation_valid():
    """Test valid RFC 4 orientation validation."""
    # Valid LPS orientation
    axes_lps = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "left-to-right"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "posterior-to-anterior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "superior-to-inferior"},
        },
    ]

    # Should not raise any exception
    validate_rfc4_orientation(axes_lps)


@pytest.mark.parametrize(
    ("x_value", "y_value", "z_value"),
    [
        # Forward directions. RFC 4 requires the value to be unique per spatial
        # axis, so each triple pairs three distinct orientations.
        ("superficial-to-deep", "apical-to-basal", "apex-to-base"),
        # Reverse directions, covering the remaining three new values.
        ("deep-to-superficial", "basal-to-apical", "base-to-apex"),
    ],
)
def test_validate_rfc4_orientation_subject_local_values(x_value, y_value, z_value):
    """Layered/polarized-tissue values pass validation (ome/ngff#528).

    Exercises both the in-code value set and the JSON-schema enum, so a value
    missing from either artifact would be caught here. The two parameter sets
    together cover all six new subject-local orientation values.
    """
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    axes_subject_local = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": x_value},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": y_value},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": z_value},
        },
    ]

    # Should not raise any exception
    validate_rfc4_orientation(axes_subject_local)


def test_validate_rfc4_orientation_mixed_types():
    """Test RFC 4 validation with mixed axis types."""
    # Valid with time and channel axes mixed in
    axes_mixed = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
    ]

    # Should not raise any exception
    validate_rfc4_orientation(axes_mixed)


def test_validate_rfc4_orientation_partial_is_valid():
    """RFC 4 makes orientation optional per spatial axis, so partial is valid.

    The specification states no all-or-none completeness requirement: an image may
    orient only some of its spatial axes and leave the rest undefined.
    """
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Orientation is defined for two spatial axes and left undefined on the third.
    axes_incomplete = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            # Missing orientation
        },
    ]

    # Should not raise: an unoriented spatial axis is simply undefined.
    validate_rfc4_orientation(axes_incomplete)


def test_validate_rfc4_orientation_non_anatomical_type():
    """RFC 4 defines a single orientation type, so any other type is rejected."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # The "y" axis declares a type the RFC 4 vocabulary does not define.
    axes_inconsistent = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "different_type", "value": "anterior-to-posterior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
    ]

    with pytest.raises(ValueError, match="must be anatomical"):
        validate_rfc4_orientation(axes_inconsistent)


def test_validate_rfc4_orientation_non_object_is_rejected():
    """A non-object orientation is malformed, not undefined.

    Only an absent field, ``null`` and an empty object are undefined. Any other
    non-null value (string, list, number) is a real -- but malformed -- use of
    orientation and must fail: on a spatial axis via the type check, on a
    non-spatial axis via the spatial-only rule.
    """
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    with pytest.raises(ValueError, match="must be anatomical"):
        validate_rfc4_orientation(
            [
                {
                    "name": "x",
                    "type": "space",
                    "unit": "micrometer",
                    "orientation": "nope",
                }
            ]
        )

    with pytest.raises(ValueError, match="non-space axes"):
        validate_rfc4_orientation(
            [
                {"name": "t", "type": "time", "orientation": "nope"},
                {
                    "name": "x",
                    "type": "space",
                    "unit": "micrometer",
                    "orientation": {"type": "anatomical", "value": "right-to-left"},
                },
            ]
        )


def test_validate_rfc4_orientation_invalid_value():
    """Test RFC 4 validation fails with invalid orientation value."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Invalid orientation value
    axes_invalid = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "invalid-orientation"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
    ]

    with pytest.raises(ValidationError):
        validate_rfc4_orientation(axes_invalid)


def test_validate_rfc4_orientation_on_non_space_axis():
    """RFC 4 orientation is rejected on a non-spatial (time/channel) axis."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    axes_bad = [
        {
            "name": "t",
            "type": "time",
            "unit": "second",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
    ]

    with pytest.raises(ValueError, match="non-space axes"):
        validate_rfc4_orientation(axes_bad)


def test_validate_rfc4_orientation_empty_dict_on_non_space_is_undefined():
    """An empty orientation object on a non-spatial axis is not a violation.

    RFC 4 treats an absent orientation and an explicit ``null`` as equivalent --
    the orientation is simply undefined -- so an empty object is likewise not a
    use of orientation on that axis."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    axes_bad = [
        {"name": "t", "type": "time", "unit": "second", "orientation": {}},
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
    ]

    # Should not raise: an empty orientation is undefined, not a use of it.
    validate_rfc4_orientation(axes_bad)


def test_validate_rfc4_orientation_duplicate_anatomical_axis():
    """Two spatial axes describing the same anatomical axis are rejected."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # z and x both lie on the left-right axis (mutually exclusive antonyms).
    axes_dup = [
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "left-to-right"},
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
        },
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
    ]

    with pytest.raises(ValueError, match="same anatomical axis"):
        validate_rfc4_orientation(axes_dup)


def test_validate_rfc4_orientation_duplicate_anatomical_axis_same_name():
    """Two spatial entries that reuse one axis name still collide.

    Malformed metadata can repeat a name. A name-keyed map would overwrite the
    first entry and let the pair slip through, so the check keeps every entry.
    """
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    axes_dup = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "left-to-right"},
        },
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
        },
    ]

    with pytest.raises(ValueError, match="same anatomical axis"):
        validate_rfc4_orientation(axes_dup)


def test_from_ngff_zarr_with_rfc4_validation(tmp_path):
    """Test from_ngff_zarr with RFC 4 validation enabled."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Create a store with RFC 4 orientation metadata
    store = str(tmp_path / "test.zarr")

    # Create a simple zarr array
    root = zarr.open_group(store, mode="w")
    if hasattr(root, "create_array"):
        root.create_array("0", shape=(10, 10, 10), dtype="uint8")
    else:
        root.create_dataset("0", shape=(10, 10, 10), dtype="uint8")

    # Add OME-NGFF metadata with RFC 4 orientation. Spatial axes are ordered
    # (z, y, x) -- the suffix of (z, y, x) required by the v0.4 axis-order MUST
    # now enforced on read -- with each orientation kept on its named axis.
    multiscales_metadata = {
        "version": "0.4",
        "name": "test",
        "axes": [
            {
                "name": "z",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
            },
            {
                "name": "y",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
            },
            {
                "name": "x",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "right-to-left"},
            },
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }
        ],
    }

    root.attrs["multiscales"] = [multiscales_metadata]

    # Should succeed with valid orientation
    multiscales = from_ngff_zarr(store, validate=True)
    assert multiscales is not None


def test_from_ngff_zarr_with_rfc4_validation_invalid(tmp_path):
    """Test from_ngff_zarr with RFC 4 validation fails on invalid orientation."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Create a store with invalid RFC 4 orientation metadata
    store = str(tmp_path / "test.zarr")

    # Create a simple zarr array
    root = zarr.open_group(store, mode="w")
    if hasattr(root, "create_array"):
        root.create_array("0", shape=(10, 10, 10), dtype="uint8")
    else:
        root.create_dataset("0", shape=(10, 10, 10), dtype="uint8")

    # Add OME-NGFF metadata with incomplete RFC 4 orientation (missing z orientation)
    multiscales_metadata = {
        "version": "0.4",
        "name": "test",
        "axes": [
            {
                "name": "x",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "right-to-left"},
            },
            {
                "name": "y",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
            },
            {
                "name": "z",
                "type": "space",
                "unit": "micrometer",
                # Out-of-vocabulary value - this should cause validation to fail
                "orientation": {"type": "anatomical", "value": "not-a-direction"},
            },
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }
        ],
    }

    root.attrs["multiscales"] = [multiscales_metadata]

    # Should fail on the out-of-vocabulary orientation value
    with pytest.raises(ValidationError, match="Invalid orientation value"):
        from_ngff_zarr(store, validate=True)


def test_from_ngff_zarr_without_rfc4_validation(tmp_path):
    """Test from_ngff_zarr works without validation even with invalid orientation."""
    # Create a store with invalid RFC 4 orientation metadata
    store = str(tmp_path / "test.zarr")

    # Create a simple zarr array
    root = zarr.open_group(store, mode="w")
    if hasattr(root, "create_array"):
        root.create_array("0", shape=(10, 10, 10), dtype="uint8")
    else:
        root.create_dataset("0", shape=(10, 10, 10), dtype="uint8")

    # Add OME-NGFF metadata with incomplete RFC 4 orientation
    multiscales_metadata = {
        "version": "0.4",
        "name": "test",
        "axes": [
            {
                "name": "x",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "right-to-left"},
            },
            {
                "name": "y",
                "type": "space",
                "unit": "micrometer",
                "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
            },
            {
                "name": "z",
                "type": "space",
                "unit": "micrometer",
                # Missing orientation - should be fine without validation
            },
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }
        ],
    }

    root.attrs["multiscales"] = [multiscales_metadata]

    # Should succeed without validation
    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None


def test_orientation_on_a_non_space_axis_is_reachable():
    """An orientation only on a non-spatial axis is itself the violation.

    Gating validation on a *spatial* orientation would skip exactly the
    document the non-space rule exists to catch.
    """
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    from ngff_zarr.rfc4_validation import has_any_rfc4_orientation

    axes = [
        {
            "name": "t",
            "type": "time",
            "orientation": {"type": "anatomical", "value": "left-to-right"},
        },
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    assert not has_rfc4_orientation_metadata(axes)
    assert has_any_rfc4_orientation(axes)
    with pytest.raises(ValueError, match="non-space axes"):
        validate_rfc4_orientation(axes)


def test_an_empty_orientation_does_not_trigger_validation():
    """A null or empty orientation is undefined under RFC 4, not a violation."""
    from ngff_zarr.rfc4_validation import has_any_rfc4_orientation

    assert not has_any_rfc4_orientation(
        [{"name": "y", "type": "space", "orientation": None}]
    )
    assert not has_any_rfc4_orientation(
        [{"name": "y", "type": "space", "orientation": {}}]
    )
    assert not has_any_rfc4_orientation([{"name": "y", "type": "space"}])


@pytest.mark.parametrize("orientation", [[], "", 0, False, "left-to-right"])
def test_a_falsey_orientation_still_reaches_the_validator(orientation):
    """Only ``None`` and ``{}`` are undefined; anything else is malformed.

    A truthiness test would call ``[]``, ``""``, ``0`` and ``False`` absent and
    skip validation, but :func:`validate_rfc4_orientation` rejects each of them,
    so the gate would hide exactly those documents.
    """
    jsonschema = pytest.importorskip(
        "jsonschema", reason="jsonschema required for RFC 4 validation"
    )

    from ngff_zarr.rfc4_validation import has_any_rfc4_orientation

    axes = [
        {"name": "y", "type": "space", "orientation": orientation},
        {"name": "x", "type": "space"},
    ]
    assert has_any_rfc4_orientation(axes)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        validate_rfc4_orientation(axes)


def test_read_rejects_orientation_on_a_non_space_axis(tmp_path):
    """The reader reaches the non-space rule with no spatial axis oriented.

    Gating on a spatial orientation left this document accepted: nothing in it
    orients a space axis, which is exactly what makes it invalid.
    """
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    store = str(tmp_path / "test.zarr")
    root = zarr.open_group(store, mode="w")
    if hasattr(root, "create_array"):
        root.create_array("0", shape=(2, 10, 10), dtype="uint8")
    else:
        root.create_dataset("0", shape=(2, 10, 10), dtype="uint8")

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "test",
            "axes": [
                {
                    "name": "t",
                    "type": "time",
                    "orientation": {
                        "type": "anatomical",
                        "value": "left-to-right",
                    },
                },
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="non-space axes"):
        from_ngff_zarr(store, validate=True)
