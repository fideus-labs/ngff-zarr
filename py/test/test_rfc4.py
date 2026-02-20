# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test RFC 4 anatomical orientation implementation."""

import pytest

from ngff_zarr.rfc4 import (
    AnatomicalOrientation,
    AnatomicalOrientationValues,
    itk_lps_to_anatomical_orientation,
    itk_direction_to_anatomical_orientation,
    is_rfc4_enabled,
    add_anatomical_orientation_to_axis,
    remove_anatomical_orientation_from_axis,
)


def test_anatomical_orientation_creation():
    """Test creating AnatomicalOrientation objects."""
    orientation = AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right)
    assert orientation.type == "anatomical"
    assert orientation.value == AnatomicalOrientationValues.left_to_right


def test_itk_lps_to_anatomical_orientation():
    """Test ITK LPS coordinate mapping."""
    # Test X axis (right to left in LPS)
    x_orientation = itk_lps_to_anatomical_orientation("x")
    assert x_orientation is not None
    assert x_orientation.type == "anatomical"
    assert x_orientation.value == AnatomicalOrientationValues.right_to_left

    # Test Y axis (anterior to posterior in LPS)
    y_orientation = itk_lps_to_anatomical_orientation("y")
    assert y_orientation is not None
    assert y_orientation.type == "anatomical"
    assert y_orientation.value == AnatomicalOrientationValues.anterior_to_posterior

    # Test Z axis (inferior to superior in LPS)
    z_orientation = itk_lps_to_anatomical_orientation("z")
    assert z_orientation is not None
    assert z_orientation.type == "anatomical"
    assert z_orientation.value == AnatomicalOrientationValues.inferior_to_superior

    # Test non-spatial axis
    c_orientation = itk_lps_to_anatomical_orientation("c")
    assert c_orientation is None


def test_is_rfc4_enabled():
    """Test RFC 4 enablement check."""
    assert is_rfc4_enabled([4]) is True
    assert is_rfc4_enabled([1, 2, 4, 5]) is True
    assert is_rfc4_enabled([1, 2, 3]) is False
    assert is_rfc4_enabled([]) is False
    assert is_rfc4_enabled(None) is False


def test_add_anatomical_orientation_to_axis():
    """Test adding orientation metadata to axis dictionary."""
    axis_dict = {"name": "x", "type": "space", "unit": "micrometer"}
    orientation = AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right)

    result = add_anatomical_orientation_to_axis(axis_dict, orientation)

    assert "orientation" in result
    assert result["orientation"]["type"] == "anatomical"
    assert result["orientation"]["value"] == "left-to-right"


def test_remove_anatomical_orientation_from_axis():
    """Test removing orientation metadata from axis dictionary."""
    axis_dict = {
        "name": "x",
        "type": "space",
        "unit": "micrometer",
        "orientation": {"type": "anatomical", "value": "left-to-right"},
    }

    result = remove_anatomical_orientation_from_axis(axis_dict)

    assert "orientation" not in result
    assert result["name"] == "x"
    assert result["type"] == "space"
    assert result["unit"] == "micrometer"

    # Test on axis without orientation
    axis_dict_no_orientation = {"name": "y", "type": "space"}
    result_no_orientation = remove_anatomical_orientation_from_axis(
        axis_dict_no_orientation
    )
    assert "orientation" not in result_no_orientation
    assert result_no_orientation == axis_dict_no_orientation


def test_anatomical_orientation_values_enum():
    """Test AnatomicalOrientationValues enum values."""
    # Test a few key values
    assert AnatomicalOrientationValues.left_to_right.value == "left-to-right"
    assert AnatomicalOrientationValues.right_to_left.value == "right-to-left"
    assert (
        AnatomicalOrientationValues.anterior_to_posterior.value
        == "anterior-to-posterior"
    )
    assert (
        AnatomicalOrientationValues.posterior_to_anterior.value
        == "posterior-to-anterior"
    )
    assert (
        AnatomicalOrientationValues.inferior_to_superior.value == "inferior-to-superior"
    )
    assert (
        AnatomicalOrientationValues.superior_to_inferior.value == "superior-to-inferior"
    )


# --- Tests for itk_direction_to_anatomical_orientation ---


class TestItkDirectionToAnatomicalOrientation:
    """Unit tests for itk_direction_to_anatomical_orientation."""

    # Primary axis directions (3D)

    def test_positive_x(self):
        o = itk_direction_to_anatomical_orientation([1, 0, 0])
        assert o.value == AnatomicalOrientationValues.right_to_left

    def test_negative_x(self):
        o = itk_direction_to_anatomical_orientation([-1, 0, 0])
        assert o.value == AnatomicalOrientationValues.left_to_right

    def test_positive_y(self):
        o = itk_direction_to_anatomical_orientation([0, 1, 0])
        assert o.value == AnatomicalOrientationValues.anterior_to_posterior

    def test_negative_y(self):
        o = itk_direction_to_anatomical_orientation([0, -1, 0])
        assert o.value == AnatomicalOrientationValues.posterior_to_anterior

    def test_positive_z(self):
        o = itk_direction_to_anatomical_orientation([0, 0, 1])
        assert o.value == AnatomicalOrientationValues.inferior_to_superior

    def test_negative_z(self):
        o = itk_direction_to_anatomical_orientation([0, 0, -1])
        assert o.value == AnatomicalOrientationValues.superior_to_inferior

    # Oblique with dominant component

    def test_oblique_dominant_positive_x(self):
        o = itk_direction_to_anatomical_orientation([0.9, 0.3, 0.2])
        assert o.value == AnatomicalOrientationValues.right_to_left

    def test_oblique_dominant_negative_x(self):
        o = itk_direction_to_anatomical_orientation([-0.8, 0.3, 0.1])
        assert o.value == AnatomicalOrientationValues.left_to_right

    def test_oblique_dominant_positive_y(self):
        o = itk_direction_to_anatomical_orientation([0.2, 0.85, 0.3])
        assert o.value == AnatomicalOrientationValues.anterior_to_posterior

    def test_oblique_dominant_negative_y(self):
        o = itk_direction_to_anatomical_orientation([0.1, -0.9, 0.2])
        assert o.value == AnatomicalOrientationValues.posterior_to_anterior

    def test_oblique_dominant_positive_z(self):
        o = itk_direction_to_anatomical_orientation([0.2, 0.3, 0.88])
        assert o.value == AnatomicalOrientationValues.inferior_to_superior

    def test_oblique_dominant_negative_z(self):
        o = itk_direction_to_anatomical_orientation([0.1, 0.2, -0.95])
        assert o.value == AnatomicalOrientationValues.superior_to_inferior

    # 2D cases

    def test_2d_positive_x(self):
        o = itk_direction_to_anatomical_orientation([1, 0])
        assert o.value == AnatomicalOrientationValues.right_to_left

    def test_2d_negative_x(self):
        o = itk_direction_to_anatomical_orientation([-1, 0])
        assert o.value == AnatomicalOrientationValues.left_to_right

    def test_2d_positive_y(self):
        o = itk_direction_to_anatomical_orientation([0, 1])
        assert o.value == AnatomicalOrientationValues.anterior_to_posterior

    def test_2d_negative_y(self):
        o = itk_direction_to_anatomical_orientation([0, -1])
        assert o.value == AnatomicalOrientationValues.posterior_to_anterior

    def test_2d_oblique_dominant_x(self):
        o = itk_direction_to_anatomical_orientation([0.8, 0.4])
        assert o.value == AnatomicalOrientationValues.right_to_left

    def test_2d_oblique_dominant_y(self):
        o = itk_direction_to_anatomical_orientation([0.3, 0.9])
        assert o.value == AnatomicalOrientationValues.anterior_to_posterior

    # Edge cases

    def test_equal_xy_favors_first(self):
        o = itk_direction_to_anatomical_orientation([0.707, 0.707, 0])
        assert o.value == AnatomicalOrientationValues.right_to_left

    def test_equal_magnitude_negative_x(self):
        o = itk_direction_to_anatomical_orientation([-0.707, 0.707, 0])
        assert o.value == AnatomicalOrientationValues.left_to_right

    def test_near_zero_tiny_x(self):
        o = itk_direction_to_anatomical_orientation([0.001, 0, 0])
        assert o.value == AnatomicalOrientationValues.right_to_left

    def test_too_few_elements_raises(self):
        with pytest.raises(ValueError, match="at least 2 elements"):
            itk_direction_to_anatomical_orientation([1.0])

    def test_type_is_anatomical(self):
        o = itk_direction_to_anatomical_orientation([1, 0, 0])
        assert o.type == "anatomical"
