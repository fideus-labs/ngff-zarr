# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test RFC 4 anatomical orientation implementation."""

import pytest
from ngff_zarr.rfc4 import (
    AnatomicalOrientation,
    AnatomicalOrientationValues,
    add_anatomical_orientation_to_axis,
    anatomical_orientation_to_itk_direction,
    is_rfc4_enabled,
    itk_direction_to_anatomical_orientation,
    itk_lps_to_anatomical_orientation,
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
    # Subject-local layered- and polarized-tissue values (per ome/ngff#528)
    assert (
        AnatomicalOrientationValues.superficial_to_deep.value == "superficial-to-deep"
    )
    assert (
        AnatomicalOrientationValues.deep_to_superficial.value == "deep-to-superficial"
    )
    assert AnatomicalOrientationValues.apical_to_basal.value == "apical-to-basal"
    assert AnatomicalOrientationValues.basal_to_apical.value == "basal-to-apical"
    assert AnatomicalOrientationValues.apex_to_base.value == "apex-to-base"
    assert AnatomicalOrientationValues.base_to_apex.value == "base-to-apex"


# --- Tests for anatomical_orientation_to_itk_direction ---


class TestAnatomicalOrientationToItkDirection:
    """Unit tests for anatomical_orientation_to_itk_direction."""

    def test_right_to_left(self):
        col = anatomical_orientation_to_itk_direction(
            AnatomicalOrientationValues.right_to_left
        )
        assert col == [1.0, 0.0, 0.0]

    def test_left_to_right(self):
        col = anatomical_orientation_to_itk_direction(
            AnatomicalOrientationValues.left_to_right
        )
        assert col == [-1.0, 0.0, 0.0]

    def test_anterior_to_posterior(self):
        col = anatomical_orientation_to_itk_direction(
            AnatomicalOrientationValues.anterior_to_posterior
        )
        assert col == [0.0, 1.0, 0.0]

    def test_posterior_to_anterior(self):
        col = anatomical_orientation_to_itk_direction(
            AnatomicalOrientationValues.posterior_to_anterior
        )
        assert col == [0.0, -1.0, 0.0]

    def test_inferior_to_superior(self):
        col = anatomical_orientation_to_itk_direction(
            AnatomicalOrientationValues.inferior_to_superior
        )
        assert col == [0.0, 0.0, 1.0]

    def test_superior_to_inferior(self):
        col = anatomical_orientation_to_itk_direction(
            AnatomicalOrientationValues.superior_to_inferior
        )
        assert col == [0.0, 0.0, -1.0]

    def test_non_lps_returns_none(self):
        assert (
            anatomical_orientation_to_itk_direction(
                AnatomicalOrientationValues.dorsal_to_ventral
            )
            is None
        )
        assert (
            anatomical_orientation_to_itk_direction(
                AnatomicalOrientationValues.rostral_to_caudal
            )
            is None
        )
        assert (
            anatomical_orientation_to_itk_direction(
                AnatomicalOrientationValues.proximal_to_distal
            )
            is None
        )

    def test_subject_local_returns_none(self):
        """Layered/polarized-tissue values are not LPS-compatible (ome/ngff#528)."""
        for value in (
            AnatomicalOrientationValues.superficial_to_deep,
            AnatomicalOrientationValues.deep_to_superficial,
            AnatomicalOrientationValues.apical_to_basal,
            AnatomicalOrientationValues.basal_to_apical,
            AnatomicalOrientationValues.apex_to_base,
            AnatomicalOrientationValues.base_to_apex,
        ):
            assert anatomical_orientation_to_itk_direction(value) is None


# --- Tests for itk_direction_to_anatomical_orientation ---


class TestItkDirectionToAnatomicalOrientation:
    """Unit tests for itk_direction_to_anatomical_orientation."""

    def test_positive_x(self):
        ori = itk_direction_to_anatomical_orientation([1.0, 0.0, 0.0])
        assert ori.value == AnatomicalOrientationValues.right_to_left

    def test_negative_x(self):
        ori = itk_direction_to_anatomical_orientation([-1.0, 0.0, 0.0])
        assert ori.value == AnatomicalOrientationValues.left_to_right

    def test_positive_y(self):
        ori = itk_direction_to_anatomical_orientation([0.0, 1.0, 0.0])
        assert ori.value == AnatomicalOrientationValues.anterior_to_posterior

    def test_negative_y(self):
        ori = itk_direction_to_anatomical_orientation([0.0, -1.0, 0.0])
        assert ori.value == AnatomicalOrientationValues.posterior_to_anterior

    def test_positive_z(self):
        ori = itk_direction_to_anatomical_orientation([0.0, 0.0, 1.0])
        assert ori.value == AnatomicalOrientationValues.inferior_to_superior

    def test_negative_z(self):
        ori = itk_direction_to_anatomical_orientation([0.0, 0.0, -1.0])
        assert ori.value == AnatomicalOrientationValues.superior_to_inferior

    def test_oblique_dominant_x(self):
        """Oblique direction with dominant +X component."""
        ori = itk_direction_to_anatomical_orientation([0.9, 0.3, 0.2])
        assert ori.value == AnatomicalOrientationValues.right_to_left

    def test_oblique_dominant_negative_y(self):
        """Oblique direction with dominant -Y component."""
        ori = itk_direction_to_anatomical_orientation([0.1, -0.9, 0.2])
        assert ori.value == AnatomicalOrientationValues.posterior_to_anterior

    def test_2d_positive_x(self):
        """2D direction: positive X."""
        ori = itk_direction_to_anatomical_orientation([1.0, 0.0])
        assert ori.value == AnatomicalOrientationValues.right_to_left

    def test_2d_negative_y(self):
        """2D direction: negative Y."""
        ori = itk_direction_to_anatomical_orientation([0.0, -1.0])
        assert ori.value == AnatomicalOrientationValues.posterior_to_anterior

    def test_too_short_raises(self):
        """Direction column with <2 elements raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            itk_direction_to_anatomical_orientation([1.0])
