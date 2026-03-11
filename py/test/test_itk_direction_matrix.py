# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for direction matrix handling in itk_image_to_ngff_image."""

import itkwasm
import numpy as np
import pytest
from ngff_zarr import itk_image_to_ngff_image
from ngff_zarr.itk_image_to_ngff_image import _flatten_direction
from ngff_zarr.rfc4 import AnatomicalOrientationValues


def _make_itkwasm_image(size, direction):
    """Build an itkwasm.Image with a given size and direction matrix."""
    dim = len(size)
    data = np.zeros(tuple(reversed(size)), dtype=np.uint8)
    return itkwasm.Image(
        imageType=itkwasm.ImageType(
            dimension=dim,
            componentType="uint8",
            pixelType="Scalar",
            components=1,
        ),
        name="test",
        origin=[0.0] * dim,
        spacing=[1.0] * dim,
        direction=np.asarray(direction, dtype=np.float64).flatten(),
        size=list(size),
        data=data,
    )


class TestIdentityDirection:
    """Identity direction should fall back to standard LPS orientations."""

    def test_3d_identity(self):
        img = _make_itkwasm_image([10, 20, 30], np.eye(3))
        ngff = itk_image_to_ngff_image(img)

        assert ngff.dims == ("z", "y", "x")
        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )
        assert (
            ngff.axes_orientations["z"].value
            == AnatomicalOrientationValues.inferior_to_superior
        )

    def test_2d_identity(self):
        img = _make_itkwasm_image([10, 20], np.eye(2))
        ngff = itk_image_to_ngff_image(img)

        assert ngff.dims == ("y", "x")
        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )


class TestPermutedAxes:
    """Permuted direction matrices map axes to different physical dimensions."""

    def test_3d_permuted_xyz_to_yzx(self):
        # Direction matrix:
        # [0  0  1]   (X maps to Z: I/S)
        # [1  0  0]   (Y maps to X: R/L)
        # [0  1  0]   (Z maps to Y: A/P)
        direction = [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
        ]
        img = _make_itkwasm_image([10, 20, 30], direction)
        ngff = itk_image_to_ngff_image(img)

        assert ngff.dims == ("z", "y", "x")
        # x (ITK axis 0) → column 0 → [0, 1, 0] → Y dominant → A/P
        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )
        # y (ITK axis 1) → column 1 → [0, 0, 1] → Z dominant → I/S
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.inferior_to_superior
        )
        # z (ITK axis 2) → column 2 → [1, 0, 0] → X dominant → R/L
        assert (
            ngff.axes_orientations["z"].value
            == AnatomicalOrientationValues.right_to_left
        )

    def test_3d_flipped_y(self):
        # Direction matrix with Y axis flipped:
        # [1   0   0]
        # [0  -1   0]
        # [0   0   1]
        direction = [
            1,
            0,
            0,
            0,
            -1,
            0,
            0,
            0,
            1,
        ]
        img = _make_itkwasm_image([10, 20, 30], direction)
        ngff = itk_image_to_ngff_image(img)

        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.posterior_to_anterior
        )
        assert (
            ngff.axes_orientations["z"].value
            == AnatomicalOrientationValues.inferior_to_superior
        )

    def test_2d_swapped_xy(self):
        # Direction matrix:
        # [0  1]   (X maps to Y: A/P)
        # [1  0]   (Y maps to X: R/L)
        direction = [0, 1, 1, 0]
        img = _make_itkwasm_image([10, 20], direction)
        ngff = itk_image_to_ngff_image(img)

        assert ngff.dims == ("y", "x")
        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.right_to_left
        )


class TestObliqueDirection:
    """Oblique (rotated) direction matrices use the dominant component."""

    def test_3d_45_degree_rotation_xy(self):
        cos45 = 0.707
        direction = [
            cos45,
            -cos45,
            0,
            cos45,
            cos45,
            0,
            0,
            0,
            1,
        ]
        img = _make_itkwasm_image([10, 20, 30], direction)
        ngff = itk_image_to_ngff_image(img)

        # x → col 0 → [cos45, cos45, 0] → equal, first wins → R/L
        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        # y → col 1 → [-cos45, cos45, 0] → equal, first wins, negative → L/R
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.left_to_right
        )
        # z → col 2 → [0, 0, 1] → Z dominant → I/S
        assert (
            ngff.axes_orientations["z"].value
            == AnatomicalOrientationValues.inferior_to_superior
        )

    def test_3d_dominant_components(self):
        direction = [
            0.9,
            0.2,
            0.1,
            0.3,
            0.85,
            0.2,
            0.2,
            0.3,
            0.9,
        ]
        img = _make_itkwasm_image([10, 20, 30], direction)
        ngff = itk_image_to_ngff_image(img)

        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )
        assert (
            ngff.axes_orientations["z"].value
            == AnatomicalOrientationValues.inferior_to_superior
        )

    def test_2d_30_degree_rotation(self):
        cos30 = 0.866
        sin30 = 0.5
        direction = [cos30, -sin30, sin30, cos30]
        img = _make_itkwasm_image([10, 20], direction)
        ngff = itk_image_to_ngff_image(img)

        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )


class TestDisabledOrientation:
    """When anatomical orientation is disabled, no axes_orientations are set."""

    def test_no_orientation(self):
        img = _make_itkwasm_image([10, 20, 30], np.eye(3))
        ngff = itk_image_to_ngff_image(img, add_anatomical_orientation=False)
        assert ngff.axes_orientations is None


class TestDirectionValidation:
    """Tests for direction matrix validation and error handling."""

    def test_invalid_direction_length_raises(self):
        """_flatten_direction should raise ValueError for wrong-length input."""
        with pytest.raises(ValueError, match="expected 9"):
            _flatten_direction([1.0, 0.0, 0.0, 0.0], n=3)

    def test_near_identity_direction_falls_back_to_lps(self):
        """A near-identity matrix (tiny floating-point noise) should use LPS mapping."""
        # Introduce sub-epsilon noise around identity
        near_identity = np.eye(3) + np.full((3, 3), 1e-8)
        img = _make_itkwasm_image([10, 20, 30], near_identity)
        ngff = itk_image_to_ngff_image(img)

        # Should fall back to standard LPS orientations (same as true identity)
        assert (
            ngff.axes_orientations["x"].value
            == AnatomicalOrientationValues.right_to_left
        )
        assert (
            ngff.axes_orientations["y"].value
            == AnatomicalOrientationValues.anterior_to_posterior
        )
        assert (
            ngff.axes_orientations["z"].value
            == AnatomicalOrientationValues.inferior_to_superior
        )
