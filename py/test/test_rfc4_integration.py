# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Integration test for RFC 4 anatomical orientation."""

import numpy as np
import tempfile
from pathlib import Path

from ngff_zarr import NgffImage, to_multiscales, to_ngff_zarr
from ngff_zarr.rfc4 import AnatomicalOrientation, AnatomicalOrientationValues


def test_rfc4_integration_with_enabled_rfcs():
    """Test RFC 4 anatomical orientation when enabled."""
    # Create a simple 3D image
    data = np.random.rand(10, 20, 30).astype(np.float32)

    # Create orientations for spatial axes
    orientations = {
        "x": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
        "y": AnatomicalOrientation(
            value=AnatomicalOrientationValues.posterior_to_anterior
        ),
        "z": AnatomicalOrientation(
            value=AnatomicalOrientationValues.superior_to_inferior
        ),
    }

    # Create NgffImage with orientation metadata
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        axes_orientations=orientations,
    )

    # Convert to multiscales
    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        # Test with RFC 4 enabled
        to_ngff_zarr(store=str(store_path), multiscales=multiscales, enabled_rfcs=[4])

        # Read back and check metadata
        import zarr

        zarr_group = zarr.open(str(store_path), mode="r")

        # Check that multiscales metadata contains orientation
        multiscales_metadata = zarr_group.attrs["multiscales"][0]
        axes_metadata = multiscales_metadata["axes"]

        # Find spatial axes and check for orientation
        spatial_axes_with_orientation = [
            ax for ax in axes_metadata if ax.get("orientation")
        ]
        assert (
            len(spatial_axes_with_orientation) == 3
        )  # x, y, z should have orientation

        # Check specific orientations
        for axis in axes_metadata:
            if axis["name"] == "x":
                assert "orientation" in axis
                assert axis["orientation"]["type"] == "anatomical"
                assert axis["orientation"]["value"] == "left-to-right"
            elif axis["name"] == "y":
                assert "orientation" in axis
                assert axis["orientation"]["type"] == "anatomical"
                assert axis["orientation"]["value"] == "posterior-to-anterior"
            elif axis["name"] == "z":
                assert "orientation" in axis
                assert axis["orientation"]["type"] == "anatomical"
                assert axis["orientation"]["value"] == "superior-to-inferior"


def test_rfc4_integration_without_enabled_rfcs():
    """Test RFC 4 anatomical orientation when not enabled."""
    # Create a simple 3D image
    data = np.random.rand(10, 20, 30).astype(np.float32)

    # Create orientations for spatial axes
    orientations = {
        "x": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
        "y": AnatomicalOrientation(
            value=AnatomicalOrientationValues.posterior_to_anterior
        ),
        "z": AnatomicalOrientation(
            value=AnatomicalOrientationValues.superior_to_inferior
        ),
    }

    # Create NgffImage with orientation metadata
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        axes_orientations=orientations,
    )

    # Convert to multiscales
    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        # Test with RFC 4 NOT enabled (default)
        to_ngff_zarr(
            store=str(store_path),
            multiscales=multiscales,
            # enabled_rfcs not specified, so RFC 4 should be filtered out
        )

        # Read back and check metadata
        import zarr

        zarr_group = zarr.open(str(store_path), mode="r")

        # Check that multiscales metadata does NOT contain orientation
        multiscales_metadata = zarr_group.attrs["multiscales"][0]
        axes_metadata = multiscales_metadata["axes"]

        # Check that no spatial axes have orientation
        for axis in axes_metadata:
            assert (
                "orientation" not in axis
            ), f"Axis {axis['name']} should not have orientation when RFC 4 is disabled"


def test_rfc4_integration_with_other_rfcs():
    """Test RFC 4 anatomical orientation when enabled alongside other RFCs."""
    # Create a simple 3D image
    data = np.random.rand(5, 10, 15).astype(np.float32)

    # Create orientations for spatial axes
    orientations = {
        "x": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
        "y": AnatomicalOrientation(
            value=AnatomicalOrientationValues.posterior_to_anterior
        ),
        "z": AnatomicalOrientation(
            value=AnatomicalOrientationValues.superior_to_inferior
        ),
    }

    # Create NgffImage with orientation metadata
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        axes_orientations=orientations,
    )

    # Convert to multiscales
    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        # Test with RFC 4 enabled alongside hypothetical other RFCs
        to_ngff_zarr(
            store=str(store_path),
            multiscales=multiscales,
            enabled_rfcs=[1, 2, 4, 5],  # RFC 4 is enabled
        )

        # Read back and check metadata
        import zarr

        zarr_group = zarr.open(str(store_path), mode="r")

        # Check that multiscales metadata contains orientation
        multiscales_metadata = zarr_group.attrs["multiscales"][0]
        axes_metadata = multiscales_metadata["axes"]

        # Find spatial axes and check for orientation
        spatial_axes_with_orientation = [
            ax for ax in axes_metadata if ax.get("orientation")
        ]
        assert (
            len(spatial_axes_with_orientation) == 3
        )  # x, y, z should have orientation
