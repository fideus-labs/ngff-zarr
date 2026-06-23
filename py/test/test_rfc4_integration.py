# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Integration test for RFC 4 anatomical orientation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from ngff_zarr import LPS, RAS, NgffImage, to_multiscales, to_ngff_zarr
from ngff_zarr.rfc4 import AnatomicalOrientation, AnatomicalOrientationValues
from packaging import version

zarr_version = version.parse(zarr.__version__)

pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)


def _read_axes_metadata(store_path):
    zarr_group = zarr.open(str(store_path), mode="r")
    # v0.5 stores metadata under "ome", v0.4 at root
    raw_attrs = dict(zarr_group.attrs)
    if "ome" in raw_attrs:
        multiscales_metadata = raw_attrs["ome"]["multiscales"][0]
    else:
        multiscales_metadata = raw_attrs["multiscales"][0]
    return multiscales_metadata["axes"]


def test_rfc4_orientation_written_automatically_when_present():
    """Anatomical orientation is written whenever it is present, no opt-in."""
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

        # No opt-in flag: orientation is written because it is present
        to_ngff_zarr(store=str(store_path), multiscales=multiscales)

        axes_metadata = _read_axes_metadata(store_path)

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


def test_rfc4_no_orientation_written_when_absent():
    """When no orientation metadata is present, none is written."""
    # Create a simple 3D image without any orientation metadata
    data = np.random.rand(10, 20, 30).astype(np.float32)

    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
    )

    # Convert to multiscales
    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        to_ngff_zarr(store=str(store_path), multiscales=multiscales)

        axes_metadata = _read_axes_metadata(store_path)

        # Check that no spatial axes have orientation
        for axis in axes_metadata:
            assert "orientation" not in axis, (
                f"Axis {axis['name']} should not have orientation when none "
                "was specified"
            )


def test_rfc4_orientation_keyword_preset():
    """The ``orientation`` keyword on to_multiscales accepts a preset name."""
    # Create a simple 3D image without orientation metadata on the image
    data = np.random.rand(5, 10, 15).astype(np.float32)

    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
    )

    # Request the LPS coordinate system explicitly via the orientation keyword
    multiscales = to_multiscales(ngff_image, scale_factors=[2], orientation="LPS")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        to_ngff_zarr(store=str(store_path), multiscales=multiscales)

        axes_metadata = _read_axes_metadata(store_path)

        # Find spatial axes and check for orientation
        spatial_axes_with_orientation = [
            ax for ax in axes_metadata if ax.get("orientation")
        ]
        assert (
            len(spatial_axes_with_orientation) == 3
        )  # x, y, z should have orientation

        # LPS preset orientations
        expected = {
            "x": "right-to-left",
            "y": "anterior-to-posterior",
            "z": "inferior-to-superior",
        }
        for axis in axes_metadata:
            if axis["name"] in expected:
                assert axis["orientation"]["type"] == "anatomical"
                assert axis["orientation"]["value"] == expected[axis["name"]]


def test_rfc4_orientation_keyword_mapping():
    """The ``orientation`` keyword on to_multiscales accepts a per-axis mapping."""
    data = np.random.rand(5, 10, 15).astype(np.float32)

    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
    )

    orientations = {
        "x": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
        "y": AnatomicalOrientation(
            value=AnatomicalOrientationValues.posterior_to_anterior
        ),
        "z": AnatomicalOrientation(
            value=AnatomicalOrientationValues.superior_to_inferior
        ),
    }

    multiscales = to_multiscales(
        ngff_image, scale_factors=[2], orientation=orientations
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        to_ngff_zarr(store=str(store_path), multiscales=multiscales)

        axes_metadata = _read_axes_metadata(store_path)

        expected = {
            "x": "left-to-right",
            "y": "posterior-to-anterior",
            "z": "superior-to-inferior",
        }
        for axis in axes_metadata:
            if axis["name"] in expected:
                assert axis["orientation"]["type"] == "anatomical"
                assert axis["orientation"]["value"] == expected[axis["name"]]


def test_rfc4_orientation_keyword_overrides_image_orientations():
    """The ``orientation`` keyword takes precedence over image axes_orientations."""
    data = np.random.rand(5, 10, 15).astype(np.float32)

    # Image carries LPS orientation...
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        axes_orientations=LPS,
    )

    # ...but the explicit orientation keyword (RAS) wins.
    multiscales = to_multiscales(ngff_image, scale_factors=[2], orientation="RAS")

    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        to_ngff_zarr(store=str(store_path), multiscales=multiscales)

        axes_metadata = _read_axes_metadata(store_path)

        by_name = {axis["name"]: axis for axis in axes_metadata}
        # RAS values, not LPS, must be written.
        assert (
            by_name["x"]["orientation"]["value"]
            == RAS["x"].value.value
            == "left-to-right"
        )
        assert (
            by_name["y"]["orientation"]["value"]
            == RAS["y"].value.value
            == "posterior-to-anterior"
        )


def test_rfc4_no_null_orientation_on_non_spatial_axes():
    """Non-spatial axes must not emit ``orientation: null``.

    Only the spatial axes carry an anatomical orientation. The time and
    channel axes have ``orientation=None``, which must be culled rather than
    serialized as an explicit ``null`` (regression test for issue #496).
    """
    data = np.random.rand(3, 2, 8, 16, 24).astype(np.float32)

    orientations = {
        "x": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
        "y": AnatomicalOrientation(
            value=AnatomicalOrientationValues.posterior_to_anterior
        ),
        "z": AnatomicalOrientation(
            value=AnatomicalOrientationValues.superior_to_inferior
        ),
    }

    ngff_image = NgffImage(
        data=data,
        dims=("t", "c", "z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0, "c": 1.0, "t": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0, "c": 0.0, "t": 0.0},
        axes_orientations=orientations,
    )

    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        to_ngff_zarr(store=str(store_path), multiscales=multiscales)

        axes_metadata = _read_axes_metadata(store_path)

        for axis in axes_metadata:
            if axis["name"] in ("t", "c"):
                # The key must be absent entirely, not present with a null value.
                assert "orientation" not in axis, (
                    f"Non-spatial axis {axis['name']} should not carry orientation"
                )
            else:
                assert axis["orientation"]["type"] == "anatomical"


def test_rfc4_legacy_enabled_rfcs_kwarg_rejected():
    """The removed ``enabled_rfcs`` kwarg raises a clear migration error.

    ``enabled_rfcs`` was dropped in favor of automatic orientation writing.
    A legacy caller passing it must get an explicit message rather than an
    opaque error from the downstream zarr-creation kwargs.
    """
    data = np.random.rand(4, 8, 12).astype(np.float32)

    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
    )
    multiscales = to_multiscales(ngff_image, scale_factors=[2])

    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        with pytest.raises(TypeError, match="no longer accepts 'enabled_rfcs'"):
            to_ngff_zarr(
                store=str(store_path), multiscales=multiscales, enabled_rfcs=[4]
            )
