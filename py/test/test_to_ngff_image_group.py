# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test to_ngff_image with zarr.Group objects.

This test addresses the issue where tifffile.imread with aszarr can return
a zarr.Group (e.g., for pyramidal TIFF files), which would cause an
AttributeError when passed to to_ngff_image because Group objects don't
have an 'ndim' attribute.
"""

import tempfile
from pathlib import Path

import numpy as np
import zarr
from ngff_zarr import to_ngff_image
from packaging import version

# Check Zarr version for API compatibility
zarr_version = version.parse(zarr.__version__)
is_zarr_v3 = zarr_version >= version.parse("3.0.0")


def test_to_ngff_image_with_zarr_group():
    """Test that to_ngff_image can handle a zarr.Group by selecting the first array."""
    # Create a zarr group with multiple arrays (simulating multi-series structure)
    tmpdir = Path(tempfile.mkdtemp())
    root = zarr.open_group(str(tmpdir / "test.zarr"), mode="w")

    # Create two arrays in the group (like multi-series TIFF)
    # Use API appropriate for the Zarr version
    data0 = np.random.rand(10, 100, 100).astype(np.float32)
    data1 = np.random.rand(5, 50, 50).astype(np.float32)

    if is_zarr_v3:
        # Zarr v3: use create_array with shape and dtype, then assign data
        arr0 = root.create_array("0", shape=data0.shape, dtype=data0.dtype)
        arr0[:] = data0
        arr1 = root.create_array("1", shape=data1.shape, dtype=data1.dtype)
        arr1[:] = data1
    else:
        # Zarr v2: use create_dataset with data parameter
        root.create_dataset("0", data=data0)
        root.create_dataset("1", data=data1)

    # Reopen as Group (this is what would be returned by zarr.open)
    reopened_group = zarr.open(str(tmpdir / "test.zarr"), mode="r")
    assert isinstance(reopened_group, zarr.Group)

    # Test that to_ngff_image handles the Group correctly
    # It should use the first array in the group
    ngff_image = to_ngff_image(reopened_group)

    # Verify the result uses the first array
    assert ngff_image.dims == ("z", "y", "x")
    assert ngff_image.data.shape == (10, 100, 100)
    assert ngff_image.scale == {"z": 1.0, "y": 1.0, "x": 1.0}
    assert ngff_image.translation == {"z": 0.0, "y": 0.0, "x": 0.0}


def test_to_ngff_image_with_empty_zarr_group():
    """Test that to_ngff_image raises an error for an empty zarr.Group."""
    # Create an empty zarr group
    tmpdir = Path(tempfile.mkdtemp())
    zarr.open_group(str(tmpdir / "empty.zarr"), mode="w")

    # Reopen as Group
    reopened_group = zarr.open(str(tmpdir / "empty.zarr"), mode="r")
    assert isinstance(reopened_group, zarr.Group)

    # Test that to_ngff_image raises ValueError for empty group
    import pytest

    with pytest.raises(ValueError, match="no arrays found"):
        to_ngff_image(reopened_group)


def test_to_ngff_image_group_with_custom_metadata():
    """Test that to_ngff_image with Group respects custom metadata parameters."""
    # Create a zarr group with an array
    tmpdir = Path(tempfile.mkdtemp())
    root = zarr.open_group(str(tmpdir / "test.zarr"), mode="w")

    # Use API appropriate for the Zarr version
    data = np.random.rand(5, 64, 64).astype(np.float32)

    if is_zarr_v3:
        # Zarr v3: use create_array with shape and dtype, then assign data
        arr = root.create_array("data", shape=data.shape, dtype=data.dtype)
        arr[:] = data
    else:
        # Zarr v2: use create_dataset with data parameter
        root.create_dataset("data", data=data)

    # Reopen as Group
    reopened_group = zarr.open(str(tmpdir / "test.zarr"), mode="r")

    # Test with custom metadata
    ngff_image = to_ngff_image(
        reopened_group,
        dims=("z", "y", "x"),
        scale={"z": 2.0, "y": 0.5, "x": 0.5},
        translation={"z": 10.0, "y": 5.0, "x": 5.0},
        name="custom_image",
        axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
    )

    # Verify custom metadata is preserved
    assert ngff_image.dims == ("z", "y", "x")
    assert ngff_image.scale == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert ngff_image.translation == {"z": 10.0, "y": 5.0, "x": 5.0}
    assert ngff_image.name == "custom_image"
    assert ngff_image.axes_units == {
        "z": "micrometer",
        "y": "micrometer",
        "x": "micrometer",
    }
