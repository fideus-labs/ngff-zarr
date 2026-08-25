# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test to_ngff_image with zarr group nodes.

Multi-level TIFFs read through the store reader (e.g. pyramidal OME-TIFF via
tifffile's aszarr) hand to_ngff_image a group node rather than an array.
to_ngff_image must extract the full-resolution array from the group instead
of failing on the missing 'ndim' attribute. zarr-python is used here only to
construct the on-disk fixtures; the groups under test are ngff-zarr's own
compat-layer nodes.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import to_ngff_image
from ngff_zarr._zarrista_utils import LocalZarrGroup, open_local_node
from packaging import version

pytestmark = pytest.mark.skipif(
    version.parse(zarr.__version__).major < 3,
    reason="fixtures are authored as zarr format 3 with the zarr-python 3 API",
)


def _open_group(path) -> LocalZarrGroup:
    node = open_local_node(str(path), zarr_format=3)
    assert isinstance(node, LocalZarrGroup)
    return node


def test_to_ngff_image_with_zarr_group(tmp_path):
    """Test that to_ngff_image can handle a group by selecting the first array."""
    # Create a zarr group with multiple arrays (simulating multi-series structure)
    store_path = tmp_path / "test.zarr"
    root = zarr.open_group(str(store_path), mode="w")

    # Create two arrays in the group (like multi-series TIFF)
    data0 = np.random.rand(10, 100, 100).astype(np.float32)
    data1 = np.random.rand(5, 50, 50).astype(np.float32)

    arr0 = root.create_array("0", shape=data0.shape, dtype=data0.dtype)
    arr0[:] = data0
    arr1 = root.create_array("1", shape=data1.shape, dtype=data1.dtype)
    arr1[:] = data1

    # Reopen through the compat layer (what the read pipeline hands over)
    reopened_group = _open_group(store_path)

    # Test that to_ngff_image handles the group correctly
    # It should use the first array in the group
    ngff_image = to_ngff_image(reopened_group)

    # Verify the result uses the first array
    assert ngff_image.dims == ("z", "y", "x")
    assert ngff_image.data.shape == (10, 100, 100)
    assert ngff_image.scale == {"z": 1.0, "y": 1.0, "x": 1.0}
    assert ngff_image.translation == {"z": 0.0, "y": 0.0, "x": 0.0}


def test_to_ngff_image_with_empty_zarr_group(tmp_path):
    """Test that to_ngff_image raises an error for an empty group."""
    store_path = tmp_path / "empty.zarr"
    zarr.open_group(str(store_path), mode="w")

    reopened_group = _open_group(store_path)

    # Test that to_ngff_image raises ValueError for empty group
    with pytest.raises(ValueError, match="no arrays found"):
        to_ngff_image(reopened_group)


def test_to_ngff_image_group_with_custom_metadata(tmp_path):
    """Test that to_ngff_image with a group respects custom metadata parameters."""
    store_path = tmp_path / "test.zarr"
    root = zarr.open_group(str(store_path), mode="w")

    data = np.random.rand(5, 64, 64).astype(np.float32)
    arr = root.create_array("data", shape=data.shape, dtype=data.dtype)
    arr[:] = data

    reopened_group = _open_group(store_path)

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
