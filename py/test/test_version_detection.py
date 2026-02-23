# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for OME-Zarr version detection."""

import pytest
import numpy as np
import packaging.version
import zarr
from zarr.storage import MemoryStore

from ngff_zarr import to_ngff_zarr, from_ngff_zarr, to_multiscales, to_ngff_image
from ngff_zarr.parse_metadata import _detect_version
from ngff_zarr._supported_versions import NgffVersion


zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major


def test_detect_version_v04():
    """Test version detection for v0.4 format."""
    root_attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ]
    }
    version = _detect_version(root_attrs)
    assert version == NgffVersion.V04


def test_detect_version_v04_no_version_key():
    """Test version detection for v0.4 format without explicit version key."""
    root_attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ]
    }
    version = _detect_version(root_attrs)
    # Should default to 0.4 when multiscales exists but no version
    assert version == NgffVersion.V04


def test_detect_version_v05():
    """Test version detection for v0.5 format."""
    root_attrs = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ],
        }
    }
    version = _detect_version(root_attrs)
    assert version == NgffVersion.V05


def test_detect_version_v05_no_version_key():
    """Test version detection for v0.5 format without explicit version key."""
    root_attrs = {
        "ome": {
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ]
        }
    }
    version = _detect_version(root_attrs)
    # Should default to 0.5 when 'ome' key exists
    assert version == NgffVersion.V05


def test_detect_version_missing_keys():
    """Test version detection fails gracefully with missing keys."""
    root_attrs = {}
    with pytest.raises(ValueError, match="Could not detect NGFF version"):
        _detect_version(root_attrs)


def test_detect_version_empty_root_attrs_detailed_error():
    """Test that empty root attributes provide detailed error message."""
    root_attrs = {}
    with pytest.raises(ValueError, match="root attributes are empty"):
        _detect_version(root_attrs)


def test_detect_version_malformed_ome_key():
    """Test that malformed 'ome' key (not a dict) gives helpful error."""
    # Test with ome as null/None
    root_attrs = {"ome": None}
    with pytest.raises(ValueError, match="'ome' key must be a dictionary"):
        _detect_version(root_attrs)

    # Test with ome as string
    root_attrs = {"ome": "0.5"}
    with pytest.raises(ValueError, match="'ome' key must be a dictionary"):
        _detect_version(root_attrs)

    # Test with ome as number
    root_attrs = {"ome": 0.5}
    with pytest.raises(ValueError, match="'ome' key must be a dictionary"):
        _detect_version(root_attrs)


def test_detect_version_malformed_multiscales_not_list():
    """Test that multiscales as non-list gives helpful error."""
    # Test with multiscales as dict
    root_attrs = {"multiscales": {"version": "0.4"}}
    with pytest.raises(ValueError, match="'multiscales' must be a list"):
        _detect_version(root_attrs)

    # Test with multiscales as string
    root_attrs = {"multiscales": "invalid"}
    with pytest.raises(ValueError, match="'multiscales' must be a list"):
        _detect_version(root_attrs)


def test_detect_version_malformed_multiscales_invalid_elements():
    """Test that multiscales list with non-dict elements gives helpful error."""
    # Test with multiscales containing non-dict element
    root_attrs = {"multiscales": ["invalid"]}
    with pytest.raises(
        ValueError, match="'multiscales' list must contain dictionaries"
    ):
        _detect_version(root_attrs)

    # Test with multiscales containing None
    root_attrs = {"multiscales": [None]}
    with pytest.raises(
        ValueError, match="'multiscales' list must contain dictionaries"
    ):
        _detect_version(root_attrs)


def test_from_ngff_zarr_v04_auto_detect():
    """Test loading v0.4 OME-Zarr with automatic version detection."""
    # Create test data
    data = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = to_ngff_image(data, dims=("y", "x"))
    multiscales = to_multiscales(image)

    # Write to v0.4 format
    store = MemoryStore()
    to_ngff_zarr(store, multiscales, version="0.4")

    # Load without specifying version (should auto-detect)
    loaded = from_ngff_zarr(store)

    assert loaded is not None
    assert len(loaded.images) > 0
    assert loaded.images[0].data.shape == (64, 64)


@pytest.mark.skipif(zarr_version_major < 3, reason="v0.5 format requires zarr-python 3")
def test_from_ngff_zarr_v05_auto_detect():
    """Test loading v0.5 OME-Zarr with automatic version detection."""
    # Create test data
    data = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = to_ngff_image(data, dims=("y", "x"))
    multiscales = to_multiscales(image)

    # Write to v0.5 format
    store = MemoryStore()
    to_ngff_zarr(store, multiscales, version="0.5")

    # Load without specifying version (should auto-detect)
    loaded = from_ngff_zarr(store)

    assert loaded is not None
    assert len(loaded.images) > 0
    assert loaded.images[0].data.shape == (64, 64)


def test_from_ngff_zarr_v04_wrong_structure():
    """Test that claiming v0.4 with v0.5 structure (ome key) gives helpful error.

    When a file has the 'ome' key (v0.5 structure) but we try to load it as v0.4,
    it should fail because v0.4 expects 'multiscales' at root level, not under 'ome'.
    """
    # Create a store with v0.5 structure (has 'ome' key)
    store = MemoryStore()
    # Use zarr_format=2 to match what from_ngff_zarr will use for v0.4
    # (zarr_format parameter only exists in zarr 3.x)
    if zarr_version_major >= 3:
        root = zarr.open_group(store, mode="w", zarr_format=2)
    else:
        root = zarr.open_group(store, mode="w")
    # Create the array data that the metadata references
    data_array = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    # zarr 2.x uses create_dataset, zarr 3.x uses create_array
    if zarr_version_major >= 3:
        root.create_array("0", data=data_array, chunks=(32, 32))
    else:
        root.create_dataset("0", data=data_array, chunks=(32, 32))

    root.attrs["ome"] = {
        "version": "0.4",  # Incorrect: v0.4 files have multiscales at root, not under 'ome'
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ],
    }

    # When loading with version="0.4", should fail because 'multiscales' is not at root level
    with pytest.raises(ValueError, match="multiscales"):
        from_ngff_zarr(store, version="0.4")


@pytest.mark.skipif(zarr_version_major < 3, reason="v0.5 format requires zarr-python 3")
def test_from_ngff_zarr_v05_wrong_structure():
    """Test that v0.5 format with wrong structure gives helpful error."""
    # Create a store with v0.4 structure
    store = MemoryStore()
    # Use zarr_format=3 to match what from_ngff_zarr will use for v0.5
    root = zarr.open_group(store, mode="w", zarr_format=3)
    # Create the array data that the metadata references
    data_array = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    root.create_array("0", data=data_array, chunks=(32, 32))

    root.attrs["multiscales"] = [
        {
            "version": "0.5",  # Claims v0.5 but has v0.4 structure
            "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [{"path": "0"}],
        }
    ]

    # When loading with version="0.5", should fail with helpful error
    with pytest.raises(ValueError, match="ome"):
        from_ngff_zarr(store, version="0.5")


def test_from_ngff_zarr_empty_multiscales():
    """Test that empty multiscales list gives helpful error."""
    store = MemoryStore()
    # Use zarr_format=2 for v0.4 style
    # (zarr_format parameter only exists in zarr 3.x)
    if zarr_version_major >= 3:
        root = zarr.open_group(store, mode="w", zarr_format=2)
    else:
        root = zarr.open_group(store, mode="w")
    # Create a dummy array so the group is valid
    data_array = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    # zarr 2.x uses create_dataset, zarr 3.x uses create_array
    if zarr_version_major >= 3:
        root.create_array("0", data=data_array, chunks=(32, 32))
    else:
        root.create_dataset("0", data=data_array, chunks=(32, 32))
    root.attrs["multiscales"] = []  # Empty list

    with pytest.raises(ValueError, match="Could not detect NGFF version"):
        from_ngff_zarr(store)


def test_detect_version_v04_hcs_plate():
    """Test version detection for v0.4 HCS plate format.

    HCS plates with 'plate' key at root level use v0.4 format.
    """
    root_attrs = {
        "plate": {
            "name": "test_plate",
            "columns": [{"name": "1"}],
            "rows": [{"name": "A"}],
            "wells": [{"path": "A/1"}],
        }
    }
    version = _detect_version(root_attrs)
    # HCS plates without explicit version should default to 0.4
    assert version == NgffVersion.V04


def test_detect_version_v05_hcs_plate():
    """Test version detection for v0.5 HCS plate format."""
    root_attrs = {
        "ome": {
            "version": "0.5",
            "plate": {
                "name": "test_plate",
                "columns": [{"name": "1"}],
                "rows": [{"name": "A"}],
                "wells": [{"path": "A/1"}],
            },
        }
    }
    version = _detect_version(root_attrs)
    assert version == NgffVersion.V05


def test_from_ngff_zarr_hcs_plate_error():
    """Test that trying to load HCS plate with from_ngff_zarr gives helpful error."""
    store = MemoryStore()
    if zarr_version_major >= 3:
        root = zarr.open_group(store, mode="w", zarr_format=2)
    else:
        root = zarr.open_group(store, mode="w")

    # Create HCS plate structure
    root.attrs["plate"] = {
        "name": "test_plate",
        "columns": [{"name": "1"}],
        "rows": [{"name": "A"}],
        "wells": [{"path": "A/1"}],
    }

    # Should raise ValueError with helpful message about using from_hcs_zarr
    with pytest.raises(ValueError, match="(?s)HCS.*from_hcs_zarr"):
        from_ngff_zarr(store)


@pytest.mark.skipif(zarr_version_major < 3, reason="v0.5 format requires zarr-python 3")
def test_from_ngff_zarr_v05_ome_key_without_multiscales():
    """Test that v0.5 format with ome key but missing multiscales gives helpful error.

    This can happen with corrupted files or files with ome key containing other metadata
    but not the required multiscales.
    """
    store = MemoryStore()
    # Use zarr_format=3 for v0.5
    root = zarr.open_group(store, mode="w", zarr_format=3)
    # Create the array data
    data_array = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    root.create_array("0", data=data_array, chunks=(32, 32))

    # Set ome key but without multiscales (corrupted/incomplete file)
    root.attrs["ome"] = {
        "version": "0.5",
        "some_other_metadata": "value",
    }

    # Should raise ValueError with helpful error explaining possible causes
    with pytest.raises(
        ValueError, match=r"multiscales[\s\S]*missing[\s\S]*Possible causes"
    ) as exc_info:
        from_ngff_zarr(store)

    # Verify error message contains all expected guidance
    error_msg = str(exc_info.value)
    assert "Available keys under 'ome'" in error_msg
    assert "corrupted" in error_msg.lower() or "incomplete" in error_msg.lower()
