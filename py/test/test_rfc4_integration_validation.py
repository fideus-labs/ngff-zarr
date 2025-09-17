# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Integration test for RFC4 validation with from_ngff_zarr using real data."""

import pytest
import numpy as np
import dask.array as da
from zarr.storage import MemoryStore
from ngff_zarr import to_ngff_zarr, from_ngff_zarr, to_ngff_image, to_multiscales
from ngff_zarr.ngff_image import NgffImage
from ngff_zarr.rfc4 import LPS, RAS


def test_rfc4_validation_integration_with_lps():
    """Test RFC4 validation works with LPS coordinate system in a full workflow."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Create test data
    data = np.random.randint(0, 255, size=(10, 20, 30), dtype=np.uint8)
    dask_data = da.from_array(data)

    # Create NgffImage with LPS orientation
    ngff_image = NgffImage(
        data=dask_data,
        dims=("z", "y", "x"),
        scale={"z": 2.5, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_lps",
        axes_orientations=LPS,
    )

    # Convert to multiscales and store to zarr
    multiscales = to_multiscales(ngff_image)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales, version="0.4", enabled_rfcs=[4])

    # Read back with validation enabled - should pass (no validation errors)
    multiscales_back = from_ngff_zarr(store, validate=True)
    assert multiscales_back is not None
    assert len(multiscales_back.images) == 1


def test_rfc4_validation_integration_with_ras():
    """Test RFC4 validation works with RAS coordinate system in a full workflow."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Create test data
    data = np.random.randint(0, 255, size=(5, 10, 15), dtype=np.uint8)
    dask_data = da.from_array(data)

    # Create NgffImage with RAS orientation
    ngff_image = NgffImage(
        data=dask_data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 0.5, "x": 0.5},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_ras",
        axes_orientations=RAS,
    )

    # Convert to multiscales and store to zarr
    multiscales = to_multiscales(ngff_image)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales, version="0.4", enabled_rfcs=[4])

    # Read back with validation enabled - should pass (no validation errors)
    multiscales_back = from_ngff_zarr(store, validate=True)
    assert multiscales_back is not None
    assert len(multiscales_back.images) == 1


def test_rfc4_validation_integration_mixed_axes():
    """Test RFC4 validation works with time and channel axes mixed in."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Create test data with time and channel dimensions
    data = np.random.randint(0, 255, size=(3, 2, 5, 10, 15), dtype=np.uint8)
    dask_data = da.from_array(data)

    # Create NgffImage with time, channel, and spatial axes with LPS orientation
    ngff_image = NgffImage(
        data=dask_data,
        dims=("t", "c", "z", "y", "x"),
        scale={"t": 1.0, "c": 1.0, "z": 1.0, "y": 0.5, "x": 0.5},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name="test_mixed",
        axes_orientations=LPS,  # Only applies to spatial axes
    )

    # Convert to multiscales and store to zarr
    multiscales = to_multiscales(ngff_image)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales, version="0.4", enabled_rfcs=[4])

    # Read back with validation enabled - should pass (no validation errors)
    multiscales_back = from_ngff_zarr(store, validate=True)
    assert multiscales_back is not None
    assert len(multiscales_back.images) == 1


def test_rfc4_validation_no_orientation():
    """Test that validation passes when no orientation metadata is present."""
    pytest.importorskip("jsonschema", reason="jsonschema required for RFC 4 validation")

    # Create test data
    data = np.random.randint(0, 255, size=(10, 20, 30), dtype=np.uint8)

    # Create NgffImage without orientation metadata - use to_ngff_image
    ngff_image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 2.5, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_no_orientation",
        # No axes_orientations parameter
    )

    # Convert to multiscales and store to zarr
    multiscales = to_multiscales(ngff_image)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales, version="0.4")

    # Read back with validation enabled - should pass (no orientation to validate)
    multiscales_back = from_ngff_zarr(store, validate=True)
    assert multiscales_back is not None
    assert len(multiscales_back.images) == 1


if __name__ == "__main__":
    test_rfc4_validation_integration_with_lps()
    test_rfc4_validation_integration_with_ras()
    test_rfc4_validation_integration_mixed_axes()
    test_rfc4_validation_no_orientation()
    print("All integration tests passed!")
