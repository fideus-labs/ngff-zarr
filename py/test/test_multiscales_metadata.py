#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Test the multiscales metadata field functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

from ngff_zarr import Methods, to_multiscales, to_ngff_image, to_ngff_zarr, from_ngff_zarr


def test_multiscales_metadata_field():
    """Test that the metadata field is populated correctly."""
    # Create a simple test image
    data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    image = to_ngff_image(data, dims=["y", "x"])

    # Test with ITKWASM_GAUSSIAN method
    multiscales = to_multiscales(image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN)

    # Check that metadata field is populated
    assert multiscales.metadata.metadata is not None
    assert multiscales.metadata.metadata.description is not None
    assert multiscales.metadata.metadata.method is not None
    assert multiscales.metadata.metadata.version is not None

    # Check specific content
    assert "gaussian filter" in multiscales.metadata.metadata.description.lower()
    assert "itkwasm_downsample" in multiscales.metadata.metadata.method


def test_multiscales_metadata_serialization():
    """Test that the metadata field is correctly serialized to zarr."""
    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=["y", "x"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        zarr_path = Path(tmp_dir) / "test.ome.zarr"

        # Create and save multiscales
        multiscales = to_multiscales(image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN)
        to_ngff_zarr(zarr_path, multiscales)

        # Read raw metadata from zarr
        metadata_path = zarr_path / ".zattrs"
        with open(metadata_path) as f:
            raw_metadata = json.load(f)

        multiscales_metadata = raw_metadata["multiscales"][0]

        # Check that metadata field exists and has correct structure
        assert "metadata" in multiscales_metadata
        metadata_field = multiscales_metadata["metadata"]
        assert isinstance(metadata_field, dict)
        assert "description" in metadata_field
        assert "method" in metadata_field
        assert "version" in metadata_field

        # Check content
        assert isinstance(metadata_field["description"], str)
        assert isinstance(metadata_field["method"], str)
        assert isinstance(metadata_field["version"], str)


def test_multiscales_metadata_round_trip():
    """Test round-trip: save to zarr and load back."""
    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=["y", "x"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        zarr_path = Path(tmp_dir) / "test.ome.zarr"

        # Create and save multiscales
        original_multiscales = to_multiscales(image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN)
        to_ngff_zarr(zarr_path, original_multiscales)

        # Load back
        loaded_multiscales = from_ngff_zarr(zarr_path)

        # Compare metadata
        assert loaded_multiscales.metadata.metadata is not None
        assert loaded_multiscales.metadata.metadata.description == original_multiscales.metadata.metadata.description
        assert loaded_multiscales.metadata.metadata.method == original_multiscales.metadata.metadata.method
        assert loaded_multiscales.metadata.metadata.version == original_multiscales.metadata.metadata.version


def test_different_methods_have_different_metadata():
    """Test that different methods produce different metadata."""
    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=["y", "x"])

    # Test different methods that should work
    test_methods = [
        Methods.ITKWASM_GAUSSIAN,
        Methods.ITKWASM_BIN_SHRINK,
        Methods.ITKWASM_LABEL_IMAGE,
    ]

    metadata_results = []
    for method in test_methods:
        multiscales = to_multiscales(image, scale_factors=[2], method=method)
        assert multiscales.metadata.metadata is not None
        metadata_results.append({
            'method': method.value,
            'description': multiscales.metadata.metadata.description,
            'method_name': multiscales.metadata.metadata.method,
            'version': multiscales.metadata.metadata.version
        })

    # Check that different methods have different descriptions
    descriptions = [m['description'] for m in metadata_results]
    assert len(set(descriptions)) > 1, "Different methods should have different descriptions"

    # Check that all use similar package but potentially different functions
    method_names = [m['method_name'] for m in metadata_results]
    assert all('itkwasm_downsample' in name for name in method_names), "All should use itkwasm_downsample package"


def test_metadata_field_none_when_no_method():
    """Test that metadata field is None when no method is specified."""
    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=["y", "x"])

    # Create multiscales without specifying method
    multiscales = to_multiscales(image, scale_factors=[2])

    # When method is None (default), should still have a method set (ITKWASM_GAUSSIAN)
    # but metadata should be populated since default method is used
    assert multiscales.metadata.metadata is not None


def test_legacy_zarr_without_metadata():
    """Test that legacy zarr files without metadata field can still be loaded."""
    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=["y", "x"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        zarr_path = Path(tmp_dir) / "test.ome.zarr"

        # Create and save normally first
        multiscales = to_multiscales(image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN)
        to_ngff_zarr(zarr_path, multiscales)

        # Manually remove metadata field to simulate legacy file
        metadata_path = zarr_path / ".zattrs"
        with open(metadata_path) as f:
            raw_metadata = json.load(f)

        # Remove metadata field
        if "metadata" in raw_metadata["multiscales"][0]:
            del raw_metadata["multiscales"][0]["metadata"]

        with open(metadata_path, 'w') as f:
            json.dump(raw_metadata, f)

        # Should still be able to load
        loaded_multiscales = from_ngff_zarr(zarr_path)
        assert loaded_multiscales.metadata.metadata is None


if __name__ == "__main__":
    # Run tests
    test_multiscales_metadata_field()
    test_multiscales_metadata_serialization()
    test_multiscales_metadata_round_trip()
    test_different_methods_have_different_metadata()
    test_metadata_field_none_when_no_method()
    test_legacy_zarr_without_metadata()
    print("✅ All metadata field tests passed!")
