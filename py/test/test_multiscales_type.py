# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from zarr.storage import MemoryStore

from ngff_zarr import Methods, to_multiscales, to_ngff_image, to_ngff_zarr, from_ngff_zarr


def test_multiscales_type_roundtrip():
    """Test that multiscales type field is preserved during write/read roundtrip."""
    # Create test data
    data = np.random.randint(0, 256, (32, 64, 64)).astype(np.uint8)
    image = to_ngff_image(data, dims=["z", "y", "x"])
    
    # Test with different methods
    test_methods = [
        Methods.ITKWASM_GAUSSIAN,
        Methods.ITKWASM_BIN_SHRINK,
        Methods.DASK_IMAGE_GAUSSIAN,
        Methods.ITK_GAUSSIAN,
    ]
    
    for method in test_methods:
        # Create multiscales with specific method
        multiscales = to_multiscales(image, scale_factors=[2, 4], method=method)
        
        # Verify method is set in multiscales
        assert multiscales.method == method
        
        # Write to zarr
        store = MemoryStore()
        to_ngff_zarr(store, multiscales)
        
        # Read back from zarr
        loaded_multiscales = from_ngff_zarr(store)
        
        # Verify method is preserved
        assert loaded_multiscales.method == method
        
        # Verify metadata type field contains the lowercase method name
        assert loaded_multiscales.metadata.type == method.value


def test_multiscales_type_in_metadata():
    """Test that the type field appears correctly in the multiscales metadata."""
    # Create test data
    data = np.random.randint(0, 256, (16, 32, 32)).astype(np.uint8)
    image = to_ngff_image(data, dims=["z", "y", "x"])
    
    # Create multiscales with gaussian method
    multiscales = to_multiscales(image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN)
    
    # Write to zarr
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    
    # Check the raw metadata contains the type field
    import zarr
    root = zarr.open_group(store, mode="r")
    metadata = root.attrs["multiscales"][0]
    
    assert "type" in metadata
    assert metadata["type"] == "itkwasm_gaussian"


def test_multiscales_type_none():
    """Test behavior when no method is specified."""
    # Create test data
    data = np.random.randint(0, 256, (16, 32, 32)).astype(np.uint8)
    image = to_ngff_image(data, dims=["z", "y", "x"])
    
    # Create multiscales without specifying method (should default to ITKWASM_GAUSSIAN)
    multiscales = to_multiscales(image, scale_factors=[2])
    
    # Write to zarr
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    
    # Read back from zarr
    loaded_multiscales = from_ngff_zarr(store)
    
    # Should have default method
    assert loaded_multiscales.method == Methods.ITKWASM_GAUSSIAN
    assert loaded_multiscales.metadata.type == "itkwasm_gaussian"


def test_legacy_zarr_without_type():
    """Test reading zarr files that don't have the type field (backward compatibility)."""
    # Create test data
    data = np.random.randint(0, 256, (16, 32, 32)).astype(np.uint8)
    image = to_ngff_image(data, dims=["z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2])
    
    # Write to zarr
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    
    # Manually remove the type field to simulate legacy data
    import zarr
    root = zarr.open_group(store, mode="a")
    metadata = root.attrs["multiscales"][0]
    if "type" in metadata:
        del metadata["type"]
    root.attrs["multiscales"] = [metadata]
    
    # Read back from zarr
    loaded_multiscales = from_ngff_zarr(store)
    
    # Should handle missing type gracefully
    assert loaded_multiscales.method is None
    assert loaded_multiscales.metadata.type is None
