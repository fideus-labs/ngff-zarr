# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for handling unknown fields in axis metadata."""
import asyncio
import json
import logging

import numpy as np
import packaging.version
import pytest
import zarr
from zarr.storage import MemoryStore

from ngff_zarr import from_ngff_zarr, to_multiscales, to_ngff_image, to_ngff_zarr

zarr_version = packaging.version.parse(zarr.__version__)


@pytest.fixture
def zarr_helpers():
    """Fixture providing helper functions for accessing zarr store attributes."""
    
    async def get_attrs(store):
        """Get attributes from a zarr store (v3 compatible)."""
        from zarr.core.buffer import default_buffer_prototype
        
        attrs_key = ".zattrs"
        attrs_bytes = await store.get(attrs_key, default_buffer_prototype())
        return json.loads(attrs_bytes.to_bytes().decode())
    
    async def set_attrs(store, attrs):
        """Set attributes in a zarr store (v3 compatible)."""
        from zarr.core.buffer import default_buffer_prototype
        
        attrs_key = ".zattrs"
        attrs_bytes = json.dumps(attrs).encode()
        proto = default_buffer_prototype()
        buffer = proto.buffer.from_bytes(attrs_bytes)
        await store.set(attrs_key, buffer)
    
    return {"get": get_attrs, "set": set_attrs}


def test_unknown_axis_fields_are_filtered(caplog, zarr_helpers):
    """Test that non-standard axis fields are filtered out and a warning is logged."""
    # Create a basic image
    data = np.random.rand(10, 20, 30).astype(np.float32)
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_image",
    )
    multiscales = to_multiscales(image, scale_factors=[])
    
    # Write to zarr store
    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    
    # Manually modify the metadata to add a non-standard field
    attrs = asyncio.run(zarr_helpers["get"](store))
    
    # Add a non-standard "discrete" field to the time axis (like BigStitcher-Spark does)
    for axis in attrs["multiscales"][0]["axes"]:
        if axis["name"] == "z":
            axis["discrete"] = False
            axis["custom_field"] = "custom_value"
    
    asyncio.run(zarr_helpers["set"](store, attrs))
    
    # Try to load the data - it should succeed and log warnings
    with caplog.at_level(logging.WARNING):
        multiscales_back = from_ngff_zarr(store, version=version)
    
    # Verify that warnings were logged
    assert len(caplog.records) > 0
    warning_messages = [record.message for record in caplog.records]
    assert any("Ignoring unknown fields" in msg for msg in warning_messages)
    assert any("discrete" in msg for msg in warning_messages)
    assert any("custom_field" in msg for msg in warning_messages)
    
    # Verify that the data loaded successfully
    assert multiscales_back is not None
    assert len(multiscales_back.images) > 0
    
    # Verify that the unknown fields are not present in the Axis objects
    axes = multiscales_back.metadata.axes
    z_axis = next((axis for axis in axes if axis.name == "z"), None)
    assert z_axis is not None
    assert not hasattr(z_axis, "discrete")
    assert not hasattr(z_axis, "custom_field")


def test_unknown_fields_multiple_axes(caplog, zarr_helpers):
    """Test that unknown fields in multiple axes are all logged."""
    # Create a 4D image
    data = np.random.rand(2, 10, 20, 30).astype(np.float32)
    image = to_ngff_image(
        data=data,
        dims=("t", "z", "y", "x"),
        scale={"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation={"t": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name="test_image",
    )
    multiscales = to_multiscales(image, scale_factors=[])
    
    # Write to zarr store
    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    
    # Manually modify the metadata to add non-standard fields to multiple axes
    attrs = asyncio.run(zarr_helpers["get"](store))
    
    for axis in attrs["multiscales"][0]["axes"]:
        if axis["name"] == "t":
            axis["discrete"] = True
        elif axis["name"] == "z":
            axis["continuous"] = True
    
    asyncio.run(zarr_helpers["set"](store, attrs))
    
    # Try to load the data
    with caplog.at_level(logging.WARNING):
        multiscales_back = from_ngff_zarr(store, version=version)
    
    # Verify warnings for both axes
    warning_messages = [record.message for record in caplog.records]
    assert any("discrete" in msg and "'t'" in msg for msg in warning_messages)
    assert any("continuous" in msg and "'z'" in msg for msg in warning_messages)
    
    # Verify the data loaded successfully
    assert multiscales_back is not None


def test_missing_required_name_field(zarr_helpers):
    """Test that missing 'name' field raises ValueError."""
    # Create a basic image
    data = np.random.rand(10, 20, 30).astype(np.float32)
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_image",
    )
    multiscales = to_multiscales(image, scale_factors=[])
    
    # Write to zarr store
    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    
    # Manually corrupt the metadata by removing 'name' field
    attrs = asyncio.run(zarr_helpers["get"](store))
    
    # Remove the 'name' field from one axis
    del attrs["multiscales"][0]["axes"][0]["name"]
    
    asyncio.run(zarr_helpers["set"](store, attrs))
    
    # Try to load the data - should raise ValueError
    with pytest.raises(ValueError, match="missing required field 'name'"):
        from_ngff_zarr(store, version=version)


def test_missing_required_type_field(zarr_helpers):
    """Test that missing 'type' field raises ValueError."""
    # Create a basic image
    data = np.random.rand(10, 20, 30).astype(np.float32)
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_image",
    )
    multiscales = to_multiscales(image, scale_factors=[])
    
    # Write to zarr store
    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    
    # Manually corrupt the metadata by removing 'type' field
    attrs = asyncio.run(zarr_helpers["get"](store))
    
    # Remove the 'type' field from one axis
    del attrs["multiscales"][0]["axes"][0]["type"]
    
    asyncio.run(zarr_helpers["set"](store, attrs))
    
    # Try to load the data - should raise ValueError
    with pytest.raises(ValueError, match="missing required field 'type'"):
        from_ngff_zarr(store, version=version)


def test_only_unknown_fields(zarr_helpers):
    """Test that an axis with only unknown fields (no valid fields) raises ValueError."""
    # Create a basic image
    data = np.random.rand(10, 20, 30).astype(np.float32)
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_image",
    )
    multiscales = to_multiscales(image, scale_factors=[])
    
    # Write to zarr store
    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    
    # Manually corrupt the metadata to have only unknown fields
    attrs = asyncio.run(zarr_helpers["get"](store))
    
    # Replace one axis with only unknown fields
    attrs["multiscales"][0]["axes"][0] = {
        "custom_field_1": "value1",
        "custom_field_2": "value2",
    }
    
    asyncio.run(zarr_helpers["set"](store, attrs))
    
    # Try to load the data - should raise ValueError about missing required fields
    with pytest.raises(ValueError, match="missing required field"):
        from_ngff_zarr(store, version=version)


def test_valid_optional_fields_preserved(zarr_helpers):
    """Test that valid optional fields (like 'unit') are preserved."""
    # Create a basic image with units
    data = np.random.rand(10, 20, 30).astype(np.float32)
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="test_image",
    )
    # Set units using the correct property name
    image.axes_units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
    multiscales = to_multiscales(image, scale_factors=[])
    
    # Write to zarr store
    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    
    # Manually add a non-standard field alongside the valid 'unit' field
    attrs = asyncio.run(zarr_helpers["get"](store))
    
    for axis in attrs["multiscales"][0]["axes"]:
        if axis["name"] == "z":
            axis["discrete"] = False  # Non-standard field
    
    asyncio.run(zarr_helpers["set"](store, attrs))
    
    # Load the data
    multiscales_back = from_ngff_zarr(store, version=version)
    
    # Verify that the valid 'unit' field is preserved
    z_axis = next(
        (axis for axis in multiscales_back.metadata.axes if axis.name == "z"), None
    )
    assert z_axis is not None
    assert z_axis.unit == "micrometer"
    
    # Verify that the non-standard field is not present
    assert not hasattr(z_axis, "discrete")
