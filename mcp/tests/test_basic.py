# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Basic tests for ngff-zarr MCP server."""

import pytest

from ngff_zarr_mcp.models import ConversionOptions
from ngff_zarr_mcp.utils import get_supported_formats, get_available_methods


def test_get_supported_formats():
    """Test getting supported formats."""
    formats = get_supported_formats()
    assert "ngff_zarr" in formats.input_formats
    assert ".ome.zarr" in formats.output_formats
    assert len(formats.backends) > 0


def test_get_available_methods():
    """Test getting available methods."""
    methods = get_available_methods()
    assert "itkwasm_gaussian" in methods
    assert len(methods) > 0


@pytest.mark.asyncio
async def test_conversion_options_validation():
    """Test conversion options validation."""
    # Valid options should not raise errors
    options = ConversionOptions(
        output_path="test.ome.zarr",
        ome_zarr_version="0.4",
        dims=["z", "y", "x"],
        method="itkwasm_gaussian",
    )
    assert options.output_path == "test.ome.zarr"
    assert options.dims == ["z", "y", "x"]


def test_invalid_dims():
    """Test validation of invalid dimensions."""
    with pytest.raises(ValueError):
        ConversionOptions(output_path="test.ome.zarr", dims=["invalid", "dims"])
