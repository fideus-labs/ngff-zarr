# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Basic tests for ngff-zarr MCP server."""

import pytest

from ngff_zarr_mcp.models import ConversionOptions
from ngff_zarr_mcp.utils import get_available_methods, get_supported_formats


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


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_supported_ome_zarr_versions(version):
    """Every released OME-Zarr version the writer supports is accepted."""
    from ngff_zarr_mcp.utils import validate_conversion_options

    options = ConversionOptions(output_path="test.ome.zarr", ome_zarr_version=version)
    assert options.ome_zarr_version == version
    assert validate_conversion_options(options.model_dump()) == []


def test_invalid_ome_zarr_version():
    """An unknown OME-Zarr version is rejected by the model and the validator."""
    from ngff_zarr_mcp.utils import validate_conversion_options

    with pytest.raises(ValueError):
        ConversionOptions(output_path="test.ome.zarr", ome_zarr_version="0.3")
    errors = validate_conversion_options(
        {"output_path": "test.ome.zarr", "ome_zarr_version": "0.3"}
    )
    assert any("OME-Zarr version" in error for error in errors)


def test_invalid_dims():
    """Test validation of invalid dimensions."""
    with pytest.raises(ValueError):
        ConversionOptions(output_path="test.ome.zarr", dims=["invalid", "dims"])
