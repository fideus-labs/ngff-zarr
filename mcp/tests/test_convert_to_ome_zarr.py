# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for convert_to_ome_zarr function."""

import pytest
import tempfile
import shutil
from pathlib import Path

from ngff_zarr_mcp.tools import convert_to_ome_zarr
from ngff_zarr_mcp.models import ConversionOptions


@pytest.fixture
def test_input_file():
    """Path to the test input file."""
    return Path(__file__).parent.parent / "test" / "data" / "input" / "MR-head.nrrd"


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_basic(test_input_file, temp_output_dir):
    """Test basic conversion of MR-head.nrrd to OME-Zarr."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head.ome.zarr"

    # Configure conversion options
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        # Single scale (no multiscale)
        scale_factors=None,
        # Use small chunks for testing
        chunks=64,
    )

    # Perform conversion
    result = await convert_to_ome_zarr([str(test_input_file)], options)

    # Verify conversion succeeded
    assert result.success, f"Conversion failed: {result.error}"
    assert result.output_path == str(output_path)
    assert output_path.exists(), "Output OME-Zarr store was not created"

    # Verify store info
    store_info = result.store_info
    assert store_info is not None
    assert store_info.get("num_scales", 0) > 0
    assert store_info.get("version") == "0.4"
    assert len(store_info.get("dimensions", [])) > 0
    assert len(store_info.get("shape", [])) > 0


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_with_multiscale(test_input_file, temp_output_dir):
    """Test conversion with multiscale generation."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_multiscale.ome.zarr"

    # Configure conversion options with multiscale
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        # Generate 3 scales with 2x downsampling
        scale_factors=[2, 2],
        chunks=64,
    )

    # Perform conversion
    result = await convert_to_ome_zarr([str(test_input_file)], options)

    # Verify conversion succeeded
    assert result.success, f"Conversion failed: {result.error}"
    assert result.output_path == str(output_path)
    assert output_path.exists(), "Output OME-Zarr store was not created"

    # Verify multiscale was created
    store_info = result.store_info
    assert store_info is not None
    assert store_info.get("num_scales", 0) >= 2, "Expected at least 2 scales"


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_with_metadata(test_input_file, temp_output_dir):
    """Test conversion with custom metadata."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_metadata.ome.zarr"

    # Configure conversion options with metadata
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        name="MR Head Test Image",
        dims=["z", "y", "x"],
        scale={"z": 1.25, "y": 0.9375, "x": 0.9375},
        units={"z": "millimeter", "y": "millimeter", "x": "millimeter"},
        chunks=64,
    )

    # Perform conversion
    result = await convert_to_ome_zarr([str(test_input_file)], options)

    # Verify conversion succeeded
    assert result.success, f"Conversion failed: {result.error}"
    assert result.output_path == str(output_path)
    assert output_path.exists(), "Output OME-Zarr store was not created"

    # Verify metadata was applied
    store_info = result.store_info
    assert store_info is not None
    assert store_info.get("dimensions") == ["z", "y", "x"]

    # Check that scale information was preserved
    scale_info = store_info.get("scale_info", {})
    assert "z" in scale_info
    assert "y" in scale_info
    assert "x" in scale_info


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_with_compression(test_input_file, temp_output_dir):
    """Test conversion with different compression settings."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_compressed.ome.zarr"

    # Configure conversion options with compression
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        compression_codec="gzip",
        compression_level=6,
        chunks=64,
    )

    # Perform conversion
    result = await convert_to_ome_zarr([str(test_input_file)], options)

    # Verify conversion succeeded
    assert result.success, f"Conversion failed: {result.error}"
    assert result.output_path == str(output_path)
    assert output_path.exists(), "Output OME-Zarr store was not created"

    # Verify compression was applied (if detectable)
    store_info = result.store_info
    assert store_info is not None
    # Note: compression detection depends on implementation details


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_zarr_v05(test_input_file, temp_output_dir):
    """Test conversion to OME-Zarr v0.5 format."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_v05.ome.zarr"

    # Configure conversion options for v0.5
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.5",
        method="itkwasm_gaussian",
        chunks=64,
    )

    # Perform conversion
    result = await convert_to_ome_zarr([str(test_input_file)], options)

    # Verify conversion succeeded
    assert result.success, f"Conversion failed: {result.error}"
    assert result.output_path == str(output_path)
    assert output_path.exists(), "Output OME-Zarr store was not created"

    # Verify version
    store_info = result.store_info
    assert store_info is not None
    assert store_info.get("version") == "0.5"


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_invalid_input():
    """Test conversion with invalid input file."""
    # Use a non-existent file
    invalid_input = "/nonexistent/file.nrrd"

    # Setup output path
    output_path = "/tmp/should_not_be_created.ome.zarr"

    # Configure conversion options
    options = ConversionOptions(
        output_path=output_path,
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
    )

    # Perform conversion
    result = await convert_to_ome_zarr([invalid_input], options)

    # Verify conversion failed
    assert not result.success
    assert result.error is not None
    assert len(result.error) > 0


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_invalid_options(test_input_file, temp_output_dir):
    """Test conversion with invalid options."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_invalid.ome.zarr"

    # Configure conversion options with invalid dimensions
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="invalid_method",  # This should cause validation to fail
        dims=["z", "y", "x"],
    )

    # Perform conversion
    result = await convert_to_ome_zarr([str(test_input_file)], options)

    # Verify conversion failed due to validation
    assert not result.success
    assert result.error is not None
    assert "validation" in result.error.lower() or "method" in result.error.lower()
