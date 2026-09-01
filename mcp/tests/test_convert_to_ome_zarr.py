# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for convert_to_ome_zarr function."""

from pathlib import Path

import pytest

from ngff_zarr_mcp.models import ConversionOptions
from ngff_zarr_mcp.tools import convert_to_ome_zarr

# ``test_input_file`` and ``temp_output_dir`` come from tests/conftest.py.


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


@pytest.mark.asyncio
async def test_convert_with_anatomical_orientation(test_input_file, temp_output_dir):
    """The ``anatomical_orientation`` preset is applied and written to the output."""
    # zarr-python is a test-only dependency; reading the store back with it
    # doubles as a compatibility check of zarrista-written output.
    import zarr

    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    output_path = Path(temp_output_dir) / "mr_head_orientation.ome.zarr"

    # Request the RAS coordinate system explicitly.
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        scale_factors=None,
        chunks=64,
        anatomical_orientation="RAS",
    )

    result = await convert_to_ome_zarr([str(test_input_file)], options)
    assert result.success, f"Conversion failed: {result.error}"
    assert output_path.exists()

    # Read the written metadata; v0.4 stores multiscales at the root.
    zarr_group = zarr.open(str(output_path), mode="r")
    raw_attrs = dict(zarr_group.attrs)
    if "ome" in raw_attrs:
        multiscales_metadata = raw_attrs["ome"]["multiscales"][0]
    else:
        multiscales_metadata = raw_attrs["multiscales"][0]
    axes = multiscales_metadata["axes"]

    # RAS preset values must be written to the spatial axes.
    expected = {
        "x": "left-to-right",
        "y": "posterior-to-anterior",
        "z": "inferior-to-superior",
    }
    by_name = {ax["name"]: ax for ax in axes}
    for dim, value in expected.items():
        assert by_name[dim]["orientation"]["type"] == "anatomical"
        assert by_name[dim]["orientation"]["value"] == value
