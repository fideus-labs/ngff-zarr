# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import pytest
import tempfile
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
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_tensorstore_with_compression(
    test_input_file, temp_output_dir
):
    """Test conversion with TensorStore and compression (reproduces the original bug)."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_tensorstore_compression.ome.zarr"

    # Configure conversion options with TensorStore and compression
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        compression_codec="blosc",
        use_tensorstore=True,
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


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_tensorstore_with_gzip_compression(
    test_input_file, temp_output_dir
):
    """Test conversion with TensorStore and gzip compression."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_tensorstore_gzip.ome.zarr"

    # Configure conversion options with TensorStore and gzip compression
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        compression_codec="gzip",
        compression_level=6,
        use_tensorstore=True,
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


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_zarr_v3_tensorstore_with_compression(
    test_input_file, temp_output_dir
):
    """Test conversion with TensorStore, Zarr v3, and compression."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = (
        Path(temp_output_dir) / "mr_head_zarr_v3_tensorstore_compression.ome.zarr"
    )

    # Configure conversion options with TensorStore, Zarr v3, and compression
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.5",  # Uses Zarr v3
        method="itkwasm_gaussian",
        compression_codec="blosc",
        use_tensorstore=True,
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
    assert store_info.get("version") == "0.5"
