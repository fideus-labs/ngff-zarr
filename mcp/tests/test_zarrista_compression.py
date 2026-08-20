# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Compression coverage for the zarrista-backed write path.

zarr-python is a test-only dependency here: reading the written stores back
with it verifies that other zarr implementations can decode zarrista output.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from ngff_zarr_mcp.models import ConversionOptions
from ngff_zarr_mcp.tools import convert_to_ome_zarr, validate_ome_zarr


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


def _read_first_scale(output_path: Path) -> np.ndarray:
    """Read the first-scale array back with zarr-python.

    A successful read proves zarr-python can decode the compressed chunks
    zarrista wrote.
    """
    group = zarr.open(str(output_path), mode="r")
    attrs = dict(group.attrs)
    metadata = attrs["ome"] if "ome" in attrs else attrs
    dataset_path = metadata["multiscales"][0]["datasets"][0]["path"]
    return np.asarray(group[dataset_path])


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_with_blosc_compression(
    test_input_file, temp_output_dir
):
    """Test conversion with blosc compression on the default zarrista path."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_blosc_compression.ome.zarr"

    # Configure conversion options with blosc compression
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        compression_codec="blosc",
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

    # Verify zarr-python can decompress the written chunks
    data = _read_first_scale(output_path)
    assert data.size > 0


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_with_gzip_compression(
    test_input_file, temp_output_dir
):
    """Test conversion with gzip compression on the default zarrista path."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_gzip_compression.ome.zarr"

    # Configure conversion options with gzip compression
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

    # Verify store info
    store_info = result.store_info
    assert store_info is not None
    assert store_info.get("num_scales", 0) > 0

    # Verify zarr-python can decompress the written chunks
    data = _read_first_scale(output_path)
    assert data.size > 0


@pytest.mark.asyncio
async def test_convert_to_ome_zarr_zarr_v3_with_compression(
    test_input_file, temp_output_dir
):
    """Test conversion with Zarr v3 and blosc compression on the zarrista path."""
    # Verify input file exists
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    # Setup output path
    output_path = Path(temp_output_dir) / "mr_head_zarr_v3_compression.ome.zarr"

    # Configure conversion options with Zarr v3 and compression
    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.5",  # Uses Zarr v3
        method="itkwasm_gaussian",
        compression_codec="blosc",
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

    # Verify zarr-python can decompress the written chunks
    data = _read_first_scale(output_path)
    assert data.size > 0


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error:use_tensorstore is deprecated")
async def test_use_tensorstore_alias_contract(test_input_file, temp_output_dir):
    """``use_tensorstore=True`` still succeeds and produces a valid store.

    The flag is a deprecated alias for the zarrista backend; the resulting
    DeprecationWarning must not leak out of the MCP tool (the filterwarnings
    marker turns a leak into a test failure).
    """
    assert test_input_file.exists(), f"Test input file not found: {test_input_file}"

    output_path = Path(temp_output_dir) / "mr_head_tensorstore_alias.ome.zarr"

    options = ConversionOptions(
        output_path=str(output_path),
        ome_zarr_version="0.4",
        method="itkwasm_gaussian",
        compression_codec="blosc",
        use_tensorstore=True,
        chunks=64,
    )

    result = await convert_to_ome_zarr([str(test_input_file)], options)

    assert result.success, f"Conversion failed: {result.error}"
    assert output_path.exists(), "Output OME-Zarr store was not created"

    store_info = result.store_info
    assert store_info is not None
    assert store_info.get("num_scales", 0) > 0

    validation = await validate_ome_zarr(str(output_path))
    assert validation.valid, f"Store failed validation: {validation.errors}"
