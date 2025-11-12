# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for RFC-9: Zipped OME-Zarr (.ozx) support"""

import json
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr
from dask_image import imread

from ngff_zarr import (
    Methods,
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
    config,
)
from ngff_zarr.rfc9_zip import is_ozx_path, read_ozx_version

from ._data import verify_against_baseline

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

# RFC-9 requires zarr v3 (OME-Zarr 0.5)
pytestmark = pytest.mark.skipif(
    zarr_version_major < 3, reason="RFC-9 requires zarr-python >= 3.0.0"
)

# Output directory for test files
OUTPUT_DIR = Path(__file__).parent / "output"


def _close_zipstore_handles(multiscales):
    """
    Force close ZipStore file handles on Windows.

    ZipStore in zarr 3.1.3 doesn't properly close underlying ZIP files,
    causing PermissionError on Windows when trying to delete temp directories.
    """
    for img in multiscales.images:
        if hasattr(img.data, '_meta_array'):
            zarr_array = img.data._meta_array
            if hasattr(zarr_array, 'store'):
                store = zarr_array.store
                # Try to close the underlying ZipFile via various attribute names
                for attr in ['zip_file', '_zfile', 'zf']:
                    if hasattr(store, attr):
                        zf = getattr(store, attr)
                        if zf is not None:
                            try:
                                zf.close()
                            except Exception:
                                pass


def test_is_ozx_path():
    """Test .ozx path detection"""
    assert is_ozx_path("test.ozx")
    assert is_ozx_path("/path/to/file.ozx")
    assert is_ozx_path(Path("test.ozx"))
    assert not is_ozx_path("test.zarr")
    assert not is_ozx_path("test.ome.zarr")
    assert not is_ozx_path("test.zip")


def test_write_small_ozx_file(input_images):
    """Test writing a small dataset to .ozx file (should use memory store)"""
    dataset_name = "cthead1"
    image = input_images[dataset_name]

    multiscales = to_multiscales(
        image, [2, 4], chunks=(64, 64), method=Methods.ITKWASM_GAUSSIAN
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_write_small.ozx"

    # Write to .ozx file
    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Verify file exists and is a valid ZIP
    assert ozx_path.exists()
    import zipfile
    assert zipfile.is_zipfile(ozx_path)

    # Verify ZIP structure
    with zipfile.ZipFile(ozx_path, 'r') as zf:
        files = zf.namelist()

        # Root zarr.json should be first
        assert files[0] == "zarr.json"

        # Should contain zarr.json files for scales
        zarr_jsons = [f for f in files if f.endswith("zarr.json")]
        assert len(zarr_jsons) >= 1  # At least root zarr.json

        # Check for OME-Zarr version comment
        version = read_ozx_version(ozx_path)
        assert version == "0.5"

    # Read back and verify
    multiscales_read = from_ngff_zarr(str(ozx_path))

    # Verify metadata matches
    assert len(multiscales_read.images) == len(multiscales.images)
    for orig, read in zip(multiscales.images, multiscales_read.images):
        assert orig.dims == read.dims
        assert orig.data.shape == read.data.shape

    # Close ZipStore handles before cleanup
    _close_zipstore_handles(multiscales_read)


def test_write_ozx_requires_version_05():
    """Test that .ozx files require OME-Zarr version 0.5"""
    image = to_ngff_image(
        data=np.array([[1, 2], [3, 4]]),
        dims=("y", "x"),
    )
    multiscales = to_multiscales(image)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_version_check.ozx"

    # Should raise error for version 0.4
    with pytest.raises(ValueError, match="RFC-9.*requires.*0.5"):
        to_ngff_zarr(str(ozx_path), multiscales, version="0.4")


def test_ozx_default_chunks_per_shard(input_images):
    """Test that .ozx files default to chunks_per_shard=2"""
    import zipfile

    dataset_name = "cthead1"
    image = input_images[dataset_name]

    multiscales = to_multiscales(
        image, [2], chunks=(64, 64), method=Methods.ITKWASM_GAUSSIAN
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_default_chunks.ozx"

    # Write without specifying chunks_per_shard
    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Verify sharding is enabled
    with zipfile.ZipFile(ozx_path, 'r') as zf:
        zarr_json_content = zf.read("zarr.json")
        zarr_metadata = json.loads(zarr_json_content)

        # Check consolidated metadata for scale0
        if "consolidated_metadata" in zarr_metadata:
            scale0_metadata = zarr_metadata["consolidated_metadata"]["metadata"].get("scale0/image")
            if scale0_metadata and "codecs" in scale0_metadata:
                codecs = scale0_metadata["codecs"]
                # Should have sharding codec
                sharding_codecs = [c for c in codecs if c.get("name") == "sharding_indexed"]
                assert len(sharding_codecs) > 0, "Expected sharding codec in .ozx file"


def test_ozx_custom_chunks_per_shard(input_images):
    """Test .ozx files with custom chunks_per_shard"""
    dataset_name = "cthead1"
    image = input_images[dataset_name]

    multiscales = to_multiscales(
        image, [2], chunks=(64, 64), method=Methods.ITKWASM_GAUSSIAN
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_custom_chunks.ozx"

    # Write with custom chunks_per_shard
    to_ngff_zarr(str(ozx_path), multiscales, version="0.5", chunks_per_shard=2)

    # Verify it was created
    assert ozx_path.exists()

    # Read back and verify
    multiscales_read = from_ngff_zarr(str(ozx_path))
    assert len(multiscales_read.images) == len(multiscales.images)

    # Close ZipStore handles before cleanup
    _close_zipstore_handles(multiscales_read)


def test_large_ozx_file_uses_cache(input_images):
    """Test that large datasets use disk cache when writing .ozx"""
    import zipfile

    # Temporarily reduce memory target to force disk caching
    original_memory_target = config.memory_target
    config.memory_target = int(1e6)  # 1 MB

    try:
        dataset_name = "lung_series"
        data = imread.imread(input_images[dataset_name])
        image = to_ngff_image(
            data=data,
            dims=("z", "y", "x"),
            scale={"z": 2.5, "y": 1.40625, "x": 1.40625},
            translation={"z": 332.5, "y": 360.0, "x": 0.0},
            name="LIDC2",
        )
        multiscales = to_multiscales(image, [2])

        OUTPUT_DIR.mkdir(exist_ok=True)
        ozx_path = OUTPUT_DIR / "test_large_cache.ozx"

        # Write to .ozx file (should use disk cache)
        to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

        # Verify file was created
        assert ozx_path.exists()
        assert zipfile.is_zipfile(ozx_path)

        # Read back and verify basic properties
        multiscales_read = from_ngff_zarr(str(ozx_path))
        assert len(multiscales_read.images) == len(multiscales.images)

        # Close ZipStore handles before cleanup
        _close_zipstore_handles(multiscales_read)

    finally:
        config.memory_target = original_memory_target


def test_read_ozx_file(input_images):
    """Test reading .ozx files"""
    dataset_name = "cthead1"
    image = input_images[dataset_name]

    multiscales = to_multiscales(
        image, [2], chunks=(64, 64), method=Methods.ITKWASM_GAUSSIAN
    )
    baseline_name = "2/RFC3_GAUSSIAN.zarr"

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_read.ozx"

    # Write
    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Read back
    multiscales_read = from_ngff_zarr(str(ozx_path))

    # Verify against baseline
    verify_against_baseline(
        dataset_name, baseline_name, multiscales_read, version="0.5"
    )

    # Close ZipStore handles before cleanup
    _close_zipstore_handles(multiscales_read)


def test_ozx_zip_comment():
    """Test that .ozx files have proper ZIP comment with OME-Zarr version"""
    import zipfile

    image = to_ngff_image(
        data=np.array([[1, 2], [3, 4]]),
        dims=("y", "x"),
    )
    multiscales = to_multiscales(image)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_comment.ozx"

    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Check ZIP comment
    with zipfile.ZipFile(ozx_path, 'r') as zf:
        comment = zf.comment.decode('utf-8').rstrip('\0')
        comment_dict = json.loads(comment)

        assert "ome" in comment_dict
        assert "version" in comment_dict["ome"]
        assert comment_dict["ome"]["version"] == "0.5"
def test_ozx_zarr_json_ordering():
    """Test that zarr.json files appear first in ZIP archive"""
    import zipfile

    image = to_ngff_image(
        data=np.array([[1, 2], [3, 4]]),
        dims=("y", "x"),
    )
    multiscales = to_multiscales(image, [2])

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_ordering.ozx"

    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Check file ordering
    with zipfile.ZipFile(ozx_path, 'r') as zf:
        files = zf.namelist()

        # Find all zarr.json files
        zarr_jsons = [f for f in files if f.endswith("zarr.json")]

        # Find their indices
        zarr_json_indices = [files.index(f) for f in zarr_jsons]

        # Root zarr.json should be at index 0
        assert "zarr.json" in files
        assert files.index("zarr.json") == 0

        # All other zarr.json files should come early (before data files)
        # Data files typically have numeric names
        first_data_file_index = None
        for i, f in enumerate(files):
            if not f.endswith(".json") and f != "zarr.json":
                first_data_file_index = i
                break

        if first_data_file_index is not None:
            # All zarr.json files should come before first data file
            for idx in zarr_json_indices:
                assert idx < first_data_file_index, \
                    f"zarr.json at index {idx} should come before data files at {first_data_file_index}"


def test_ozx_no_compression():
    """Test that .ozx files use ZIP_STORED (no compression)"""
    import zipfile

    image = to_ngff_image(
        data=np.array([[1, 2], [3, 4]]),
        dims=("y", "x"),
    )
    multiscales = to_multiscales(image)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_no_compression.ozx"

    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Check compression method
    with zipfile.ZipFile(ozx_path, 'r') as zf:
        for info in zf.infolist():
            # ZIP_STORED = 0, ZIP_DEFLATED = 8
            assert info.compress_type == zipfile.ZIP_STORED, \
                f"File {info.filename} uses compression type {info.compress_type}, expected ZIP_STORED (0)"
def test_roundtrip_ozx(input_images):
    """Test full roundtrip: write to .ozx, read back, verify data"""
    dataset_name = "cthead1"
    image = input_images[dataset_name]

    multiscales = to_multiscales(
        image, [2, 4], chunks=(64, 64), method=Methods.ITKWASM_GAUSSIAN
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    ozx_path = OUTPUT_DIR / "test_roundtrip.ozx"

    # Write
    to_ngff_zarr(str(ozx_path), multiscales, version="0.5")

    # Read
    multiscales_read = from_ngff_zarr(str(ozx_path))

    # Verify all scales
    assert len(multiscales_read.images) == len(multiscales.images)

    for orig_img, read_img in zip(multiscales.images, multiscales_read.images):
        assert orig_img.dims == read_img.dims
        assert orig_img.data.shape == read_img.data.shape
        assert orig_img.scale == read_img.scale
        assert orig_img.translation == read_img.translation

        # Verify actual data matches
        orig_data = orig_img.data.compute()
        read_data = read_img.data.compute()
        assert orig_data.shape == read_data.shape
        # Data should be identical
        np.testing.assert_array_equal(orig_data, read_data)

    # Close ZipStore handles before cleanup
    _close_zipstore_handles(multiscales_read)

