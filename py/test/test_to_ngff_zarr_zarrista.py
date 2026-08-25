# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the zarrista fast direct-write path of to_ngff_zarr.

Until the zarrista backend becomes the default, the path is opted into via
the deprecated ``use_tensorstore`` keyword. The deprecation warning itself is
asserted in ``test_use_tensorstore_deprecation_warning`` and silenced for the
rest of the module.
"""

import logging
import pathlib
import tempfile
import time
import uuid

import numpy as np
import pytest
import zarr
from dask_image import imread
from ngff_zarr import (
    Methods,
    config,
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)
from packaging import version

from ._data import verify_against_baseline

# zarrista requires Python >= 3.11; the py310 environment runs without it.
pytest.importorskip("zarrista")

zarr_version = version.parse(zarr.__version__)

pytestmark = pytest.mark.filterwarnings(
    "ignore:use_tensorstore is deprecated:DeprecationWarning"
)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)
def test_use_tensorstore_deprecation_warning():
    """use_tensorstore=True warns but still produces a valid store."""
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    image = to_ngff_image(data, dims=("y", "x"), scale={"y": 1.0, "x": 1.0})
    multiscales = to_multiscales(image, [2], method=Methods.DASK_IMAGE_GAUSSIAN)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.warns(DeprecationWarning, match="zarrista backend"):
            to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)
        back = from_ngff_zarr(tmpdir)
        np.testing.assert_array_equal(np.asarray(back.images[0].data), data)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)
def test_gaussian_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_GAUSSIAN)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)
        multiscales = from_ngff_zarr(tmpdir)
        verify_against_baseline(dataset_name, baseline_name, multiscales)

    baseline_name = "auto/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, method=Methods.ITKWASM_GAUSSIAN)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)
        multiscales = from_ngff_zarr(tmpdir)
        verify_against_baseline(dataset_name, baseline_name, multiscales)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 3], method=Methods.ITKWASM_GAUSSIAN)
    with tempfile.TemporaryDirectory() as tmpdir:
        # The baselines hold the exact geometry. The default "pad" strategy
        # trades it for speed and is covered in
        # test_non_power_of_2_scale_factors.py.
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True, scale_strategy="exact")
        multiscales = from_ngff_zarr(tmpdir)
        verify_against_baseline(
            dataset_name, baseline_name, multiscales, scale_strategy="exact"
        )

    dataset_name = "MR-head"
    image = input_images[dataset_name]
    baseline_name = "2_3_4/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 3, 4], method=Methods.ITKWASM_GAUSSIAN)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True, scale_strategy="exact")
        multiscales = from_ngff_zarr(tmpdir)
        verify_against_baseline(
            dataset_name, baseline_name, multiscales, scale_strategy="exact"
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)
def test_large_image_serialization(input_images):
    default_mem_target = config.memory_target
    config.memory_target = int(1e6)

    dataset_name = "lung_series"
    data = imread.imread(input_images[dataset_name])
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 2.5, "y": 1.40625, "x": 1.40625},
        translation={"z": 332.5, "y": 360.0, "x": 0.0},
        name="LIDC2",
    )
    multiscales = to_multiscales(image)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)
        config.memory_target = default_mem_target


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)
def test_zarrista_regional_write_large_image():
    """Regional writing of a large image succeeds and does not fail on
    pre-existing arrays.

    Regression coverage for the historical ALREADY_EXISTS failure the
    tensorstore backend hit when large data sizes triggered regional writing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("ZarristaRegionalWriteTest")

    large_shape = (512, 512, 512)  # Large shape to trigger regional writing
    input_dtype = np.float32
    pixel_size = 1.3

    input_array = np.random.rand(*large_shape).astype(input_dtype)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_output_dir = pathlib.Path(tmpdir)
        scan_id = str(uuid.uuid4())[:8]
        zarr_path = test_output_dir / f"{scan_id}.zarr"
        logger.info(f"Test output directory: {test_output_dir}")

        # 1. Create NGFF Image
        logger.info("  Creating NGFF image object...")
        image = to_ngff_image(
            input_array,
            dims=("z", "y", "x"),
            scale={"z": pixel_size, "y": pixel_size, "x": pixel_size},
            axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        )

        # 2. Create NgffMultiscales (Using a method known to work)
        logger.info("  Creating multiscales...")
        start_time_multiscale = time.time()
        multiscales = to_multiscales(
            image,
            method=Methods.DASK_IMAGE_GAUSSIAN,
            chunks=None,  # Default chunking
            cache=False,
        )
        end_time_multiscale = time.time()
        logger.info(
            f"  NgffMultiscales created in {end_time_multiscale - start_time_multiscale:.2f} seconds."
        )

        # 3. Write Zarr with the zarrista fast direct-write path
        logger.info("  Writing NGFF Zarr with zarrista...")
        start_time_write = time.time()
        to_ngff_zarr(
            store=zarr_path,
            multiscales=multiscales,
            use_tensorstore=True,  # Fast direct-write path
        )
        end_time_write = time.time()
        logger.info(
            f"  Zarr written in {end_time_write - start_time_write:.2f} seconds."
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_chunk_shape_consistency():
    """Chunk shapes stay consistent for edge case dimensions."""
    # This reproduces the issue from #161 where array dimensions don't divide evenly by chunk size
    # Shape needs to be large enough to trigger regional writing and have uneven chunk divisions
    shape = (
        515,
        512,
        512,
    )  # Large enough to trigger issue, with uneven division in first dimension
    test_array = np.random.rand(*shape).astype(np.float32)

    image = to_ngff_image(
        test_array,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )

    multiscales = to_multiscales(
        image,
        method=Methods.ITKWASM_GAUSSIAN,
        cache=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # This should not fail with chunk shape mismatch errors
        to_ngff_zarr(
            store=tmpdir,
            multiscales=multiscales,
            use_tensorstore=True,
            version="0.5",
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_chunk_shape_consistency_with_sharding():
    """Sharding combined with edge case dimensions writes correctly."""
    shape = (
        515,
        512,
        512,
    )  # Large enough to trigger issue, with uneven division in first dimension
    test_array = np.random.rand(*shape).astype(np.float32)

    image = to_ngff_image(
        test_array,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )

    multiscales = to_multiscales(
        image,
        method=Methods.ITKWASM_GAUSSIAN,
        cache=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # This should not fail with chunk shape mismatch errors
        to_ngff_zarr(
            store=tmpdir,
            multiscales=multiscales,
            use_tensorstore=True,
            chunks_per_shard=2,
            version="0.5",
        )


def _write_and_read_codecs(tmpdir, **to_ngff_zarr_kwargs):
    import json

    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    image = to_ngff_image(data, dims=("y", "x"), scale={"y": 1.0, "x": 1.0})
    multiscales = to_multiscales(image, [2], method=Methods.DASK_IMAGE_GAUSSIAN)
    to_ngff_zarr(
        tmpdir,
        multiscales,
        use_tensorstore=True,
        version="0.5",
        **to_ngff_zarr_kwargs,
    )
    array_metadata = json.loads(
        (pathlib.Path(tmpdir) / "scale0" / "image" / "zarr.json").read_text()
    )
    return array_metadata["codecs"]


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_codec_chain_preserved():
    """An explicit codec chain is written in order, not reduced to one codec."""
    from zarr.codecs import BytesCodec, GzipCodec, ZstdCodec

    with tempfile.TemporaryDirectory() as tmpdir:
        codecs = _write_and_read_codecs(
            tmpdir,
            compressors=[BytesCodec(), ZstdCodec(level=3), GzipCodec(level=2)],
        )
    assert [codec["name"] for codec in codecs] == ["bytes", "zstd", "gzip"]
    assert codecs[1]["configuration"]["level"] == 3
    assert codecs[2]["configuration"]["level"] == 2


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_codec_chain_bytes_only():
    """A chain holding only the bytes codec writes uncompressed data."""
    from zarr.codecs import BytesCodec

    with tempfile.TemporaryDirectory() as tmpdir:
        codecs = _write_and_read_codecs(tmpdir, compressors=[BytesCodec()])
    assert [codec["name"] for codec in codecs] == ["bytes"]


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_codec_chain_empty():
    """An explicit empty chain writes uncompressed data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        codecs = _write_and_read_codecs(tmpdir, compressors=())
    assert [codec["name"] for codec in codecs] == ["bytes"]


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_codec_chain_default():
    """Without a codec option the default zstd compression is applied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        codecs = _write_and_read_codecs(tmpdir)
    assert [codec["name"] for codec in codecs] == ["bytes", "zstd"]


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_zarrista_zero_chunk_validation():
    """Zero chunk sizes are properly validated."""
    shape = (513, 512, 512)  # Large enough to trigger regional writing with edge chunks
    test_array = np.random.rand(*shape).astype(np.float32)

    image = to_ngff_image(
        test_array,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )

    multiscales = to_multiscales(
        image,
        method=Methods.ITKWASM_GAUSSIAN,
        cache=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # This should not fail with zero chunk size errors
        to_ngff_zarr(
            store=tmpdir,
            multiscales=multiscales,
            use_tensorstore=True,
            version="0.5",
        )
