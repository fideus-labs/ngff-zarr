# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import tempfile
from packaging import version
import logging
import pathlib
import time
import uuid

import pytest
import zarr
from dask_image import imread
import numpy as np

from ngff_zarr import (
    Methods,
    to_multiscales,
    to_ngff_zarr,
    from_ngff_zarr,
    config,
    to_ngff_image,
)

from ._data import verify_against_baseline

zarr_version = version.parse(zarr.__version__)


def test_gaussian_isotropic_scale_factors(input_images):
    pytest.importorskip("tensorstore")

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
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)
        multiscales = from_ngff_zarr(tmpdir)
        verify_against_baseline(dataset_name, baseline_name, multiscales)

    dataset_name = "MR-head"
    image = input_images[dataset_name]
    baseline_name = "2_3_4/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 3, 4], method=Methods.ITKWASM_GAUSSIAN)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)
        multiscales = from_ngff_zarr(tmpdir)
        verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_large_image_serialization(input_images):
    pytest.importorskip("tensorstore")

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


def test_tensorstore_already_exists_failure():
    """
    Demonstrates the ALREADY_EXISTS failure with use_tensorstore=True during Zarr writing.
    This failure occurs with large data sizes that trigger regional writing.
    """
    pytest.importorskip("tensorstore")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("TensorstoreAlreadyExistsTest")

    large_shape = (512, 512, 512)  # Large shape for TensorStore test
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

        # 2. Create Multiscales (Using a method known to work)
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
            f"  Multiscales created in {end_time_multiscale - start_time_multiscale:.2f} seconds."
        )

        # 3. Write Zarr (This was failing)
        logger.info("  Writing NGFF Zarr with TensorStore...")
        start_time_write = time.time()
        to_ngff_zarr(
            store=zarr_path,
            multiscales=multiscales,
            use_tensorstore=True,  # Enable TensorStore
        )
        end_time_write = time.time()
        logger.info(
            f"  Zarr written in {end_time_write - start_time_write:.2f} seconds (UNEXPECTED SUCCESS)."
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_tensorstore_chunk_shape_consistency():
    """Test that TensorStore handles chunk shape consistency for edge case dimensions."""
    # This reproduces the issue from #161 where array dimensions don't divide evenly by chunk size
    # Shape needs to be large enough to trigger regional writing and have uneven chunk divisions
    pytest.importorskip("tensorstore")

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
def test_tensorstore_chunk_shape_consistency_with_sharding():
    """Test TensorStore with sharding and edge case dimensions."""
    pytest.importorskip("tensorstore")
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


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"), reason="zarr version < 3.0.0b1"
)
def test_tensorstore_zero_chunk_validation():
    """Test that zero chunk sizes are properly validated."""
    pytest.importorskip("tensorstore")

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
