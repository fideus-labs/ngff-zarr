# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test for PR #447 fix: correctly calculate regions in to_ngff_zarr using slab_slices."""

import tempfile

import numpy as np
import pytest
from ngff_zarr import config, from_ngff_zarr, to_multiscales, to_ngff_image, to_ngff_zarr


def test_slab_slices_regional_writing():
    """Test that regional writing uses slab_slices instead of z_chunks for Z dimension.

    This test validates the fix in PR #447 where _compute_write_regions was using
    z_chunks instead of slab_slices to calculate write region boundaries.

    The bug would cause incomplete writing of Z planes when slab_slices > z_chunks,
    resulting in only the first z_chunks planes being written for each slab instead
    of the full slab_slices planes.

    To trigger the bug, we need:
    1. A 3D array with Z dimension
    2. memory_target set low enough that slab_slices > z_chunks
    3. Multiple Z planes to write
    """
    pytest.importorskip("tensorstore")

    # Save original memory target
    default_mem_target = config.memory_target

    try:
        # Create a 3D array with dimensions that will trigger regional writing
        # Shape: (z=64, y=128, x=128) with z_chunks=32
        # We'll set memory_target so that slab_slices becomes 64 (2 * z_chunks)
        shape = (64, 128, 128)
        z_chunks = 32
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)

        # Create NgffImage
        image = to_ngff_image(
            data=data,
            dims=("z", "y", "x"),
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
        )

        # Calculate memory_target to force slab_slices = 64 (2 * z_chunks)
        # Each Z plane is 128 * 128 * 2 bytes = 32768 bytes
        # For slab_slices = 64, we need memory_target >= 64 * 32768 = 2097152 bytes
        # Set it just above that threshold to get slab_slices = 64
        config.memory_target = int(2.5e6)  # 2.5 MB

        # Create multiscales with specific chunks
        multiscales = to_multiscales(image, chunks=z_chunks)

        # Write to zarr using tensorstore (which uses regional writing)
        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)

            # Read back and verify all data was written correctly
            read_multiscales = from_ngff_zarr(tmpdir)
            read_data = np.asarray(read_multiscales.images[0].data)

            # Verify shape is correct
            assert (
                read_data.shape == shape
            ), f"Shape mismatch: expected {shape}, got {read_data.shape}"

            # Verify all Z planes were written (not just first z_chunks planes)
            # If the bug existed, only planes 0-31 would be written for first slab
            # and planes 32-63 would be all zeros or missing
            np.testing.assert_array_equal(
                read_data, data, err_msg="Data mismatch: not all Z planes were written"
            )

            # Specifically check the boundary between slabs (around z=32)
            # This is where the bug would manifest
            assert np.array_equal(
                read_data[31], data[31]
            ), "Z plane 31 (end of first slab) was not written correctly"
            assert np.array_equal(
                read_data[32], data[32]
            ), "Z plane 32 (start of second slab) was not written correctly"

    finally:
        # Restore original memory target
        config.memory_target = default_mem_target


def test_slab_slices_with_non_divisible_shape():
    """Test slab_slices fix with Z dimension not evenly divisible by z_chunks.

    This tests an edge case where the Z dimension doesn't divide evenly by z_chunks,
    ensuring the last slab is handled correctly.
    """
    pytest.importorskip("tensorstore")

    default_mem_target = config.memory_target

    try:
        # Create array where Z dimension (50) is not evenly divisible by z_chunks (32)
        shape = (50, 128, 128)
        z_chunks = 32
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)

        image = to_ngff_image(
            data=data,
            dims=("z", "y", "x"),
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
        )

        # Set memory target to force regional writing
        config.memory_target = int(2.5e6)

        multiscales = to_multiscales(image, chunks=z_chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True)

            read_multiscales = from_ngff_zarr(tmpdir)
            read_data = np.asarray(read_multiscales.images[0].data)

            # Verify all data including the last partial slab
            assert read_data.shape == shape
            np.testing.assert_array_equal(read_data, data)

            # Check the last few Z planes specifically
            assert np.array_equal(read_data[48], data[48])
            assert np.array_equal(read_data[49], data[49])

    finally:
        config.memory_target = default_mem_target
