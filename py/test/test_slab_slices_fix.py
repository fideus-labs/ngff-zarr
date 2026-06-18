# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Regression tests for PR #447 fix and zarr_kwargs mutation fix.

PR #447: _compute_write_regions uses slab_slices for Z boundaries.
GH #490: _handle_large_array_writing mutated caller's zarr_kwargs dict.
"""

import tempfile

import numpy as np
import packaging.version
import pytest
import zarr
from ngff_zarr import (
    config,
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)

zarr_version_major = packaging.version.parse(zarr.__version__).major

ome_zarr_versions = ["0.4"] + (["0.5"] if zarr_version_major >= 3 else [])


@pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
def test_slab_slices_regional_writing(ome_zarr_version):
    """Validate that _compute_write_regions uses slab_slices (not z_chunks) for Z boundaries.

    PR #447 fixed a bug where the region boundary for each Z-slab was computed
    using z_chunks instead of slab_slices.  When slab_slices > z_chunks the bug
    causes the wrong Z range to be written for every slab past the first,
    resulting in missing or corrupted data.

    The test forces the slab (slice_planes=False) code path by choosing a
    memory_target that satisfies:

        memory_usage(image, {"z"}) * z_chunk <= memory_target
                                               < memory_usage(image)

    which requires z_chunk**2 < z_shape.

    With shape=(32, 64, 64), chunks=2, uint32:
      - memory_usage_z  = 2 * 64 * 64 * 4**3 = 524 288 bytes
      - memory_usage    = 32 * 64 * 64 * 4**3 = 8 388 608 bytes
      - memory_target   = 8 000 000
      - slab_slices     = ceil(8_000_000 / 524_288) = 16  (>= z_chunk=2)
      - slice_planes    = False                            (PR #447 code path)
      - num_slabs       = ceil(32 / 16) = 2
    """
    pytest.importorskip("tensorstore")

    default_mem_target = config.memory_target

    try:
        shape = (32, 64, 64)
        z_chunk = 2

        # uint32 ensures all 32*64*64 = 131 072 index values are distinct
        data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)

        image = to_ngff_image(
            data=data,
            dims=("z", "y", "x"),
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
        )

        # scale_factors=[] keeps only the base scale to focus the test on
        # _compute_write_regions without running any downsampling.
        # cache=False avoids disk serialization during multiscale creation.
        multiscales = to_multiscales(
            image, chunks=z_chunk, scale_factors=[], cache=False
        )

        # memory_usage(image, {"z"}) for this array = 524 288 bytes.
        # Setting memory_target = 8 000 000 gives slab_slices = 16,
        # which is greater than z_chunk=2, so slice_planes=False and
        # two slabs are produced: z[0:16] and z[16:32].
        config.memory_target = 8_000_000

        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(
                tmpdir, multiscales, version=ome_zarr_version, use_tensorstore=True
            )

            read_multiscales = from_ngff_zarr(tmpdir)
            read_data = np.asarray(read_multiscales.images[0].data)

            assert read_data.shape == shape, (
                f"Shape mismatch: expected {shape}, got {read_data.shape}"
            )

            # With the pre-fix bug (z_chunk=2 used instead of slab_slices=16)
            # slab 0 would write z[0:2] and slab 1 z[2:4], leaving z[4:32] as
            # zeros.  The fix produces z[0:16] and z[16:32].
            np.testing.assert_array_equal(read_data, data)

    finally:
        config.memory_target = default_mem_target


@pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
def test_slab_slices_with_non_divisible_shape(ome_zarr_version):
    """Verify partial final slab is written correctly when Z is not a multiple of slab_slices.

    With shape=(20, 64, 64), chunks=2, uint32:
      - memory_usage_z  = 524 288 bytes  (same z_chunk / y / x as above)
      - memory_usage    = 20 * 64 * 64 * 4**3 = 5 242 880 bytes
      - memory_target   = 4 000 000
      - slab_slices     = ceil(4_000_000 / 524_288) = 8  (>= z_chunk=2)
      - slice_planes    = False                           (PR #447 code path)
      - num_slabs       = ceil(20 / 8) = 3
      - slabs           = z[0:8], z[8:16], z[16:20]  (last is partial)
    """
    pytest.importorskip("tensorstore")

    default_mem_target = config.memory_target

    try:
        shape = (20, 64, 64)
        z_chunk = 2

        data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)

        image = to_ngff_image(
            data=data,
            dims=("z", "y", "x"),
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
        )

        multiscales = to_multiscales(
            image, chunks=z_chunk, scale_factors=[], cache=False
        )

        config.memory_target = 4_000_000

        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(
                tmpdir, multiscales, version=ome_zarr_version, use_tensorstore=True
            )

            read_multiscales = from_ngff_zarr(tmpdir)
            read_data = np.asarray(read_multiscales.images[0].data)

            assert read_data.shape == shape, (
                f"Shape mismatch: expected {shape}, got {read_data.shape}"
            )

            # Full round-trip check, including the partial final slab z[16:20].
            np.testing.assert_array_equal(read_data, data)

    finally:
        config.memory_target = default_mem_target


@pytest.mark.skipif(
    zarr_version_major < 3,
    reason="Bug only manifests with zarr v3 (IS_ZARR_V3_PLUS) writing zarr v2 format",
)
def test_zarr_kwargs_not_mutated_across_scale_iterations():
    """Regression test for zarr_kwargs mutation in _handle_large_array_writing.

    _handle_large_array_writing mutates zarr_kwargs in-place when
    zarr_format == 2 (OME-Zarr v0.4) and IS_ZARR_V3_PLUS (deleting
    chunk_key_encoding, adding dimension_separator).  Without a defensive
    copy, subsequent loop iterations in _to_ngff_zarr_impl fail with
    KeyError: 'chunk_key_encoding'.
    """
    default_mem_target = config.memory_target

    try:
        shape = (64, 128, 128)
        data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

        image = to_ngff_image(
            data=data,
            dims=("z", "y", "x"),
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
        )

        multiscales = to_multiscales(image, scale_factors=[2], cache=False)

        assert len(multiscales.images) == 2, (
            f"Expected 2 scales, got {len(multiscales.images)}"
        )

        # Force large-array path for both scales:
        #   scale 0: (64, 128, 128) = 1 048 576 bytes
        #   scale 1: (32,  64,  64) =   131 072 bytes
        config.memory_target = 100_000

        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(tmpdir, multiscales, version="0.4")

            read_multiscales = from_ngff_zarr(tmpdir)
            assert len(read_multiscales.images) == 2

            read_data = np.asarray(read_multiscales.images[0].data)
            np.testing.assert_array_equal(read_data, data)

    finally:
        config.memory_target = default_mem_target
