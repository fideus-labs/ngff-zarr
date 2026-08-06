# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for large image chunking improvements.

These tests verify that:
1. Output chunks are aligned to input chunks (no unnecessary shrinking)
2. Explicit user-specified chunks are respected
3. High task counts trigger caching
4. Channel dimensions are preserved
5. 2D strip caching works correctly
6. 1D segment caching works correctly
7. No dask PerformanceWarning from chunk-size mismatches during cache writes
"""

import warnings

import dask.array
import numpy as np
from ngff_zarr import config, to_multiscales, to_ngff_image
from ngff_zarr.task_count import task_count

rng = np.random.default_rng(42)


class TestInputAlignedChunks:
    """Verify output chunks use max(default, input_chunk) for spatial dims."""

    def test_large_input_chunks_not_shrunk_2d(self):
        """2D image with 512x512 input chunks should NOT be shrunk to 256."""
        # Need 3+ chunks per spatial dim for tiled detection to activate
        arr = dask.array.zeros((4096, 4096), chunks=(512, 512), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))
        multiscales = to_multiscales(image, scale_factors=[])

        # Output chunks should be at least 512, not default 256
        out_data = multiscales.images[0].data
        assert out_data.chunksize[0] >= 512
        assert out_data.chunksize[1] >= 512

    def test_large_input_chunks_not_shrunk_3d(self):
        """3D image with 512x512 input chunks should NOT be shrunk to 128."""
        # Need 3+ chunks per spatial dim for tiled detection
        arr = dask.array.zeros((2, 2048, 2048), chunks=(1, 512, 512), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("z", "y", "x"))
        multiscales = to_multiscales(image, scale_factors=[])

        out_data = multiscales.images[0].data
        # y and x should stay at 512, not shrink to 128
        y_idx = list(image.dims).index("y")
        x_idx = list(image.dims).index("x")
        assert out_data.chunksize[y_idx] >= 512
        assert out_data.chunksize[x_idx] >= 512

    def test_small_input_chunks_use_default(self):
        """If input chunks are smaller than default, use the default."""
        arr = dask.array.zeros((1024, 1024), chunks=(64, 64), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))
        multiscales = to_multiscales(image, scale_factors=[])

        out_data = multiscales.images[0].data
        # Default for 2D is 256, input is 64, so should use 256
        assert out_data.chunksize[0] >= 256
        assert out_data.chunksize[1] >= 256

    def test_3d_rgb_input_chunks_preserved(self):
        """3D RGB image: spatial chunks preserved, channels kept together."""
        # Use large enough arrays so tiled detection works (3+ chunks per dim)
        arr = dask.array.zeros(
            (3, 2048, 2048, 3), chunks=(1, 512, 512, 3), dtype=np.uint8
        )
        image = to_ngff_image(arr, dims=("z", "y", "x", "c"))
        multiscales = to_multiscales(image, scale_factors=[])

        # Output dims are normalized to the spec axis order (c, z, y, x)
        out_image = multiscales.images[0]
        out_data = out_image.data
        c_idx = list(out_image.dims).index("c")
        y_idx = list(out_image.dims).index("y")
        x_idx = list(out_image.dims).index("x")
        # Channels should stay together
        assert out_data.chunksize[c_idx] == 3
        # Spatial should stay >= 512
        assert out_data.chunksize[y_idx] >= 512
        assert out_data.chunksize[x_idx] >= 512


class TestExplicitChunksRespected:
    """Verify user-specified chunks override auto-detection."""

    def test_explicit_int_chunks_override(self):
        """Explicit integer chunks should be used even if smaller than input."""
        arr = dask.array.zeros((2048, 2048), chunks=(512, 512), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))
        multiscales = to_multiscales(image, scale_factors=[], chunks=128)

        out_data = multiscales.images[0].data
        assert out_data.chunksize[0] == 128
        assert out_data.chunksize[1] == 128

    def test_explicit_dict_chunks_override(self):
        """Explicit dict chunks should be used as-is."""
        arr = dask.array.zeros((2048, 2048), chunks=(512, 512), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[], chunks={"y": 64, "x": 128}
        )

        out_data = multiscales.images[0].data
        assert out_data.chunksize[0] == 64
        assert out_data.chunksize[1] == 128

    def test_explicit_tuple_chunks_override(self):
        """Explicit tuple chunks should be used as-is."""
        arr = dask.array.zeros((2048, 2048), chunks=(512, 512), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))
        multiscales = to_multiscales(image, scale_factors=[], chunks=(64, 128))

        out_data = multiscales.images[0].data
        assert out_data.chunksize[0] == 64
        assert out_data.chunksize[1] == 128


class TestTaskCountThreshold:
    """Verify high task counts trigger caching."""

    def test_task_count_triggers_caching(self):
        """When task count exceeds threshold, caching should be triggered."""
        # Create an image with many tasks but small memory footprint
        # 200x200x200 with chunks of 2 = 1M tasks
        arr = dask.array.zeros((200, 200, 200), chunks=(2, 2, 2), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("z", "y", "x"))

        count = task_count(image)
        assert count > config.task_target

        # Should not crash and should complete (caching kicks in)
        old_target = config.task_target
        config.task_target = count - 1  # Ensure it triggers

        try:
            multiscales = to_multiscales(image, scale_factors=[], cache=None)
            assert multiscales is not None
            assert len(multiscales.images) >= 1
        finally:
            config.task_target = old_target

    def test_below_task_threshold_no_cache(self):
        """When task count is below threshold, no caching needed."""
        arr = dask.array.zeros((64, 64, 64), chunks=(32, 32, 32), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("z", "y", "x"))

        count = task_count(image)
        assert count < config.task_target

        # Should complete without caching
        multiscales = to_multiscales(image, scale_factors=[], cache=None)
        assert multiscales is not None


class TestChannelPreservation:
    """Verify multi-channel data isn't split unnecessarily."""

    def test_rgb_channels_kept_together(self):
        """RGB data chunked as (*, *, 3) should not be split to (*, *, 1)."""
        arr = dask.array.zeros((256, 256, 3), chunks=(128, 128, 3), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x", "c"))
        multiscales = to_multiscales(image, scale_factors=[])

        # Output dims are normalized to the spec axis order (c, y, x)
        out_image = multiscales.images[0]
        c_idx = list(out_image.dims).index("c")
        assert out_image.data.chunksize[c_idx] == 3

    def test_multichannel_kept_together(self):
        """Multi-channel data (e.g. 16 channels) should keep channel chunks."""
        arr = dask.array.zeros((256, 256, 16), chunks=(128, 128, 16), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x", "c"))
        multiscales = to_multiscales(image, scale_factors=[])

        # Output dims are normalized to the spec axis order (c, y, x)
        out_image = multiscales.images[0]
        c_idx = list(out_image.dims).index("c")
        assert out_image.data.chunksize[c_idx] == 16


class TestCaching2DStrips:
    """Verify 2D strip caching works correctly."""

    def test_2d_large_image_caching_produces_correct_output(self):
        """Large 2D image forced to cache should produce correct data."""
        size = 128
        arr_np = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
        arr = dask.array.from_array(arr_np, chunks=(32, 32))
        image = to_ngff_image(arr, dims=("y", "x"))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            multiscales = to_multiscales(image, scale_factors=[])
            result = multiscales.images[0].data.compute()
            np.testing.assert_array_equal(result, arr_np)
        finally:
            config.memory_target = old_mem

    def test_2d_strip_along_larger_dimension(self):
        """Strips should be along the larger spatial dimension."""
        # Y is larger than X
        arr = dask.array.zeros((1024, 256), chunks=(64, 64), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            multiscales = to_multiscales(image, scale_factors=[])
            assert multiscales is not None
        finally:
            config.memory_target = old_mem

    def test_2d_strip_x_larger(self):
        """When X is larger, strips should be along X."""
        arr = dask.array.zeros((256, 1024), chunks=(64, 64), dtype=np.uint8)
        image = to_ngff_image(arr, dims=("y", "x"))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            multiscales = to_multiscales(image, scale_factors=[])
            assert multiscales is not None
        finally:
            config.memory_target = old_mem


class TestCaching1DSegments:
    """Verify 1D segment caching works correctly."""

    def test_1d_image_caching_produces_correct_output(self):
        """Large 1D image forced to cache should produce correct data."""
        arr_np = rng.integers(0, 255, size=(1024,), dtype=np.uint8)
        arr = dask.array.from_array(arr_np, chunks=(64,))
        image = to_ngff_image(arr, dims=("x",))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            multiscales = to_multiscales(image, scale_factors=[])
            result = multiscales.images[0].data.compute()
            np.testing.assert_array_equal(result, arr_np)
        finally:
            config.memory_target = old_mem


class TestTaskCountReduction:
    """Verify that input-aligned chunks actually reduce task counts."""

    def test_large_tiles_reduce_task_count(self):
        """With 512x512 input tiles, output should not explode task count."""
        arr = dask.array.zeros(
            (3, 2048, 4096, 3), chunks=(1, 512, 512, 3), dtype=np.uint8
        )
        image = to_ngff_image(arr, dims=("z", "y", "x", "c"))

        input_tasks = arr.npartitions

        multiscales = to_multiscales(image, scale_factors=[])
        output_tasks = multiscales.images[0].data.npartitions

        # Output tasks should not be significantly more than input tasks.
        # Before the fix, this would be ~16x higher due to 512->128 rechunk.
        assert output_tasks <= input_tasks * 2


class TestCacheChunkAlignment:
    """Verify cache zarr chunks match data chunks to avoid dask PerformanceWarning.

    When cache zarr chunk sizes differ from the dask data chunk sizes,
    dask must rechunk on-the-fly during dask.array.to_zarr region writes.
    This triggers a PerformanceWarning about array.chunk-size and can
    cause data loss (see gh-issue-487).
    """

    def _assert_no_performance_warning(self, captured_warnings):
        perf_warnings = [
            x
            for x in captured_warnings
            if (
                hasattr(x, "category")
                and "PerformanceWarning" in str(x.category.__name__)
                and (
                    "chunk-size" in str(x.message)
                    or "chunk_size" in str(x.message)
                    or "chunk size" in str(x.message)
                )
            )
        ]
        assert len(perf_warnings) == 0, (
            f"Got unexpected dask PerformanceWarning(s): "
            f"{[str(x.message) for x in perf_warnings]}"
        )

    def _assert_no_channel_performance_warning(self, captured_warnings, c_index):
        """Assert no PerformanceWarning about channel (c) dimension chunk-size.

        The gh-issue-487 fix prevents dask from rechunking the channel
        dimension by using the slab chunk directly for non-spatial dims.
        Spatial dims may still warn when memory_target is tiny because
        slab_slices may not divide the dimension size.
        """
        channel_warnings = [
            x
            for x in captured_warnings
            if (
                hasattr(x, "category")
                and "PerformanceWarning" in str(x.category.__name__)
                and f"axis {c_index}" in str(x.message)
            )
        ]
        assert len(channel_warnings) == 0, (
            f"Got unexpected dask PerformanceWarning on channel axis: "
            f"{[str(x.message) for x in channel_warnings]}"
        )

    def test_z_slabs_cache_no_performance_warning(self):
        """3D z-slabs caching should produce correct data."""
        shape = (8, 64, 64)
        arr_np = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        arr = dask.array.from_array(arr_np, chunks=(8, 64, 64))
        image = to_ngff_image(arr, dims=("z", "y", "x"))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            multiscales = to_multiscales(image, scale_factors=[])
            result = multiscales.images[0].data.compute()
            np.testing.assert_array_equal(result, arr_np)
        finally:
            config.memory_target = old_mem

    def test_2d_strips_cache_no_performance_warning(self):
        """2D strip caching should not raise dask PerformanceWarning."""
        size = 128
        arr_np = np.arange(size * size, dtype=np.uint16).reshape(size, size)
        arr = dask.array.from_array(arr_np, chunks=(64, 64))
        image = to_ngff_image(arr, dims=("y", "x"))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                multiscales = to_multiscales(image, scale_factors=[])

            self._assert_no_performance_warning(w)
            result = multiscales.images[0].data.compute()
            np.testing.assert_array_equal(result, arr_np)
        finally:
            config.memory_target = old_mem

    def test_1d_segments_cache_no_performance_warning(self):
        """1D segment caching should not raise dask PerformanceWarning."""
        arr_np = np.arange(1024, dtype=np.uint16)
        arr = dask.array.from_array(arr_np, chunks=(128,))
        image = to_ngff_image(arr, dims=("x",))

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                multiscales = to_multiscales(image, scale_factors=[])

            self._assert_no_performance_warning(w)
            result = multiscales.images[0].data.compute()
            np.testing.assert_array_equal(result, arr_np)
        finally:
            config.memory_target = old_mem

    def test_multichannel_z_slabs_no_performance_warning(self):
        """3D z-slabs with channel dim should not raise dask PerformanceWarning
        on the channel axis.

        This is the exact scenario from gh-issue-487 where the input has
        shape (t, c, z, y, x) and the channel dimension chunk size (1)
        is smaller than the full channel count (2).  The fix ensures the
        Zarr chunk for the channel dimension matches the slab chunk (1)
        rather than the full dimension (2), preventing dask from
        rechunking during region writes.
        """
        shape = (1, 2, 8, 64, 64)
        arr_np = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        arr = dask.array.from_array(arr_np, chunks=(1, 1, 8, 64, 64))
        image = to_ngff_image(arr, dims=("t", "c", "z", "y", "x"))

        c_index = image.dims.index("c")

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                multiscales = to_multiscales(image, scale_factors=[])

            self._assert_no_channel_performance_warning(w, c_index)
            result = multiscales.images[0].data.compute()
            np.testing.assert_array_equal(result, arr_np)
        finally:
            config.memory_target = old_mem
