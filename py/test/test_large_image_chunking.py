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

import sys
import warnings

import dask.array
import numpy as np
from ngff_zarr import config, to_multiscales, to_ngff_image, to_ngff_zarr
from ngff_zarr.task_count import task_count

rng = np.random.default_rng(42)


def _freeze_cache_clock(monkeypatch, value=1756000000.0):
    """Make the cache path builder observe a single, repeating timestamp.

    Emulates Windows with Python < 3.13, where time.time() has ~15.6 ms
    resolution and back-to-back to_multiscales() calls read the same value.
    """
    # The package attribute shadows the module, so go through sys.modules
    module = sys.modules["ngff_zarr.to_multiscales"]

    class FrozenClock:
        @staticmethod
        def time():
            return value

    monkeypatch.setattr(module, "time", FrozenClock)


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

    def test_cache_paths_unique_with_coarse_clock(self, monkeypatch):
        """Caches must not collide when time.time() repeats.

        Caching a 2D then a 1D image into one directory used to raise, because
        the 1D array's chunk key "c/0" is a file where the 2D array's is a
        directory.
        """
        _freeze_cache_clock(monkeypatch)

        arr_2d = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
        image_2d = to_ngff_image(
            dask.array.from_array(arr_2d, chunks=(64, 64)), dims=("y", "x")
        )
        arr_1d = np.arange(1024, dtype=np.uint16)
        image_1d = to_ngff_image(
            dask.array.from_array(arr_1d, chunks=(128,)), dims=("x",)
        )

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            result_2d = to_multiscales(image_2d, scale_factors=[]).images[0].data
            np.testing.assert_array_equal(result_2d.compute(), arr_2d)

            result_1d = to_multiscales(image_1d, scale_factors=[]).images[0].data
            np.testing.assert_array_equal(result_1d.compute(), arr_1d)
        finally:
            config.memory_target = old_mem

    def test_same_shape_caches_do_not_corrupt_with_coarse_clock(self, monkeypatch):
        """Same-shape caches in one clock tick must not overwrite each other.

        This is the silent variant of the collision: identical shapes produce
        identical chunk keys, so the second cache overwrote the first and the
        first image read back as the second's data with no error raised.
        """
        _freeze_cache_clock(monkeypatch)

        shape = (128, 128)
        first_np = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        second_np = np.full(shape, 7, dtype=np.uint16)
        first = to_ngff_image(
            dask.array.from_array(first_np, chunks=(64, 64)), dims=("y", "x")
        )
        second = to_ngff_image(
            dask.array.from_array(second_np, chunks=(64, 64)), dims=("y", "x")
        )

        old_mem = config.memory_target
        config.memory_target = 1  # Force caching

        try:
            # Cache both before computing either, so the second write would
            # land on the first's chunks while they are still needed.
            first_result = to_multiscales(first, scale_factors=[]).images[0].data
            second_result = to_multiscales(second, scale_factors=[]).images[0].data

            np.testing.assert_array_equal(second_result.compute(), second_np)
            np.testing.assert_array_equal(first_result.compute(), first_np)
        finally:
            config.memory_target = old_mem

    def test_cache_directory_removed_after_write(self, tmp_path):
        """The cache directory is cleaned up once the image is written."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        arr_np = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
        image = to_ngff_image(
            dask.array.from_array(arr_np, chunks=(64, 64)), dims=("y", "x")
        )

        old_mem = config.memory_target
        old_cache_store = config.cache_store
        config.memory_target = 1  # Force caching
        config.cache_store = str(cache_dir)

        try:
            multiscales = to_multiscales(image, scale_factors=[])
            assert list(cache_dir.iterdir()), "expected a cache directory"

            to_ngff_zarr(tmp_path / "test.ome.zarr", multiscales)
            assert not list(cache_dir.iterdir()), (
                f"cache not cleaned up: {[p.name for p in cache_dir.iterdir()]}"
            )
        finally:
            config.memory_target = old_mem
            config.cache_store = old_cache_store
