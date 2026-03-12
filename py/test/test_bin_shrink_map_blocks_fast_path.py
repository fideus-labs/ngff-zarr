# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the map_blocks fast path in bin-shrink downsampling.

When the shrink factor divides evenly into *every* chunk (including the
last, possibly smaller chunk) along each spatial dimension, bin-shrink
methods can use a simple ``dask.array.map_blocks`` call instead of the
heavier ``_align_chunks`` + per-non-spatial-dim iteration path.  This
produces a much smaller dask task graph.
"""

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import Methods, to_multiscales, to_ngff_image, to_ngff_zarr
from ngff_zarr.methods._support import _can_use_map_blocks_fast_path
from packaging import version
from zarr.storage import MemoryStore

zarr_version = version.parse(zarr.__version__)

pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)

# ---------------------------------------------------------------------------
# Unit tests for the helper
# ---------------------------------------------------------------------------


class TestCanUseMapBlocksFastPath:
    """Unit tests for ``_can_use_map_blocks_fast_path``."""

    def test_even_chunks(self):
        """All chunks divide evenly by the shrink factor -> True."""
        data = da.zeros((64, 64), chunks=(32, 32))
        image = to_ngff_image(data, dims=("y", "x"))
        assert _can_use_map_blocks_fast_path(image, {"y": 2, "x": 2})

    def test_uneven_last_chunk(self):
        """Last chunk not divisible -> False."""
        # 49 / 32 = chunks (32, 17); 17 % 2 != 0
        data = da.zeros((49, 64), chunks=(32, 32))
        image = to_ngff_image(data, dims=("y", "x"))
        assert not _can_use_map_blocks_fast_path(image, {"y": 2, "x": 2})

    def test_non_spatial_dims_ignored(self):
        """Non-spatial dimensions are not checked."""
        data = da.zeros((3, 64, 64), chunks=(1, 32, 32))
        image = to_ngff_image(data, dims=("c", "y", "x"))
        # c has factor 1 (not spatial), y and x divide evenly
        assert _can_use_map_blocks_fast_path(image, {"c": 1, "y": 2, "x": 2})

    def test_factor_one_always_ok(self):
        """Factor 1 is always fine regardless of chunk size."""
        data = da.zeros((50, 50), chunks=(17, 17))
        image = to_ngff_image(data, dims=("y", "x"))
        assert _can_use_map_blocks_fast_path(image, {"y": 1, "x": 1})

    def test_3d_even(self):
        """3D spatial data where all chunks are divisible."""
        data = da.zeros((32, 64, 64), chunks=(16, 32, 32))
        image = to_ngff_image(data, dims=("z", "y", "x"))
        assert _can_use_map_blocks_fast_path(image, {"z": 2, "y": 2, "x": 2})

    def test_3d_uneven_z(self):
        """3D data where z last chunk is not divisible."""
        # 30 / 16 = chunks (16, 14); 14 % 4 != 0
        data = da.zeros((30, 64, 64), chunks=(16, 32, 32))
        image = to_ngff_image(data, dims=("z", "y", "x"))
        assert not _can_use_map_blocks_fast_path(image, {"z": 4, "y": 2, "x": 2})

    def test_last_chunk_divisible(self):
        """Last chunk is smaller but still divisible -> True."""
        # 96 / 64 = chunks (64, 32); 32 % 2 == 0
        data = da.zeros((96, 128), chunks=(64, 64))
        image = to_ngff_image(data, dims=("y", "x"))
        assert _can_use_map_blocks_fast_path(image, {"y": 2, "x": 2})


# ---------------------------------------------------------------------------
# Integration tests: ITKWASM_BIN_SHRINK
# ---------------------------------------------------------------------------


class TestItkwasmBinShrinkFastPath:
    """End-to-end tests exercising the fast path via ITKWASM_BIN_SHRINK."""

    def test_2d_yx(self):
        """Simple 2D (y, x) with evenly divisible chunks."""
        data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        image = to_ngff_image(data, dims=("y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2, 4], chunks=64, method=Methods.ITKWASM_BIN_SHRINK
        )
        assert len(multiscales.images) == 3
        assert multiscales.images[1].data.shape == (64, 64)
        assert multiscales.images[2].data.shape == (32, 32)

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_3d_zyx(self):
        """3D (z, y, x) with evenly divisible chunks."""
        data = np.random.randint(0, 256, (32, 64, 64), dtype=np.uint8)
        image = to_ngff_image(data, dims=("z", "y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2, 4], chunks=16, method=Methods.ITKWASM_BIN_SHRINK
        )
        assert len(multiscales.images) == 3
        assert multiscales.images[1].data.shape == (16, 32, 32)
        assert multiscales.images[2].data.shape == (8, 16, 16)

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_tczyx_fast_path(self):
        """5D (t, c, z, y, x) with evenly divisible spatial chunks."""
        data = np.random.randint(0, 256, (2, 2, 32, 64, 64), dtype=np.uint8)
        image = to_ngff_image(data, dims=("t", "c", "z", "y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2, 4], chunks=32, method=Methods.ITKWASM_BIN_SHRINK
        )
        assert len(multiscales.images) == 3
        # Non-spatial dims preserved
        assert multiscales.images[1].data.shape[0] == 2  # t
        assert multiscales.images[1].data.shape[1] == 2  # c
        # Spatial dims halved
        assert multiscales.images[1].data.shape[2] == 16  # z
        assert multiscales.images[1].data.shape[3] == 32  # y
        assert multiscales.images[1].data.shape[4] == 32  # x

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_zyxc_vector_mode(self):
        """4D (z, y, x, c) with few channels -> vector mode on fast path."""
        data = np.random.randint(0, 255, (16, 32, 32, 3), dtype=np.uint8)
        image = to_ngff_image(data, dims=("z", "y", "x", "c"))
        multiscales = to_multiscales(
            image, scale_factors=[2], method=Methods.ITKWASM_BIN_SHRINK
        )
        assert len(multiscales.images) == 2
        assert multiscales.images[1].data.shape == (8, 16, 16, 3)

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_zyxc_many_channels(self):
        """4D (z, y, x, c) with >8 channels falls back to standard path."""
        # With 12 channels (> _MAX_VECTOR_COMPONENTS=8), the fast path
        # is skipped; verify the standard path still works correctly.
        data = np.random.randint(0, 65535, (8, 32, 32, 12), dtype=np.uint16)
        image = to_ngff_image(data, dims=("z", "y", "x", "c"))
        multiscales = to_multiscales(
            image, scale_factors=[2], method=Methods.ITKWASM_BIN_SHRINK
        )
        assert len(multiscales.images) == 2
        # Verify channels are preserved
        assert multiscales.images[1].data.shape[-1] == 12
        # Verify spatial dimensions are downsampled
        assert multiscales.images[1].data.shape[0] == 4  # z
        assert multiscales.images[1].data.shape[1] == 16  # y
        assert multiscales.images[1].data.shape[2] == 16  # x

    def test_dict_scale_factors(self):
        """Dict-based scale factors on the fast path."""
        data = da.ones(
            (96, 128, 128, 128, 2), chunks=(1, 64, 64, 64, 1), dtype="uint16"
        )
        image = to_ngff_image(data, dims=("t", "z", "y", "x", "c"))
        multiscales = to_multiscales(
            image,
            scale_factors=[{"z": 2, "y": 2, "x": 2}, {"z": 4, "y": 4, "x": 4}],
            method=Methods.ITKWASM_BIN_SHRINK,
        )
        assert len(multiscales.images) == 3

    def test_fast_path_matches_standard_path(self):
        """Verify fast path produces identical output to the standard path.

        Construct data whose chunk sizes divide evenly (fast path eligible),
        compute the result, then compare it against a manually rechunked
        version that forces the standard path (by making the last chunk
        indivisible) — the pixel values should be the same apart from
        edge-chunk rounding.
        """
        np.random.seed(42)
        data_np = np.random.randint(0, 256, (64, 128), dtype=np.uint8)

        # Fast path: 64 and 128 divide by 2
        image_fast = to_ngff_image(
            da.from_array(data_np, chunks=(32, 64)), dims=("y", "x")
        )
        ms_fast = to_multiscales(
            image_fast, scale_factors=[2], chunks=32, method=Methods.ITKWASM_BIN_SHRINK
        )

        # Standard path: append a row so last chunk is 33 (not divisible by 2)
        padded = np.pad(data_np, ((0, 1), (0, 0)), constant_values=0)
        image_std = to_ngff_image(
            da.from_array(padded, chunks=(32, 64)), dims=("y", "x")
        )
        ms_std = to_multiscales(
            image_std, scale_factors=[2], chunks=32, method=Methods.ITKWASM_BIN_SHRINK
        )

        # The first 32 rows of the downsampled result should match exactly
        fast_arr = ms_fast.images[1].data.compute()
        std_arr = ms_std.images[1].data.compute()
        np.testing.assert_array_equal(fast_arr, std_arr[: fast_arr.shape[0], :])

    def test_custom_out_chunks(self):
        """Fast path respects a user-specified out_chunks that differs from input."""
        data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        image = to_ngff_image(da.from_array(data, chunks=(64, 64)), dims=("y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2], chunks=16, method=Methods.ITKWASM_BIN_SHRINK
        )
        # Output should be rechunked to 16
        downsampled = multiscales.images[1].data
        # All interior chunks should be 16 (last may differ)
        assert downsampled.chunksize[0] == 16
        assert downsampled.chunksize[1] == 16

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_smaller_dask_graph(self):
        """The fast path should produce fewer dask tasks than the standard path."""
        np.random.seed(0)
        data_np = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        # Fast path: chunks divide evenly
        image_fast = to_ngff_image(
            da.from_array(data_np, chunks=(64, 64)), dims=("y", "x")
        )
        ms_fast = to_multiscales(
            image_fast, scale_factors=[2], chunks=64, method=Methods.ITKWASM_BIN_SHRINK
        )

        # Standard path: last chunk is 65 (not divisible by 2), force align
        padded = np.pad(data_np, ((0, 1), (0, 1)), constant_values=0)
        image_std = to_ngff_image(
            da.from_array(padded, chunks=(64, 64)), dims=("y", "x")
        )
        ms_std = to_multiscales(
            image_std, scale_factors=[2], chunks=64, method=Methods.ITKWASM_BIN_SHRINK
        )

        fast_tasks = len(dict(ms_fast.images[1].data.__dask_graph__()))
        std_tasks = len(dict(ms_std.images[1].data.__dask_graph__()))
        assert fast_tasks <= std_tasks, (
            f"Fast path ({fast_tasks} tasks) should not have more tasks "
            f"than standard path ({std_tasks} tasks)"
        )


# ---------------------------------------------------------------------------
# Integration tests: ITK_BIN_SHRINK
# ---------------------------------------------------------------------------


class TestItkBinShrinkFastPath:
    """End-to-end tests exercising the fast path via ITK_BIN_SHRINK."""

    def test_2d_yx(self):
        """Simple 2D (y, x) with evenly divisible chunks."""
        data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        image = to_ngff_image(data, dims=("y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2, 4], chunks=64, method=Methods.ITK_BIN_SHRINK
        )
        assert len(multiscales.images) == 3
        assert multiscales.images[1].data.shape == (64, 64)
        assert multiscales.images[2].data.shape == (32, 32)

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_3d_zyx(self):
        """3D (z, y, x) with evenly divisible chunks."""
        data = np.random.randint(0, 256, (32, 64, 64), dtype=np.uint8)
        image = to_ngff_image(data, dims=("z", "y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2, 4], chunks=16, method=Methods.ITK_BIN_SHRINK
        )
        assert len(multiscales.images) == 3
        assert multiscales.images[1].data.shape == (16, 32, 32)
        assert multiscales.images[2].data.shape == (8, 16, 16)

        store = MemoryStore()
        to_ngff_zarr(store, multiscales)

    def test_smaller_dask_graph(self):
        """The fast path should produce fewer dask tasks than the standard path."""
        np.random.seed(0)
        data_np = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        # Fast path: chunks divide evenly
        image_fast = to_ngff_image(
            da.from_array(data_np, chunks=(64, 64)), dims=("y", "x")
        )
        ms_fast = to_multiscales(
            image_fast, scale_factors=[2], chunks=64, method=Methods.ITK_BIN_SHRINK
        )

        # Standard path: last chunk is 65 (not divisible by 2), force align
        padded = np.pad(data_np, ((0, 1), (0, 1)), constant_values=0)
        image_std = to_ngff_image(
            da.from_array(padded, chunks=(64, 64)), dims=("y", "x")
        )
        ms_std = to_multiscales(
            image_std, scale_factors=[2], chunks=64, method=Methods.ITK_BIN_SHRINK
        )

        fast_tasks = len(dict(ms_fast.images[1].data.__dask_graph__()))
        std_tasks = len(dict(ms_std.images[1].data.__dask_graph__()))
        assert fast_tasks <= std_tasks, (
            f"Fast path ({fast_tasks} tasks) should not have more tasks "
            f"than standard path ({std_tasks} tasks)"
        )
