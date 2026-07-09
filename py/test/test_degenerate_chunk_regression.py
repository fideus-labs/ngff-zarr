# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import glob
import json

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)

# OME-Zarr v0.5 (sharding) requires zarr-python >= 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def _read_scale0_meta(store):
    path = glob.glob(f"{store}/scale0/*/zarr.json")[0]
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize(
    # dims chosen so the near-target divisor is tiny under the old algorithm
    "y, x",  # x = 5 * prime, y = prime
    [(29, 185), (43, 265)],
)
def test_large_path_does_not_collapse_prime_axis(y, x, tmp_path):
    """The memory_target/sharded write path must not snap a prime-factored axis
    to a tiny divisor (regression for the 5px / 1px chunk collapse)."""
    arr = da.zeros((1, 3, y, x), dtype=np.uint8, chunks=(1, 1, y, x))
    image = nz.to_ngff_image(arr, dims=("t", "c", "y", "x"))
    ms = nz.to_multiscales(image, scale_factors=[2], chunks=8)

    old_target = nz.config.memory_target
    nz.config.memory_target = 1  # force the large-array path
    try:
        store = tmp_path / "test.ome.zarr"
        nz.to_ngff_zarr(store, ms, version="0.5", chunks_per_shard=2)
    finally:
        nz.config.memory_target = old_target

    meta = _read_scale0_meta(store)
    shard = meta["chunk_grid"]["configuration"]["chunk_shape"]
    inner = meta["codecs"][0]["configuration"]["chunk_shape"]
    # requested inner chunk was 8; must not collapse well below it on any axis
    assert min(inner[-2:]) >= 8, f"degenerate inner chunk {inner}"
    assert min(shard[-2:]) >= 8, f"degenerate shard {shard}"


def test_large_path_partial_final_shard_reads_back(tmp_path):
    """A partial final shard on a prime axis reads back identical to the input."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(1, 3, 29, 185), dtype=np.uint8)
    arr = da.from_array(data, chunks=(1, 1, 29, 185))
    image = nz.to_ngff_image(arr, dims=("t", "c", "y", "x"))
    ms = nz.to_multiscales(image, scale_factors=[1], chunks=8)

    old_target = nz.config.memory_target
    nz.config.memory_target = 1
    try:
        store = tmp_path / "test.ome.zarr"
        nz.to_ngff_zarr(store, ms, version="0.5", chunks_per_shard=2)
    finally:
        nz.config.memory_target = old_target

    back = nz.from_ngff_zarr(store)
    result = np.asarray(back.images[0].data)
    assert np.array_equal(result, data)


def test_large_path_preserves_user_compressor(tmp_path):
    """A user-supplied Zstd level is preserved on the large-array / sharded write path."""
    from zarr.codecs.zstd import ZstdCodec

    arr = da.zeros((1, 3, 29, 185), dtype=np.uint8, chunks=(1, 1, 29, 185))
    image = nz.to_ngff_image(arr, dims=("t", "c", "y", "x"))
    ms = nz.to_multiscales(image, scale_factors=[1], chunks=8)

    old_target = nz.config.memory_target
    nz.config.memory_target = 1
    try:
        store = tmp_path / "test.ome.zarr"
        nz.to_ngff_zarr(
            store,
            ms,
            version="0.5",
            chunks_per_shard=2,
            compressors=[ZstdCodec(level=5)],
        )
    finally:
        nz.config.memory_target = old_target

    cfg = _read_scale0_meta(store)["codecs"][0]["configuration"]
    zstd = next(c for c in cfg["codecs"] if c["name"] == "zstd")
    assert zstd["configuration"]["level"] == 5


@pytest.mark.parametrize(
    "dims, shape, chunks, mt",
    [
        (("t", "c", "y", "x"), (1, 2, 290, 370), 16, 1),
        (("t", "c", "z", "y", "x"), (1, 2, 62, 94, 94), 16, 560_000),
    ],
)
def test_large_path_multiscale_reads_back(dims, shape, chunks, mt, tmp_path):
    """A realistic multi-scale write on the large-array region path reads back
    byte-identical. Covers the 2D single-region and 3D multi-region (z-slab)
    paths with prime-factored spatial axes, so every level has partial final
    shards. Confirms the dask shard-alignment PerformanceWarning on this path is
    benign: each shard is written by exactly one writer, so no data is lost."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=shape, dtype=np.uint8)
    arr = da.from_array(data, chunks=chunks)
    image = nz.to_ngff_image(arr, dims=dims)
    ms = nz.to_multiscales(image, scale_factors=[2, 4], chunks=chunks)
    level_shapes = [tuple(img.data.shape) for img in ms.images]

    old_target = nz.config.memory_target
    nz.config.memory_target = mt
    try:
        store = tmp_path / "test.ome.zarr"
        nz.to_ngff_zarr(store, ms, version="0.5", chunks_per_shard=2)
    finally:
        nz.config.memory_target = old_target

    back = nz.from_ngff_zarr(store)
    # Full-resolution data must survive the region write byte for byte.
    assert np.array_equal(np.asarray(back.images[0].data), data)
    # Every pyramid level must be present and fully readable (no truncation).
    assert len(back.images) == len(level_shapes)
    for i, expected_shape in enumerate(level_shapes):
        assert tuple(back.images[i].data.shape) == expected_shape
        np.asarray(back.images[i].data)
