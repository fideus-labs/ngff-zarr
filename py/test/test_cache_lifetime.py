# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The to_multiscales disk cache lives as long as the data that reads it."""

import gc
import signal
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from ngff_zarr import (
    Methods,
    config,
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
)


def _ramp(shape):
    # Strictly positive, so a level read from a deleted cache (fill value 0)
    # shows up as zeros.
    return (np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) % 1000) + 1


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "cache_store", cache)
    return cache


def test_cached_multiscales_written_twice(cache_dir, tmp_path):
    image = to_ngff_image(_ramp((256, 256)), dims=("y", "x"))
    multiscales = to_multiscales(
        image, scale_factors=[2], method=Methods.DASK_BIN_SHRINK, cache=True
    )

    for name in ("first.ome.zarr", "second.ome.zarr", "third.ozx"):
        to_ome_zarr(tmp_path / name, multiscales)
        for level in from_ome_zarr(tmp_path / name).images:
            assert np.asarray(level.data).min() > 0, f"{name} has zero-filled chunks"


def test_exact_strategy_reads_cached_level(cache_dir, tmp_path):
    # 101 // 2 // 2 != 101 // 3, so "exact" writes the cached level 2 as is.
    image = to_ngff_image(_ramp((101, 101)), dims=("y", "x"))
    multiscales = to_multiscales(
        image, scale_factors=[2, 3], method=Methods.DASK_BIN_SHRINK, cache=True
    )

    to_ome_zarr(tmp_path / "out.ome.zarr", multiscales, scale_strategy="exact")
    for level in from_ome_zarr(tmp_path / "out.ome.zarr").images:
        assert np.asarray(level.data).min() > 0


def test_cache_removed_once_unreferenced(cache_dir):
    image = to_ngff_image(_ramp((64, 64, 64)), dims=("z", "y", "x"))
    multiscales = to_multiscales(
        image, scale_factors=[2], method=Methods.DASK_BIN_SHRINK, cache=True
    )
    assert any(cache_dir.iterdir())

    del multiscales
    gc.collect()
    assert not list(cache_dir.glob("*-cache-*"))


def test_cache_leaves_sigint_alone(cache_dir):
    before = signal.getsignal(signal.SIGINT)
    image = to_ngff_image(_ramp((64, 64)), dims=("y", "x"))
    to_multiscales(image, scale_factors=[2], method=Methods.DASK_BIN_SHRINK, cache=True)
    assert signal.getsignal(signal.SIGINT) is before


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(SIGTERM) terminates a Windows process without running handlers",
)
def test_sigterm_stops_and_cleans_cache(tmp_path):
    cache = tmp_path / "cache"
    script = textwrap.dedent(
        f"""
        import os, signal, time
        import numpy as np
        from ngff_zarr import Methods, config, to_multiscales, to_ngff_image

        config.cache_store = {str(cache)!r}
        data = np.ones((64, 64), dtype=np.uint16)
        image = to_ngff_image(data, dims=("y", "x"))
        multiscales = to_multiscales(
            image, scale_factors=[2], method=Methods.DASK_BIN_SHRINK, cache=True
        )
        assert os.listdir({str(cache)!r})
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
        print("still running")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert "still running" not in result.stdout
    assert result.returncode == 128 + signal.SIGTERM, result.stderr
    assert not list(cache.glob("*-cache-*"))


def test_failed_cache_write_removed_at_exit(tmp_path):
    cache = tmp_path / "cache"
    script = textwrap.dedent(
        f"""
        import dask.array
        from ngff_zarr import Methods, config, to_multiscales, to_ngff_image

        config.cache_store = {str(cache)!r}

        def fail(block, block_info=None):
            if block_info and block_info[0]["chunk-location"][0] == 1:
                raise OSError("No space left on device")
            return block

        data = dask.array.ones((128, 64, 64), dtype="uint16", chunks=32)
        image = to_ngff_image(data.map_blocks(fail), dims=("z", "y", "x"))
        try:
            to_multiscales(
                image, scale_factors=[2], method=Methods.DASK_BIN_SHRINK, cache=True
            )
        except OSError:
            print("write failed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert "write failed" in result.stdout, result.stderr
    assert not list(cache.glob("*-cache-*"))
