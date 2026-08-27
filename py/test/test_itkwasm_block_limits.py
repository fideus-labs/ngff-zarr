# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array as da
import numpy as np
import pytest
from ngff_zarr import Methods, to_multiscales, to_ngff_image
from ngff_zarr.methods import _itkwasm


def _image(data):
    return to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )


def _block_mean(data, factor):
    c, z, y, x = data.shape
    z, y, x = (
        (z // factor) * factor,
        (y // factor) * factor,
        (x // factor) * factor,
    )
    trimmed = data[:, :z, :y, :x].astype(np.float64)
    shaped = trimmed.reshape(
        c, z // factor, factor, y // factor, factor, x // factor, factor
    )
    return shaped.mean(axis=(2, 4, 6))


def test_bin_shrink_remainder_smaller_than_the_factor():
    data = np.random.default_rng(0).random((1, 514, 33, 41)).astype(np.float32)
    multiscales = to_multiscales(
        _image(data),
        scale_factors=[4],
        chunks=(1, 2, 33, 41),
        method=Methods.ITKWASM_BIN_SHRINK,
        cache=False,
    )
    level1 = multiscales.images[1].data.compute()
    assert level1.shape == (1, 128, 8, 10)
    np.testing.assert_allclose(level1, _block_mean(data, 4), rtol=1e-5, atol=1e-6)


def test_bin_shrink_remainder_keeps_translation():
    data = np.zeros((1, 9, 9, 9), dtype=np.uint8)
    multiscales = to_multiscales(
        _image(data),
        scale_factors=[2],
        chunks=(1, 4, 9, 9),
        method=Methods.ITKWASM_BIN_SHRINK,
        cache=False,
    )
    level1 = multiscales.images[1]
    assert level1.data.shape == (1, 4, 4, 4)
    assert level1.translation["z"] == 0.5
    assert level1.scale["z"] == 2


@pytest.mark.parametrize(
    "method, smoothing",
    [
        (Methods.ITKWASM_GAUSSIAN, "gaussian"),
        (Methods.ITKWASM_BIN_SHRINK, "bin_shrink"),
    ],
)
def test_oversized_block_raises_before_compute(monkeypatch, method, smoothing):
    monkeypatch.setitem(_itkwasm._WASM_BLOCK_LIMITS, smoothing, 1024)
    data = np.zeros((1, 32, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="wasm sandbox"):
        to_multiscales(
            _image(data),
            scale_factors=[2],
            chunks=(1, 32, 32, 32),
            method=method,
            cache=False,
        )


def test_gaussian_counts_integer_input_as_float32(monkeypatch):
    data = np.zeros((1, 32, 32, 32), dtype=np.uint8)
    limit = 2 * 32 * 32 * 32
    monkeypatch.setitem(_itkwasm._WASM_BLOCK_LIMITS, "gaussian", limit)
    monkeypatch.setitem(_itkwasm._WASM_BLOCK_LIMITS, "bin_shrink", limit)

    to_multiscales(
        _image(data),
        scale_factors=[2],
        chunks=(1, 32, 32, 32),
        method=Methods.ITKWASM_BIN_SHRINK,
        cache=False,
    )
    with pytest.raises(ValueError, match="wasm sandbox"):
        to_multiscales(
            _image(data),
            scale_factors=[2],
            chunks=(1, 32, 32, 32),
            method=Methods.ITKWASM_GAUSSIAN,
            cache=False,
        )


def test_overlap_larger_than_the_chunks_counts_the_merged_chunks(monkeypatch):
    from ngff_zarr.ngff_image import NgffImage

    data = da.zeros((1, 40, 8, 8), dtype=np.float32, chunks=(1, 4, 8, 8))
    image = NgffImage(data, dims=["c", "z", "y", "x"], scale={}, translation={})
    radius = [1, 1, 6]
    # Chunks of 4 along z are merged to at least 6 before the overlap of 6 on
    # each side is added: the block is at least (6 + 12) * 8 * 8 samples.
    naive = (4 + 12) * (8 + 2) * (8 + 2) * 4
    merged = (6 + 12) * (8 + 2) * (8 + 2) * 4
    monkeypatch.setitem(_itkwasm._WASM_BLOCK_LIMITS, "gaussian", (naive + merged) // 2)
    with pytest.raises(ValueError, match="wasm sandbox"):
        _itkwasm._check_wasm_block_size(image, False, "gaussian", radius)
