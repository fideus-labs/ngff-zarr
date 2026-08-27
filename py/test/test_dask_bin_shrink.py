# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from ngff_zarr import Methods, to_multiscales, to_ngff_image
from ngff_zarr.methods._metadata import get_method_metadata


def _image(data, dims=("c", "z", "y", "x")):
    dims = list(dims)
    return to_ngff_image(
        data,
        dims=dims,
        scale=dict.fromkeys(dims, 1),
        translation=dict.fromkeys(dims, 0),
    )


def _random(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, shape, dtype=dtype)
    return rng.random(shape, dtype=dtype)


def _itkwasm_reference(data, factors_zyx):
    """BinShrink of one whole (c, z, y, x) block through itkwasm."""
    from ngff_zarr.methods._itkwasm import _itkwasm_chunk_bin_shrink

    shrink_xyz = list(reversed(factors_zyx))
    return np.stack(
        [_itkwasm_chunk_bin_shrink(channel, shrink_xyz) for channel in data]
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("scale_factors", [[2], [2, 4], [3], [{"x": 2, "y": 4}]])
def test_matches_itkwasm_bin_shrink(dtype, scale_factors):
    data = _random((2, 35, 34, 33), dtype)
    multiscales = to_multiscales(
        _image(data),
        scale_factors,
        method=Methods.DASK_BIN_SHRINK,
        chunks=(1, 16, 16, 16),
        cache=False,
    )
    assert len(multiscales.images) == len(scale_factors) + 1

    previous = data
    previous_factor = {"z": 1, "y": 1, "x": 1}
    for level, scale_factor in zip(multiscales.images[1:], scale_factors):
        absolute = (
            {dim: scale_factor.get(dim, 1) for dim in previous_factor}
            if isinstance(scale_factor, dict)
            else dict.fromkeys(previous_factor, scale_factor)
        )
        step = [absolute[dim] // previous_factor[dim] for dim in ("z", "y", "x")]
        expected = _itkwasm_reference(previous, step)
        got = level.data.compute()
        assert got.shape == expected.shape
        assert got.dtype == dtype
        if np.issubdtype(dtype, np.integer):
            np.testing.assert_array_equal(got, expected)
        else:
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
        for dim in ("z", "y", "x"):
            assert level.scale[dim] == absolute[dim]
            assert level.translation[dim] == 0.5 * (absolute[dim] - 1)
        previous = got
        previous_factor = absolute


def test_any_chunk_layout():
    data = _random((1, 514, 33, 41), np.float32)
    multiscales = to_multiscales(
        _image(data),
        scale_factors=[4],
        chunks=(1, 2, 33, 41),
        method=Methods.DASK_BIN_SHRINK,
        cache=False,
    )
    level1 = multiscales.images[1].data.compute()
    assert level1.shape == (1, 128, 8, 10)
    expected = (
        data[:, :512, :32, :40]
        .astype(np.float64)
        .reshape(1, 128, 4, 8, 4, 10, 4)
        .mean(axis=(2, 4, 6))
    )
    np.testing.assert_allclose(level1, expected, rtol=1e-6, atol=1e-7)


def test_two_dimensional_image():
    data = _random((65, 64), np.uint16)
    multiscales = to_multiscales(
        _image(data, dims=("y", "x")),
        scale_factors=[2],
        method=Methods.DASK_BIN_SHRINK,
        cache=False,
    )
    assert multiscales.images[1].data.shape == (32, 32)
    assert multiscales.metadata.type == "dask_bin_shrink"


def test_method_metadata():
    metadata = get_method_metadata(Methods.DASK_BIN_SHRINK)
    assert metadata.method == "dask.array.coarsen"
    assert metadata.version != "unknown"


def _window_mean_half_up(data, factor):
    """Reference in Python integers: floor((sum + n / 2) / n) per window."""
    c, z, y, x = data.shape
    out = np.empty((c, z // factor, y // factor, x // factor), dtype=object)
    for index in np.ndindex(out.shape):
        window = data[
            index[0],
            index[1] * factor : (index[1] + 1) * factor,
            index[2] * factor : (index[2] + 1) * factor,
            index[3] * factor : (index[3] + 1) * factor,
        ]
        total = sum(int(v) for v in window.ravel())
        count = factor**3
        out[index] = (2 * total + count) // (2 * count)
    return out


@pytest.mark.parametrize(
    "dtype, low, high",
    [
        (np.uint64, 2**64 - 64, 2**64 - 1),
        (np.int64, -(2**63), -(2**63) + 64),
        (np.int64, 2**63 - 64, 2**63 - 1),
        (np.uint64, 0, 2**64 - 1),
    ],
)
def test_64_bit_integers_keep_every_bit(dtype, low, high):
    rng = np.random.default_rng(0)
    values = rng.integers(low, high, size=(2, 4, 6, 8), endpoint=True, dtype=dtype)
    multiscales = to_multiscales(
        _image(values),
        scale_factors=[2],
        method=Methods.DASK_BIN_SHRINK,
        chunks=(1, 4, 4, 4),
        cache=False,
    )
    got = multiscales.images[1].data.compute()
    expected = _window_mean_half_up(values, 2)
    assert got.dtype == dtype
    assert [int(v) for v in got.ravel()] == [int(v) for v in expected.ravel()]
