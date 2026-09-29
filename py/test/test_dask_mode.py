# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from ngff_zarr import Methods, to_multiscales, to_ngff_image


def _image(data, dims=("c", "z", "y", "x")):
    dims = list(dims)
    return to_ngff_image(
        data,
        dims=dims,
        scale=dict.fromkeys(dims, 1),
        translation=dict.fromkeys(dims, 0),
    )


def _reference_mode(data, factors_zyx):
    """The most frequent value of each (1, fz, fy, fx) block, the smallest on a tie."""
    fz, fy, fx = factors_zyx
    shape = (
        data.shape[0],
        data.shape[1] // fz,
        data.shape[2] // fy,
        data.shape[3] // fx,
    )
    out = np.empty(shape, data.dtype)
    for index in np.ndindex(*shape):
        c, z, y, x = index
        block = data[
            c, z * fz : (z + 1) * fz, y * fy : (y + 1) * fy, x * fx : (x + 1) * fx
        ]
        values, counts = np.unique(block, return_counts=True)
        out[index] = values[np.argmax(counts)]
    return out


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int64, np.bool_])
@pytest.mark.parametrize("scale_factors", [[2], [2, 4], [3], [{"x": 2, "y": 4}]])
def test_each_block_takes_its_most_frequent_label(dtype, scale_factors):
    rng = np.random.default_rng(0)
    labels = [False, True] if dtype is np.bool_ else [0, 1, 3, 7]
    data = rng.choice(np.asarray(labels, dtype=dtype), size=(2, 35, 34, 33))
    multiscales = to_multiscales(
        _image(data),
        scale_factors,
        method=Methods.DASK_IMAGE_MODE,
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
        got = level.data.compute()
        assert got.dtype == dtype
        np.testing.assert_array_equal(got, _reference_mode(previous, step))
        for dim in ("z", "y", "x"):
            assert level.scale[dim] == absolute[dim]
            assert level.translation[dim] == 0.5 * (absolute[dim] - 1)
        previous = got
        previous_factor = absolute


def test_a_block_of_one_label_keeps_it():
    """The level's translation places each sample at the centre of its block, so the
    sample summarizes that block and no neighbour."""
    rng = np.random.default_rng(1)
    blocks = rng.integers(1, 5, (1, 8, 8, 8)).astype(np.uint8)
    data = blocks.repeat(2, 1).repeat(2, 2).repeat(2, 3)
    multiscales = to_multiscales(
        _image(data), [2], method=Methods.DASK_IMAGE_MODE, cache=False
    )
    np.testing.assert_array_equal(multiscales.images[1].data.compute(), blocks)


def test_the_result_does_not_depend_on_the_chunks():
    rng = np.random.default_rng(2)
    data = rng.choice(np.asarray([0, 2, 5], np.uint8), size=(1, 40, 21, 19))
    levels = [
        to_multiscales(
            _image(data),
            [2, 4],
            method=Methods.DASK_IMAGE_MODE,
            chunks=chunks,
            cache=False,
        ).images
        for chunks in ((1, 40, 21, 19), (1, 3, 5, 7), (1, 16, 16, 16))
    ]
    for level in range(1, 3):
        reference = levels[0][level].data.compute()
        for other in levels[1:]:
            np.testing.assert_array_equal(other[level].data.compute(), reference)
