# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array
import numpy as np
from ngff_zarr import (
    from_ngff_zarr,
    memory_usage,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)

rng = np.random.default_rng(12345)


def test_memory_usage(tmp_path):
    arr = rng.integers(0, 255, size=(4, 4, 4), dtype=np.uint8)
    arr = dask.array.from_array(arr, chunks=2)
    image = to_ngff_image(arr)
    multiscales = to_multiscales(image, scale_factors=[], chunks=2)
    store = tmp_path / "test.ome.zarr"
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)
    multiscales = from_ngff_zarr(store, version=version)

    image = multiscales.images[0]
    arr = image.data
    assert arr.nbytes == 64
    usage = memory_usage(image)
    assert usage == 64
    usage = memory_usage(image, {"z"})
    assert usage == 32

    arr1 = arr + 1
    assert arr1.nbytes == 64
    image.data = arr1
    usage = memory_usage(image)
    assert usage == 64
    usage = memory_usage(image, {"z"})
    assert usage == 32


def test_memory_usage_multi_byte_dtype():
    """The itemsize counts once, not once per dimension.

    The uint8 case above cannot see the difference: an itemsize of 1 is unchanged by being multiplied
    in repeatedly. Any wider dtype can, and by a lot -- a 4-D float32 array is overstated 64-fold if
    the itemsize enters the product on every axis.
    """
    arr = dask.array.zeros((3, 4, 5, 6), dtype=np.float32, chunks=(3, 2, 5, 6))
    image = to_ngff_image(arr, dims=("c", "z", "y", "x"))

    assert arr.nbytes == 3 * 4 * 5 * 6 * 4
    assert memory_usage(image) == arr.nbytes
    # One constrained dimension: that axis contributes its chunk rather than its full extent.
    assert memory_usage(image, {"z"}) == 3 * 2 * 5 * 6 * 4
