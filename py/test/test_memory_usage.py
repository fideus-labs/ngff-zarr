# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array
import numpy as np
import zarr
from ngff_zarr import (
    from_ngff_zarr,
    memory_usage,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)

rng = np.random.default_rng(12345)


def test_memory_usage():
    arr = rng.integers(0, 255, size=(4, 4, 4), dtype=np.uint8)
    arr = dask.array.from_array(arr, chunks=2)
    image = to_ngff_image(arr)
    multiscales = to_multiscales(image, scale_factors=[], chunks=2)
    store = zarr.storage.MemoryStore()
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
