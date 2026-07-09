# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dims, shape, chunks, mt",
    [
        (("t", "c", "z", "y", "x"), (1, 1, 40, 128, 256), 16, 800_000),
        (("t", "c", "z", "y", "x"), (1, 1, 40, 160, 320), 16, 2_000_000),
    ],
)
def test_large_array_region_write_covers_whole_array(dims, shape, chunks, mt):
    """The large-array (memory_target) write path must write the whole array.

    Regression for silent data loss: _compute_plane_regions counted planes and
    strips by the memory-band width but stepped each region by the chunk width,
    leaving the array tail unwritten (it read back as the fill value). Data is
    in 1..255 so any zero read back is unambiguously lost.
    """
    data = np.random.default_rng(0).integers(1, 256, size=shape, dtype=np.uint8)
    image = nz.to_ngff_image(da.from_array(data, chunks=chunks), dims=dims)
    ms = nz.to_multiscales(image, scale_factors=[1], chunks=chunks)

    old = nz.config.memory_target
    nz.config.memory_target = mt
    try:
        store = tempfile.mkdtemp(suffix=".ome.zarr")
        nz.to_ngff_zarr(store, ms, version="0.5", chunks_per_shard=2)
    finally:
        nz.config.memory_target = old

    result = np.asarray(nz.from_ngff_zarr(store).images[0].data)
    assert np.array_equal(result, data)
