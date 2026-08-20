# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The zarr v3 write path streams its blocks instead of materializing them.

``array[:] = arr.compute()`` holds the whole result in memory before the first
byte is written, which caps a write at what fits in RAM however lazily the
array was built. These tests pin the streaming behaviour by counting how many
output blocks are alive at once while the write runs.
"""

import weakref

import dask
import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr

pytestmark = pytest.mark.skipif(
    zarr.__version__ < "3", reason="the direct zarr v3 write path needs zarr v3"
)


class _TrackedBlock(np.ndarray):
    """An ndarray a weak reference can be taken to."""


def _counting_array(nblocks: int, block: int):
    """A dask array that records how many of its blocks are alive at once.

    Every block registers a weak reference to itself as it is produced, so a
    block the writer has stored and released stops counting. The high water
    mark is the number of blocks the write path holds simultaneously.
    """
    produced: list = []
    peak = {"blocks": 0}

    def make(x):
        tracked = np.zeros(x.shape, dtype="uint8").view(_TrackedBlock)
        produced.append(weakref.ref(tracked))
        alive = sum(1 for reference in produced if reference() is not None)
        peak["blocks"] = max(peak["blocks"], alive)
        return tracked

    source = da.zeros(
        (nblocks * block, block, block), chunks=(block, block, block), dtype="uint8"
    )
    return source.map_blocks(make, dtype="uint8"), peak


@pytest.mark.parametrize("nblocks", [8])
def test_write_holds_only_the_blocks_in_flight(tmp_path, nblocks):
    data, peak = _counting_array(nblocks, 16)
    image = to_ngff_image(data, dims=["z", "y", "x"])
    with dask.config.set(scheduler="synchronous"):
        to_ngff_zarr(
            tmp_path / "streamed.ome.zarr",
            to_multiscales(image, scale_factors=[], chunks=16),
            overwrite=True,
        )

    # One worker writes one block at a time, so a materializing write is the
    # only way to see them all at once.
    assert peak["blocks"] < nblocks


def test_streamed_write_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    expected = rng.integers(0, 255, size=(48, 32, 32), dtype="uint8")
    image = to_ngff_image(
        da.from_array(expected, chunks=(16, 16, 16)), dims=["z", "y", "x"]
    )
    store = tmp_path / "round-trip.ome.zarr"
    to_ngff_zarr(
        store, to_multiscales(image, scale_factors=[], chunks=16), overwrite=True
    )

    from ngff_zarr import from_ngff_zarr

    written = np.asarray(from_ngff_zarr(store).images[0].data)
    np.testing.assert_array_equal(written, expected)
