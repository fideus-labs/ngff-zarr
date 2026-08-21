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
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
from ngff_zarr.to_ngff_zarr import _blocks_write_disjoint_chunks
from packaging import version

pytestmark = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="the direct zarr v3 write path needs zarr v3",
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


@pytest.mark.parametrize("nblocks", [8, 32])
def test_write_holds_only_the_blocks_in_flight(tmp_path, nblocks):
    data, peak = _counting_array(nblocks, 16)
    image = to_ngff_image(data, dims=["z", "y", "x"])
    with dask.config.set(scheduler="synchronous"):
        to_ome_zarr(
            tmp_path / "streamed.ome.zarr",
            to_multiscales(image, scale_factors=[], chunks=16),
            overwrite=True,
        )

    # A fixed bound, not a fraction of the input: one worker holds the block it
    # is storing and the one already produced behind it, whether the array has
    # eight blocks or thirty-two. A materializing write holds every block, so
    # its high water mark is the block count.
    assert peak["blocks"] <= 2, f"{peak['blocks']} blocks alive at once"


def test_streamed_write_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    expected = rng.integers(0, 255, size=(48, 32, 32), dtype="uint8")
    image = to_ngff_image(
        da.from_array(expected, chunks=(16, 16, 16)), dims=["z", "y", "x"]
    )
    store = tmp_path / "round-trip.ome.zarr"
    to_ome_zarr(
        store, to_multiscales(image, scale_factors=[], chunks=16), overwrite=True
    )

    written = np.asarray(from_ome_zarr(store).images[0].data)
    np.testing.assert_array_equal(written, expected)


class _Target:
    """What the predicate reads off a zarr array: its chunks, and its shards."""

    def __init__(self, chunks, shards=None):
        self.chunks = chunks
        self.shards = shards


@pytest.mark.parametrize(
    ("blocks", "target", "region", "disjoint"),
    [
        # Every block boundary falls on a chunk boundary: the writes cannot meet.
        (((16, 16, 16), (16,), (16,)), _Target((16, 16, 16)), None, True),
        # A block twice the chunk still only spans whole chunks.
        (((32, 32), (16,), (16,)), _Target((16, 16, 16)), None, True),
        # An irregular block leaves a seam inside a chunk.
        (((10, 22), (16,), (16,)), _Target((16, 16, 16)), None, False),
        # Chunks smaller than a shard: two blocks update one shard.
        (
            ((16, 16), (16,), (16,)),
            _Target((16, 16, 16), shards=(32, 16, 16)),
            None,
            False,
        ),
        # Aligned blocks, but written at an offset that is not a chunk boundary.
        (
            ((16, 16), (16,), (16,)),
            _Target((16, 16, 16)),
            (slice(8, 40), slice(0, 16), slice(0, 16)),
            False,
        ),
    ],
)
def test_the_lock_is_taken_exactly_when_blocks_can_meet(
    blocks, target, region, disjoint
):
    """Concurrent writes are allowed only where two blocks cannot share a unit.

    A zarr chunk, or a shard when the array is sharded, is read, updated and
    rewritten whole, so two writers inside one of them lose data.
    """
    arr = da.zeros(tuple(sum(sizes) for sizes in blocks), chunks=blocks, dtype="uint8")

    assert _blocks_write_disjoint_chunks(arr, target, region) is disjoint
