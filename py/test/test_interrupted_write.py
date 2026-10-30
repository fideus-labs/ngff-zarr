# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A write that fails part way never leaves a store that reads as complete."""

import dask.array
import numpy as np
import pytest
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr


def _failing_multiscales():
    image = to_ngff_image(np.ones((64, 64), dtype=np.uint8), dims=("y", "x"))
    multiscales = to_multiscales(image, scale_factors=[2], chunks=32, cache=False)

    def fail(block):
        raise OSError("No space left on device")

    last = multiscales.images[-1]
    last.data = dask.array.map_blocks(fail, last.data, dtype=last.data.dtype)
    return multiscales


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_failed_level_leaves_no_readable_multiscales(tmp_path, version):
    store = tmp_path / "out.ome.zarr"
    with pytest.raises(OSError, match="No space left"):
        to_ome_zarr(store, _failing_multiscales(), version=version)

    with pytest.raises(ValueError, match="interrupted write"):
        from_ome_zarr(store)


def test_failed_zip_keeps_the_previous_archive(tmp_path, monkeypatch):
    archive = tmp_path / "out.ozx"
    image = to_ngff_image(np.ones((64, 64), dtype=np.uint8), dims=("y", "x"))
    to_ome_zarr(archive, to_multiscales(image, scale_factors=[2], cache=False))
    before = archive.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("ngff_zarr.rfc9_zip._write_rfc9_comment", fail)
    twos = to_ngff_image(np.full((64, 64), 2, dtype=np.uint8), dims=("y", "x"))
    with pytest.raises(OSError, match="No space left"):
        to_ome_zarr(archive, to_multiscales(twos, scale_factors=[2], cache=False))

    assert archive.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == ["out.ozx"]


def test_ozx_overwrite_false_keeps_the_archive(tmp_path):
    archive = tmp_path / "out.ozx"
    image = to_ngff_image(np.ones((64, 64), dtype=np.uint8), dims=("y", "x"))
    to_ome_zarr(archive, to_multiscales(image, scale_factors=[2], cache=False))
    before = archive.read_bytes()

    with pytest.raises(FileExistsError):
        to_ome_zarr(
            archive,
            to_multiscales(image, scale_factors=[2], cache=False),
            overwrite=False,
        )
    assert archive.read_bytes() == before
