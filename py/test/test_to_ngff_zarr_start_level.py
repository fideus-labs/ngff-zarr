# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import (
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)
needs_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)
CHUNKS = (1, 32, 32, 32)


def _image(data):
    return to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )


def _level0_store(path, ngff_version="0.4", shape=(1, 64, 64, 64)):
    values = np.random.default_rng(0).random(shape).astype(np.float32)
    multiscales = to_multiscales(
        _image(values), scale_factors=[], chunks=CHUNKS, cache=False
    )
    to_ome_zarr(path, multiscales, version=ngff_version)
    return values


def _chunk_mtimes(path, level="scale0"):
    return {
        file: file.stat().st_mtime_ns
        for file in (Path(path) / level).rglob("*")
        if file.is_file() and not file.name.startswith(".") and file.name != "zarr.json"
    }


@pytest.mark.parametrize(
    "ngff_version", ["0.4", pytest.param("0.5", marks=needs_zarr_v3)]
)
def test_append_level_leaves_level0_untouched(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path, ngff_version)
    zarr.open_group(path, mode="r+").attrs["mine"] = {"k": 1}
    before = _chunk_mtimes(path)
    assert before

    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    expected = multiscales.images[1].data.compute()
    to_ome_zarr(path, multiscales, version=ngff_version, overwrite=False, start_level=1)

    assert _chunk_mtimes(path) == before
    read = from_ome_zarr(path)
    assert len(read.images) == 2
    np.testing.assert_allclose(
        read.images[1].data.compute(), expected, rtol=1e-5, atol=1e-6
    )
    assert read.images[1].scale["z"] == 2.0
    assert dict(zarr.open_group(path, mode="r").attrs)["mine"] == {"k": 1}


def test_append_two_levels_chains_from_disk(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2, 4], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)

    read = from_ome_zarr(path)
    assert [image.data.shape for image in read.images] == [
        (1, 64, 64, 64),
        (1, 32, 32, 32),
        (1, 16, 16, 16),
    ]


def test_rerun_replaces_the_appended_levels(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)
    to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)

    read = from_ome_zarr(path)
    assert len(read.images) == 2
    assert read.images[1].data.compute().max() > 0.0


def test_metadata_only_creates_only_the_new_levels(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    values = _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    to_ome_zarr(
        path,
        multiscales,
        version="0.4",
        overwrite=False,
        start_level=1,
        metadata_only=True,
    )

    read = from_ome_zarr(path)
    np.testing.assert_array_equal(read.images[0].data.compute(), values)
    assert read.images[1].data.shape == (1, 32, 32, 32)
    assert read.images[1].data.compute().max() == 0.0


def test_overwrite_with_start_level_is_refused(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    with pytest.raises(ValueError, match="overwrite=False"):
        to_ome_zarr(path, multiscales, version="0.4", overwrite=True, start_level=1)
    with pytest.raises(ValueError, match="out of range"):
        to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=2)


def test_missing_or_mismatched_level_is_refused(tmp_path):
    values = np.random.default_rng(0).random((1, 64, 64, 64)).astype(np.float32)
    multiscales = to_multiscales(
        _image(values), scale_factors=[2], chunks=CHUNKS, cache=False
    )
    empty = str(tmp_path / "empty.ome.zarr")
    with pytest.raises(ValueError, match="no readable array"):
        to_ome_zarr(empty, multiscales, version="0.4", overwrite=False, start_level=1)

    smaller = str(tmp_path / "smaller.ome.zarr")
    _level0_store(smaller, shape=(1, 32, 32, 32))
    with pytest.raises(ValueError, match="shape"):
        to_ome_zarr(smaller, multiscales, version="0.4", overwrite=False, start_level=1)


def test_level_read_from_the_array_it_replaces_is_refused(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)

    stored = from_ome_zarr(path)
    multiscales.images[1] = stored.images[1]
    with pytest.raises(ValueError, match="replaces before its data is computed"):
        to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)
    assert zarr.open_group(path, mode="r")["scale1/image"][:].max() > 0.0


def test_existing_store_without_start_level_names_the_options(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    with pytest.raises(ValueError, match="start_level"):
        to_ome_zarr(path, multiscales, version="0.4", overwrite=False)


def test_start_level_two_appends_from_level_one(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    first = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, first, version="0.4", overwrite=False, start_level=1)

    base = from_ome_zarr(path).images[0]
    both = to_multiscales(base, scale_factors=[2, 4], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, both, version="0.4", overwrite=False, start_level=2)

    read = from_ome_zarr(path)
    assert [image.data.shape for image in read.images] == [
        (1, 64, 64, 64),
        (1, 32, 32, 32),
        (1, 16, 16, 16),
    ]
    assert read.images[2].scale["x"] == 4.0


def test_supplied_level_is_written_beside_the_existing_one(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    multiscales.images[1].data = da.full(
        (1, 32, 32, 32), 7.0, chunks=CHUNKS, dtype=np.float32
    )
    to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)

    level1 = zarr.open_group(path, mode="r")["scale1/image"][:]
    np.testing.assert_array_equal(level1, 7.0)


def test_interrupted_append_leaves_the_store_readable(tmp_path, monkeypatch):
    import sys

    writer_module = sys.modules["ngff_zarr.to_ngff_zarr"]

    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)

    def fail(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(writer_module, "_write_array_with_zarrista", fail)
    with pytest.raises(OSError):
        to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)

    read = from_ome_zarr(path)
    assert len(read.images) == 1
    assert read.images[0].data.compute().max() > 0.0


def test_interrupted_repeated_append_leaves_the_store_readable(tmp_path, monkeypatch):
    import sys

    writer_module = sys.modules["ngff_zarr.to_ngff_zarr"]

    path = str(tmp_path / "image.ome.zarr")
    _level0_store(path)
    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(base, scale_factors=[2], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)
    assert len(from_ome_zarr(path).images) == 2

    def fail(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(writer_module, "_write_array_with_zarrista", fail)
    with pytest.raises(OSError):
        to_ome_zarr(path, multiscales, version="0.4", overwrite=False, start_level=1)

    read = from_ome_zarr(path)
    assert len(read.images) == 1
    assert read.images[0].data.compute().max() > 0.0
