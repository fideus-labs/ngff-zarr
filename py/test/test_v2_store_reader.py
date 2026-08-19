# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the internal zarr format 2 MutableMapping store reader.

Fixtures are authored with zarr-python (still installed during the
migration) into a temporary directory and loaded into a plain in-memory
``dict[str, bytes]``, the store form the reader targets (zip archives,
tifffile's aszarr store, in-memory mappings).
"""

from pathlib import Path

import numcodecs
import numpy as np
import pytest
import zarr
from ngff_zarr._v2_store_reader import (
    V2Array,
    V2Group,
    open_v2,
    open_v2_array,
    open_v2_group,
)


def _store_to_dict(path: Path) -> dict[str, bytes]:
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file()
    }


def _separator_kwargs(separator: str) -> dict:
    if separator == "/":
        return {
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "/"},
            }
        }
    return {}


@pytest.mark.parametrize("separator", [".", "/"])
def test_round_trip_group(tmp_path, separator):
    data = np.arange(5 * 7, dtype="<u2").reshape(5, 7)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    root.attrs["hello"] = "world"
    arr = root.create_array(
        "image",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(2, 3),
        compressor=numcodecs.Blosc(),
        attributes={"units": "mm"},
        **_separator_kwargs(separator),
    )
    arr[:] = data

    store = _store_to_dict(tmp_path / "test.zarr")
    group = open_v2_group(store)
    assert group.attrs == {"hello": "world"}
    assert "image" in group
    assert "missing" not in group

    result = group.open_array("image")
    assert result.attrs == {"units": "mm"}
    assert result.shape == (5, 7)
    assert result.chunks == (2, 3)
    assert result.dtype == np.dtype("<u2")

    darr = result.to_dask()
    # Chunking matches the on-disk grid, including trimmed edge chunks.
    assert darr.chunks == ((2, 2, 1), (3, 3, 1))
    np.testing.assert_array_equal(darr.compute(), data)


@pytest.mark.parametrize("separator", [".", "/"])
def test_missing_chunk_fill_value(tmp_path, separator):
    data = np.arange(16, dtype="<i4").reshape(4, 4)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    arr = root.create_array(
        "image",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(2, 2),
        fill_value=7,
        **_separator_kwargs(separator),
    )
    arr[:] = data

    store = _store_to_dict(tmp_path / "test.zarr")
    chunk_key = "image/0" + separator + "1"
    del store[chunk_key]

    result = open_v2_group(store).open_array("image").to_dask().compute()
    expected = data.copy()
    expected[0:2, 2:4] = 7
    np.testing.assert_array_equal(result, expected)


def test_fill_value_nan(tmp_path):
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    arr = root.create_array(
        "image", shape=(4,), dtype="<f4", chunks=(2,), fill_value=np.nan
    )
    arr[0:2] = [1.5, 2.5]

    store = _store_to_dict(tmp_path / "test.zarr")
    # zarr-python already skips storing the all-fill chunk; make sure.
    store.pop("image/1", None)

    result = open_v2_group(store).open_array("image").to_dask().compute()
    np.testing.assert_array_equal(result[0:2], [1.5, 2.5])
    assert np.isnan(result[2:4]).all()


def test_fortran_order_with_filters(tmp_path):
    data = np.arange(6 * 8, dtype="<i4").reshape(6, 8)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    arr = root.create_array(
        "image",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(3, 4),
        order="F",
        filters=[numcodecs.Delta(dtype="<i4")],
        compressor=numcodecs.Zstd(),
    )
    arr[:] = data

    store = _store_to_dict(tmp_path / "test.zarr")
    result = open_v2_group(store).open_array("image").to_dask().compute()
    np.testing.assert_array_equal(result, data)


def test_uncompressed(tmp_path):
    data = np.arange(12, dtype="<u1").reshape(3, 4)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    arr = root.create_array(
        "image", shape=data.shape, dtype=data.dtype, chunks=(3, 4), compressor=None
    )
    arr[:] = data

    store = _store_to_dict(tmp_path / "test.zarr")
    result = open_v2_group(store).open_array("image").to_dask().compute()
    np.testing.assert_array_equal(result, data)


def test_big_endian(tmp_path):
    data = np.arange(12, dtype=">u2").reshape(3, 4)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    arr = root.create_array("image", shape=data.shape, dtype=">u2", chunks=(2, 2))
    arr[:] = data

    store = _store_to_dict(tmp_path / "test.zarr")
    result = open_v2_group(store).open_array("image")
    assert result.dtype == np.dtype(">u2")
    np.testing.assert_array_equal(result.to_dask().compute(), data)


def test_consolidated_metadata(tmp_path):
    data = np.arange(8, dtype="<f8")
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    root.attrs["title"] = "consolidated"
    arr = root.create_array("image", shape=(8,), dtype="<f8", chunks=(4,))
    arr[:] = data
    zarr.consolidate_metadata(str(tmp_path / "test.zarr"), zarr_format=2)

    store = _store_to_dict(tmp_path / "test.zarr")
    # Metadata must resolve from .zmetadata alone: drop the per-node docs.
    for key in [k for k in store if k.rsplit("/", 1)[-1].startswith(".z")]:
        if key != ".zmetadata":
            del store[key]

    group = open_v2_group(store)
    assert group.attrs == {"title": "consolidated"}
    assert group.array_keys() == ["image"]
    result = group.open_array("image").to_dask().compute()
    np.testing.assert_array_equal(result, data)


def test_group_navigation(tmp_path):
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    nested = root.create_group("a").create_group("b")
    nested.attrs["depth"] = 2
    arr = nested.create_array("0", shape=(2, 2), dtype="<u1", chunks=(2, 2))
    arr[:] = np.ones((2, 2), dtype="<u1")

    store = _store_to_dict(tmp_path / "test.zarr")
    group = open_v2_group(store)
    assert group.group_keys() == ["a"]
    assert group.array_keys() == []
    assert "a/b/0" in group
    assert "a/b" in group

    nested_group = group.open_group("a/b")
    assert nested_group.attrs == {"depth": 2}
    assert nested_group.array_keys() == ["0"]
    assert isinstance(group["a"], V2Group)
    assert isinstance(group["a/b/0"], V2Array)
    np.testing.assert_array_equal(
        group["a/b/0"].to_dask().compute(), np.ones((2, 2), dtype="<u1")
    )

    with pytest.raises(KeyError):
        group["a/missing"]
    with pytest.raises(ValueError):
        group.open_array("a")
    with pytest.raises(ValueError):
        group.open_group("a/b/0")


def test_open_v2_autodetects_root_array(tmp_path):
    data = np.arange(6, dtype="<i8").reshape(2, 3)
    arr = zarr.create_array(
        tmp_path / "array.zarr",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(2, 3),
        zarr_format=2,
    )
    arr[:] = data

    store = _store_to_dict(tmp_path / "array.zarr")
    node = open_v2(store)
    assert isinstance(node, V2Array)
    np.testing.assert_array_equal(node.to_dask().compute(), data)

    with pytest.raises(ValueError):
        open_v2_group(store)


def test_open_v2_missing_node(tmp_path):
    with pytest.raises(KeyError):
        open_v2({})
    with pytest.raises(ValueError):
        open_v2_array(
            {".zgroup": b'{"zarr_format": 2}'},
        )


def test_lazy_reads_do_not_touch_chunks(tmp_path):
    data = np.arange(16, dtype="<u2").reshape(4, 4)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=2)
    arr = root.create_array("image", shape=data.shape, dtype=data.dtype, chunks=(2, 2))
    arr[:] = data

    accessed: list[str] = []

    class RecordingStore(dict):
        def __getitem__(self, key):
            accessed.append(key)
            return super().__getitem__(key)

    def chunk_keys():
        return [k for k in accessed if not k.rsplit("/", 1)[-1].startswith(".z")]

    store = RecordingStore(_store_to_dict(tmp_path / "test.zarr"))
    darr = open_v2_group(store).open_array("image").to_dask()
    assert chunk_keys() == []

    # Graph culling: a sliced compute only fetches the covered chunk.
    np.testing.assert_array_equal(darr[0:2, 0:2].compute(), data[0:2, 0:2])
    assert chunk_keys() == ["image/0.0"]
