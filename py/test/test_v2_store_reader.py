# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the internal zarr format 2 MutableMapping store reader.

Fixtures are authored with zarr-python (a test-only dependency) into a
temporary directory and loaded into a plain in-memory ``dict[str, bytes]``,
the store form the reader targets (zip archives, tifffile's aszarr store,
in-memory mappings).
"""

from pathlib import Path

import numcodecs
import numpy as np
import pytest
import zarr
from ngff_zarr._v2_store_reader import (
    AsyncStoreMapping,
    V2Array,
    V2Group,
    V3Array,
    V3Group,
    open_store_node,
    open_v2,
    open_v2_array,
    open_v2_group,
)
from packaging import version

pytestmark = pytest.mark.skipif(
    version.parse(zarr.__version__).major < 3,
    reason="fixtures are authored with the zarr-python 3 API",
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


# -- zarr format 3 mapping reader -------------------------------------------


def _write_v3_array(tmp_path, data, compressors, chunks=(2, 3)):
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=3)
    root.attrs["hello"] = "world"
    arr = root.create_array(
        "image",
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        compressors=compressors,
        attributes={"units": "mm"},
    )
    arr[:] = data
    return _store_to_dict(tmp_path / "test.zarr")


@pytest.mark.parametrize(
    "compressors",
    [
        None,
        [{"name": "gzip", "configuration": {"level": 5}}],
        [{"name": "zstd", "configuration": {"level": 3, "checksum": False}}],
        [
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "typesize": 2,
                },
            }
        ],
    ],
    ids=["uncompressed", "gzip", "zstd", "blosc"],
)
def test_v3_round_trip_group(tmp_path, compressors):
    data = np.arange(5 * 7, dtype="uint16").reshape(5, 7)
    store = _write_v3_array(tmp_path, data, compressors)

    root = open_store_node(store)
    assert isinstance(root, V3Group)
    assert root.attrs.asdict() == {"hello": "world"}
    assert root.keys() == ["image"]
    assert "image" in root

    node = root["image"]
    assert isinstance(node, V3Array)
    assert node.shape == (5, 7)
    assert node.chunks == (2, 3)
    assert node.attrs.asdict() == {"units": "mm"}

    darr = node.to_dask()
    assert darr.chunks == ((2, 2, 1), (3, 3, 1))
    np.testing.assert_array_equal(darr.compute(), data)


def test_v3_missing_chunk_fill_value(tmp_path):
    data = np.arange(4 * 4, dtype="float64").reshape(4, 4)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=3)
    arr = root.create_array(
        "image", shape=data.shape, dtype=data.dtype, chunks=(2, 2), fill_value=np.nan
    )
    arr[:] = data
    store = _store_to_dict(tmp_path / "test.zarr")
    removed = [key for key in store if key.startswith("image/c/1/")]
    assert removed
    for key in removed:
        del store[key]

    values = open_store_node(store)["image"].to_dask().compute()
    np.testing.assert_array_equal(values[:2], data[:2])
    assert np.isnan(values[2:]).all()


def test_v3_big_endian_bytes_codec(tmp_path):
    data = np.arange(4 * 4, dtype=">u2").reshape(4, 4)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=3)
    arr = root.create_array("image", shape=data.shape, dtype=data.dtype, chunks=(2, 2))
    arr[:] = data
    store = _store_to_dict(tmp_path / "test.zarr")

    node = open_store_node(store)["image"]
    np.testing.assert_array_equal(node.to_dask().compute(), data)


def test_v3_sharded_store_rejected(tmp_path):
    data = np.arange(8 * 8, dtype="uint16").reshape(8, 8)
    root = zarr.open_group(tmp_path / "test.zarr", mode="w", zarr_format=3)
    arr = root.create_array(
        "image", shape=data.shape, dtype=data.dtype, chunks=(2, 2), shards=(4, 4)
    )
    arr[:] = data
    store = _store_to_dict(tmp_path / "test.zarr")

    with pytest.raises(ValueError, match="zarrista"):
        open_store_node(store)["image"]


def test_open_store_node_auto_detects_format(tmp_path):
    data = np.arange(6, dtype="uint8").reshape(2, 3)
    v2_root = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    arr = v2_root.create_array("image", shape=data.shape, dtype=data.dtype)
    arr[:] = data

    node = open_store_node(_store_to_dict(tmp_path / "v2.zarr"))
    assert isinstance(node, V2Group)
    np.testing.assert_array_equal(node["image"].to_dask().compute(), data)

    with pytest.raises(KeyError):
        open_store_node({})


# -- tifffile aszarr through AsyncStoreMapping -------------------------------


def test_async_store_mapping_tifffile_single_level(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(32 * 32, dtype="uint16").reshape(32, 32)
    path = tmp_path / "single.tif"
    tifffile.imwrite(path, data, tile=(16, 16))

    store = tifffile.imread(path, aszarr=True)
    node = open_store_node(AsyncStoreMapping(store))
    assert isinstance(node, V3Array)
    darr = node.to_dask()
    assert darr.chunksize == (16, 16)
    np.testing.assert_array_equal(darr.compute(), data)


def test_async_store_mapping_tifffile_pyramid(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(32 * 32, dtype="uint16").reshape(32, 32)
    path = tmp_path / "pyramid.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(data, tile=(16, 16), subifds=1)
        writer.write(data[::2, ::2], tile=(16, 16), subfiletype=1)

    store = tifffile.imread(path, aszarr=True)
    root = open_store_node(AsyncStoreMapping(store))
    assert isinstance(root, V3Group)
    assert root.array_keys() == ["0", "1"]
    np.testing.assert_array_equal(root["0"].to_dask().compute(), data)
    np.testing.assert_array_equal(root["1"].to_dask().compute(), data[::2, ::2])
