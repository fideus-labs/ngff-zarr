# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the zarrista compatibility layer (ngff_zarr._zarrista_utils).

Every zarrista-written store is verified by reading it back with zarr-python,
mirroring the Phase 01 capability spike. zarrista requires Python >= 3.11, so
tests that exercise it skip via ``pytest.importorskip`` where it is absent.
"""

import json

import dask.array as da
import numpy as np
import pytest
import zarr
import zarr.storage
from ngff_zarr._zarrista_utils import (
    consolidate_metadata,
    create_zarrista_array,
    create_zarrista_group,
    normalize_store,
    open_zarrista_lazy,
    write_dask_array,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    return rng.integers(0, 2**16, size=(16, 32), dtype=np.uint16)


def test_v2_group_and_array_roundtrip(tmp_path, sample_data):
    pytest.importorskip("zarrista")
    from numcodecs import Blosc

    store_path = tmp_path / "v2.zarr"
    attributes = {"multiscales": [{"version": "0.4"}]}
    group = create_zarrista_group(store_path, attributes, 2)
    assert group.attrs == attributes

    arr = create_zarrista_array(
        store_path,
        "0",
        sample_data.shape,
        sample_data.dtype,
        (8, 16),
        2,
        compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
    )
    write_dask_array(da.from_array(sample_data, chunks=(8, 16)), arr)
    consolidate_metadata(store_path, 2)

    zgroup = zarr.open_group(str(store_path), mode="r")
    assert zgroup.attrs["multiscales"] == attributes["multiscales"]
    assert np.array_equal(zgroup["0"][:], sample_data)

    zarray = json.loads((store_path / "0" / ".zarray").read_text())
    assert zarray["zarr_format"] == 2
    assert zarray["dimension_separator"] == "/"
    assert zarray["compressor"]["id"] == "blosc"

    zmetadata = json.loads((store_path / ".zmetadata").read_text())
    assert zmetadata["zarr_consolidated_format"] == 1
    assert "0/.zarray" in zmetadata["metadata"]
    consolidated = zarr.open_consolidated(str(store_path))
    assert np.array_equal(consolidated["0"][:], sample_data)

    lazy = open_zarrista_lazy(store_path, "0")
    assert lazy.dtype == sample_data.dtype
    assert lazy.chunks == ((8, 8), (16, 16))
    assert np.array_equal(np.asarray(lazy), sample_data)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required to read zarr format 3",
)
def test_v3_array_roundtrip_gzip_and_blosc(tmp_path, sample_data):
    pytest.importorskip("zarrista")
    from numcodecs import Blosc, GZip

    store_path = tmp_path / "v3.zarr"
    create_zarrista_group(store_path, {"ome": {"version": "0.5"}}, 3)
    gzip_arr = create_zarrista_array(
        store_path,
        "gzip",
        sample_data.shape,
        sample_data.dtype,
        (8, 16),
        3,
        compressor=GZip(level=5),
        dimension_names=("y", "x"),
    )
    write_dask_array(da.from_array(sample_data, chunks=(8, 16)), gzip_arr)
    blosc_arr = create_zarrista_array(
        store_path,
        "blosc",
        sample_data.shape,
        sample_data.dtype,
        (8, 16),
        3,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    write_dask_array(da.from_array(sample_data, chunks=(8, 16)), blosc_arr)
    consolidate_metadata(store_path, 3)

    zgroup = zarr.open_group(str(store_path), mode="r")
    assert zgroup.attrs["ome"] == {"version": "0.5"}
    for name in ("gzip", "blosc"):
        assert np.array_equal(zgroup[name][:], sample_data)
        assert np.array_equal(
            np.asarray(open_zarrista_lazy(store_path, name)), sample_data
        )

    gzip_codecs = [c.__class__.__name__ for c in zgroup["gzip"].metadata.codecs]
    assert "GzipCodec" in gzip_codecs
    assert tuple(zgroup["gzip"].metadata.dimension_names) == ("y", "x")
    blosc_codecs = [c.__class__.__name__ for c in zgroup["blosc"].metadata.codecs]
    assert "BloscCodec" in blosc_codecs

    root_metadata = json.loads((store_path / "zarr.json").read_text())
    assert "gzip" in root_metadata["consolidated_metadata"]["metadata"]
    consolidated = zarr.open_consolidated(str(store_path))
    assert np.array_equal(consolidated["blosc"][:], sample_data)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.8"),
    reason="zarr version >= 3.0.8 required to inspect sharded arrays",
)
def test_v3_sharded_array_roundtrip(tmp_path, sample_data):
    pytest.importorskip("zarrista")

    store_path = tmp_path / "v3_sharded.zarr"
    create_zarrista_group(store_path, {}, 3)
    arr = create_zarrista_array(
        store_path,
        "0",
        sample_data.shape,
        sample_data.dtype,
        (4, 8),
        3,
        compressor={"id": "zstd", "level": 3},
        chunks_per_shard=2,
    )
    assert arr.is_sharded
    write_dask_array(da.from_array(sample_data, chunks=(4, 8)), arr)

    zarray = zarr.open(str(store_path / "0"), mode="r")
    assert zarray.shards == (8, 16)
    assert zarray.chunks == (4, 8)
    assert np.array_equal(zarray[:], sample_data)

    lazy = open_zarrista_lazy(store_path, "0")
    assert lazy.chunks == ((4, 4, 4, 4), (8, 8, 8, 8))
    assert np.array_equal(np.asarray(lazy), sample_data)

    per_dim_arr = create_zarrista_array(
        store_path,
        "1",
        sample_data.shape,
        sample_data.dtype,
        (4, 8),
        3,
        chunks_per_shard=(4, 2),
    )
    write_dask_array(da.from_array(sample_data), per_dim_arr)
    zarray = zarr.open(str(store_path / "1"), mode="r")
    assert zarray.shards == (16, 16)
    assert zarray.chunks == (4, 8)
    assert np.array_equal(zarray[:], sample_data)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required to read zarr format 3",
)
def test_write_dask_array_multi_chunk(tmp_path):
    pytest.importorskip("zarrista")

    rng = np.random.default_rng(3)
    values = rng.random((32, 48)).astype(np.float32)
    store_path = tmp_path / "dask.zarr"
    arr = create_zarrista_array(store_path, "0", values.shape, values.dtype, (8, 16), 3)

    # Chunks misaligned with the target: write_dask_array rechunks to (8, 16).
    write_dask_array(da.from_array(values, chunks=(5, 7)), arr)

    zarray = zarr.open(str(store_path / "0"), mode="r")
    assert np.array_equal(zarray[:], values)

    lazy = open_zarrista_lazy(store_path, "0")
    assert lazy.dtype == np.float32
    assert lazy.chunks == ((8,) * 4, (16,) * 3)
    assert np.array_equal(np.asarray(lazy), values)

    block = np.full((8, 16), 0.5, dtype=np.float32)
    write_dask_array(
        da.from_array(block, chunks=(8, 16)),
        arr,
        region=(slice(8, 16), slice(16, 32)),
    )
    zarray = zarr.open(str(store_path / "0"), mode="r")
    assert np.array_equal(zarray[8:16, 16:32], block)
    assert np.array_equal(zarray[0:8, 0:16], values[0:8, 0:16])


def test_normalize_store_paths_and_rejections(tmp_path):
    target = tmp_path / "store.zarr"
    assert normalize_store(str(target)) == target
    assert normalize_store(target) == target

    local_store_cls = getattr(zarr.storage, "LocalStore", None)
    if local_store_cls is not None:
        assert normalize_store(local_store_cls(target)) == target
    directory_store_cls = getattr(zarr.storage, "DirectoryStore", None)
    if directory_store_cls is not None:
        assert normalize_store(directory_store_cls(str(target))) == target

    with pytest.raises(TypeError, match="local path"):
        normalize_store({})
    with pytest.raises(TypeError, match="local path"):
        normalize_store(zarr.storage.MemoryStore())
    with pytest.raises(ValueError, match="zip"):
        normalize_store(tmp_path / "archive.zip")
    with pytest.raises(ValueError, match="zip"):
        normalize_store(tmp_path / "archive.ozx")
