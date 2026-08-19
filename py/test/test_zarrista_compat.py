# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the zarrista compatibility layer (ngff_zarr._zarrista_utils).

Every zarrista-written store is verified by reading it back with zarr-python,
mirroring the Phase 01 capability spike. zarrista requires Python >= 3.11, so
tests that exercise it skip via ``pytest.importorskip`` where it is absent.
"""

import json
import shutil

import dask.array as da
import numpy as np
import pytest
import zarr
import zarr.storage
from ngff_zarr._zarrista_utils import (
    consolidate_metadata,
    create_zarrista_array,
    create_zarrista_group,
    create_zarrista_subgroup,
    normalize_store,
    open_zarrista_lazy,
    use_zarrista_for,
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


def test_use_zarrista_for_path_targets(tmp_path):
    pytest.importorskip("zarrista")
    target = tmp_path / "store.zarr"
    assert use_zarrista_for(str(target))
    assert use_zarrista_for(target)

    class _PathLikeTarget:
        def __fspath__(self):
            return str(target)

    assert use_zarrista_for(_PathLikeTarget())


def test_use_zarrista_for_legacy_targets(tmp_path, monkeypatch):
    assert not use_zarrista_for({})
    assert not use_zarrista_for(zarr.storage.MemoryStore())
    local_store_cls = getattr(zarr.storage, "LocalStore", None)
    if local_store_cls is not None:
        assert not use_zarrista_for(local_store_cls(tmp_path))
    directory_store_cls = getattr(zarr.storage, "DirectoryStore", None)
    if directory_store_cls is not None:
        assert not use_zarrista_for(directory_store_cls(str(tmp_path)))

    # zarrista can only read zip archives, so zip targets keep zarr-python
    assert not use_zarrista_for(tmp_path / "archive.zip")
    assert not use_zarrista_for(str(tmp_path / "archive.ozx"))

    import ngff_zarr._zarrista_utils as zarrista_utils

    monkeypatch.setattr(zarrista_utils, "_ZARRISTA_AVAILABLE", False)
    assert not use_zarrista_for(tmp_path / "store.zarr")


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_create_group_overwrite_and_merge(tmp_path, zarr_format):
    pytest.importorskip("zarrista")

    store_path = tmp_path / "store.zarr"
    create_zarrista_group(store_path, {"a": 1, "keep": True}, zarr_format)
    stale = store_path / "stale.txt"
    stale.write_text("stale")

    # Append mode merges with existing attributes and keeps other content,
    # matching zarr.open_group(mode="a") followed by attribute assignment.
    group = create_zarrista_group(store_path, {"a": 2}, zarr_format)
    assert group.attrs == {"a": 2, "keep": True}
    assert stale.exists()

    # Overwrite clears prior content, matching zarr.open_group(mode="w").
    group = create_zarrista_group(store_path, {"b": 3}, zarr_format, overwrite=True)
    assert group.attrs == {"b": 3}
    assert not stale.exists()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_create_group_refuses_array_node(tmp_path, zarr_format):
    pytest.importorskip("zarrista")

    store_path = tmp_path / "store.zarr"
    create_zarrista_group(store_path, {}, zarr_format)
    create_zarrista_array(store_path, "0", (4,), np.uint8, (2,), zarr_format)
    with pytest.raises(ValueError, match="array node"):
        create_zarrista_group(store_path / "0", {}, zarr_format)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_create_zarrista_subgroup_nested(tmp_path, zarr_format):
    pytest.importorskip("zarrista")
    if zarr_format == 3 and zarr_version < version.parse("3.0.0b2"):
        pytest.skip("zarr version >= 3.0.0b2 required to read zarr format 3")

    store_path = tmp_path / "store.zarr"
    create_zarrista_group(store_path, {"root": True}, zarr_format)
    dims = {"_ARRAY_DIMENSIONS": ["y", "x"]}
    create_zarrista_subgroup(store_path, "a/b", dims, zarr_format)

    # Ancestor groups gain their own metadata documents.
    if zarr_format == 2:
        assert json.loads((store_path / "a" / ".zgroup").read_text()) == {
            "zarr_format": 2
        }
        leaf_attrs = json.loads((store_path / "a" / "b" / ".zattrs").read_text())
    else:
        ancestor = json.loads((store_path / "a" / "zarr.json").read_text())
        assert ancestor["node_type"] == "group"
        leaf = json.loads((store_path / "a" / "b" / "zarr.json").read_text())
        leaf_attrs = leaf["attributes"]
    assert leaf_attrs == dims

    zgroup = zarr.open_group(str(store_path), mode="r")
    assert zgroup["a/b"].attrs["_ARRAY_DIMENSIONS"] == ["y", "x"]


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="comparison target zarr.consolidate_metadata(zarr_format=...) "
    "requires zarr-python >= 3",
)
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_consolidate_metadata_matches_zarr_python(tmp_path, sample_data, zarr_format):
    pytest.importorskip("zarrista")
    import warnings

    from deepdiff import DeepDiff

    store_path = tmp_path / "ours.zarr"
    create_zarrista_group(store_path, {"root": True}, zarr_format)
    create_zarrista_subgroup(
        store_path, "scale0", {"_ARRAY_DIMENSIONS": ["y", "x"]}, zarr_format
    )
    arr = create_zarrista_array(
        store_path,
        "scale0/image",
        sample_data.shape,
        sample_data.dtype,
        (8, 16),
        zarr_format,
    )
    write_dask_array(da.from_array(sample_data, chunks=(8, 16)), arr)

    reference = tmp_path / "zarr-python.zarr"
    shutil.copytree(store_path, reference)
    consolidate_metadata(store_path, zarr_format)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(str(reference), zarr_format=zarr_format)

    doc_name = ".zmetadata" if zarr_format == 2 else "zarr.json"
    ours = json.loads((store_path / doc_name).read_text())
    theirs = json.loads((reference / doc_name).read_text())
    assert not DeepDiff(theirs, ours, ignore_order=True)


def _assert_stores_identical(zarrista_path, legacy_path):
    """Assert two on-disk stores have identical file sets and metadata docs."""
    from deepdiff import DeepDiff

    zarrista_files = sorted(
        p.relative_to(zarrista_path).as_posix()
        for p in zarrista_path.rglob("*")
        if p.is_file()
    )
    legacy_files = sorted(
        p.relative_to(legacy_path).as_posix()
        for p in legacy_path.rglob("*")
        if p.is_file()
    )
    assert zarrista_files == legacy_files

    metadata_names = {".zgroup", ".zattrs", ".zarray", ".zmetadata", "zarr.json"}
    for name in legacy_files:
        if name.rsplit("/", 1)[-1] not in metadata_names:
            continue
        legacy_doc = json.loads((legacy_path / name).read_text())
        zarrista_doc = json.loads((zarrista_path / name).read_text())
        assert not DeepDiff(legacy_doc, zarrista_doc, ignore_order=True), name


def _assert_engines_equivalent(tmp_path, multiscales, ngff_version, **write_kwargs):
    """Write via both engines and assert the stores are indistinguishable.

    A path-like store dispatches to zarrista; a store-object target keeps the
    legacy zarr-python engine. File sets, parsed metadata documents, and pixel
    values must all agree.
    """
    from ngff_zarr import from_ngff_zarr, to_ngff_zarr

    zarrista_path = tmp_path / "zarrista.ome.zarr"
    to_ngff_zarr(str(zarrista_path), multiscales, version=ngff_version, **write_kwargs)

    legacy_path = tmp_path / "legacy.ome.zarr"
    to_ngff_zarr(
        zarr.storage.LocalStore(legacy_path),
        multiscales,
        version=ngff_version,
        **write_kwargs,
    )

    _assert_stores_identical(zarrista_path, legacy_path)

    zarrista_read = from_ngff_zarr(str(zarrista_path))
    legacy_read = from_ngff_zarr(str(legacy_path))
    for ours, theirs in zip(zarrista_read.images, legacy_read.images):
        np.testing.assert_array_equal(np.asarray(ours.data), np.asarray(theirs.data))


def _sample_multiscales(dtype=np.uint8, shape=(2, 64, 64), chunks=32):
    from ngff_zarr import to_multiscales, to_ngff_image

    rng = np.random.default_rng(11)
    info = np.iinfo(dtype)
    data = rng.integers(info.min, info.max, size=shape, dtype=dtype)
    image = to_ngff_image(
        data, dims=("z", "y", "x"), scale={"z": 2.0, "y": 0.5, "x": 0.5}
    )
    return to_multiscales(image, scale_factors=[2], chunks=chunks)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3 (LocalStore)",
)
@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_to_ngff_zarr_path_store_matches_zarr_python(tmp_path, ngff_version):
    pytest.importorskip("zarrista")
    _assert_engines_equivalent(tmp_path, _sample_multiscales(), ngff_version)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3 (LocalStore)",
)
def test_engine_equivalence_sharded(tmp_path):
    """chunks_per_shard writes: identical shard grids and sharding codecs.

    The 96-px axes make the second scale (48 px) smaller than the 64-px shard,
    exercising the partial-final-shard geometry."""
    pytest.importorskip("zarrista")
    multiscales = _sample_multiscales(shape=(2, 96, 96))
    _assert_engines_equivalent(tmp_path, multiscales, "0.5", chunks_per_shard=2)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3 (LocalStore)",
)
@pytest.mark.parametrize("chunks_per_shard", [None, 2])
def test_engine_equivalence_v3_codec_compressors(tmp_path, chunks_per_shard):
    """A user-supplied zarr v3 codec chain maps to identical stored codecs."""
    pytest.importorskip("zarrista")
    compressors = zarr.codecs.BloscCodec(
        cname="zlib", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
    )
    kwargs = {"compressors": compressors}
    if chunks_per_shard is not None:
        kwargs["chunks_per_shard"] = chunks_per_shard
    _assert_engines_equivalent(tmp_path, _sample_multiscales(), "0.5", **kwargs)


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3 (LocalStore)",
)
@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_engine_equivalence_numcodecs_compressor(tmp_path, ngff_version):
    """A numcodecs compressor object maps through _numcodecs_to_zarr_v3_codec
    (format 3) or verbatim numcodecs config (format 2) on both engines."""
    pytest.importorskip("zarrista")
    numcodecs = pytest.importorskip("numcodecs")
    compressor = numcodecs.Blosc(cname="lz4", clevel=5)
    _assert_engines_equivalent(
        tmp_path, _sample_multiscales(), ngff_version, compressor=compressor
    )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3 (LocalStore)",
)
@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_engine_equivalence_explicit_no_compression(tmp_path, ngff_version):
    """An explicit compressor=None disables compression on both engines."""
    pytest.importorskip("zarrista")
    _assert_engines_equivalent(
        tmp_path, _sample_multiscales(), ngff_version, compressor=None
    )


def test_path_store_rejects_filters(tmp_path):
    """The zarrista engine cannot write zarr filters; it must refuse loudly
    rather than silently drop them from the output."""
    pytest.importorskip("zarrista")
    numcodecs = pytest.importorskip("numcodecs")
    from ngff_zarr import to_ngff_zarr

    with pytest.raises(ValueError, match="filters are not supported"):
        to_ngff_zarr(
            str(tmp_path / "filtered.ome.zarr"),
            _sample_multiscales(),
            version="0.4",
            filters=[numcodecs.Delta(dtype="uint8")],
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3 (LocalStore)",
)
@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
@pytest.mark.parametrize("chunks_per_shard", [None, 2])
def test_engine_equivalence_memory_constrained(
    tmp_path, ngff_version, chunks_per_shard
):
    """Slab (region) writes under a small memory target produce the same
    chunk grids and payload as the legacy engine."""
    pytest.importorskip("zarrista")
    from ngff_zarr import config

    if ngff_version == "0.4" and chunks_per_shard is not None:
        pytest.skip("sharding requires OME-Zarr >= 0.5")
    multiscales = _sample_multiscales(dtype=np.uint16, shape=(8, 128, 128))
    kwargs = {}
    if chunks_per_shard is not None:
        kwargs["chunks_per_shard"] = chunks_per_shard
    saved_target = config.memory_target
    try:
        config.memory_target = 64 * 1024
        _assert_engines_equivalent(tmp_path, multiscales, ngff_version, **kwargs)
    finally:
        config.memory_target = saved_target


def _to_multiscales_module():
    # ``ngff_zarr.to_multiscales`` names the function; fetch the module.
    import importlib

    return importlib.import_module("ngff_zarr.to_multiscales")


def _fail_open_array(*args, **kwargs):
    raise AssertionError("cache write used the legacy zarr-python engine")


@pytest.mark.parametrize(
    ("dims", "shape", "memory_target"),
    [
        (("z", "y", "x"), (7, 64, 64), 8192),
        (("z", "y", "x"), (600, 32, 32), 8192),
        (("y", "x"), (250, 96), 4096),
        (("x",), (1000,), 256),
    ],
    ids=["z-slabs", "z-optimized-rechunk", "2d-strips", "1d-segments"],
)
def test_multiscales_cache_uses_zarrista(
    tmp_path, monkeypatch, dims, shape, memory_target
):
    """The to_multiscales disk cache writes through zarrista for a
    directory-backed cache store, across all cache layouts: z-slabs (with a
    partial final slab), the optimized-chunks second pass, 2D strips, and 1D
    segments. Pixel values must round-trip exactly."""
    pytest.importorskip("zarrista")
    import dask.array
    from ngff_zarr import config, to_multiscales, to_ngff_image

    tm = _to_multiscales_module()
    monkeypatch.setattr(tm, "open_array", _fail_open_array)
    monkeypatch.setattr(config, "cache_store", zarr.storage.LocalStore(tmp_path))
    monkeypatch.setattr(config, "memory_target", memory_target)

    rng = np.random.default_rng(7)
    data = rng.integers(0, 255, size=shape, dtype=np.uint8)
    image = to_ngff_image(dask.array.from_array(data, chunks=32), dims=dims)
    multiscales = to_multiscales(image, scale_factors=[], cache=True)

    cache_dirs = list(tmp_path.glob("*-cache-*"))
    assert cache_dirs, "cache was not written under the resolved directory path"
    np.testing.assert_array_equal(np.asarray(multiscales.images[0].data), data)


def test_multiscales_cache_store_object_keeps_legacy_engine(monkeypatch):
    """A user-supplied cache store without a local directory (here a
    MemoryStore) keeps the zarr-python cache engine."""
    import dask.array
    from ngff_zarr import config, to_multiscales, to_ngff_image

    tm = _to_multiscales_module()
    legacy_calls = []
    real_open_array = tm.open_array

    def spy_open_array(*args, **kwargs):
        legacy_calls.append(kwargs.get("path"))
        return real_open_array(*args, **kwargs)

    monkeypatch.setattr(tm, "open_array", spy_open_array)
    monkeypatch.setattr(config, "cache_store", zarr.storage.MemoryStore())
    monkeypatch.setattr(config, "memory_target", 4096)

    rng = np.random.default_rng(7)
    data = rng.integers(0, 255, size=(256, 96), dtype=np.uint8)
    image = to_ngff_image(dask.array.from_array(data, chunks=32), dims=("y", "x"))
    multiscales = to_multiscales(image, scale_factors=[], cache=True)

    assert legacy_calls, "MemoryStore cache should use the legacy engine"
    np.testing.assert_array_equal(np.asarray(multiscales.images[0].data), data)

    # Run the cache cleanup now, while a rmdir stub is in place: zarr-python 3
    # dropped ``zarr.storage.rmdir``, which the (pre-existing) store-object
    # cleanup fallback still calls; firing the computed callbacks here marks
    # the cache removed so the atexit hook does not spew at interpreter exit.
    monkeypatch.setattr(zarr.storage, "rmdir", lambda *args: None, raising=False)
    for image in multiscales.images:
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []


def test_resolve_store_path(tmp_path):
    """LocalStore/DirectoryStore and path-likes resolve to their directory;
    stores without one resolve to None."""
    from ngff_zarr._zarrista_utils import resolve_store_path

    assert resolve_store_path(str(tmp_path)) == tmp_path
    assert resolve_store_path(tmp_path) == tmp_path
    assert resolve_store_path({}) is None
    if hasattr(zarr.storage, "LocalStore"):
        store = zarr.storage.LocalStore(tmp_path)
        assert resolve_store_path(store) == tmp_path
    if hasattr(zarr.storage, "DirectoryStore"):
        store = zarr.storage.DirectoryStore(str(tmp_path))
        assert resolve_store_path(store) == tmp_path
    assert resolve_store_path(zarr.storage.MemoryStore()) is None


def test_read_group_attributes(tmp_path):
    """Absent group -> None, empty group -> {}, attributes round-trip; an
    array node raises like zarr-python's group-over-array refusal."""
    pytest.importorskip("zarrista")
    from ngff_zarr._zarrista_utils import read_group_attributes

    for zarr_format in (2, 3):
        store_path = tmp_path / f"v{zarr_format}.zarr"
        assert read_group_attributes(store_path, zarr_format=zarr_format) is None
        create_zarrista_group(store_path, {"root": True}, zarr_format)
        assert read_group_attributes(store_path, zarr_format=zarr_format) == {
            "root": True
        }
        assert (
            read_group_attributes(store_path, "absent", zarr_format=zarr_format) is None
        )
        create_zarrista_subgroup(store_path, "a/b", None, zarr_format)
        assert read_group_attributes(store_path, "a/b", zarr_format=zarr_format) == {}
        create_zarrista_array(store_path, "img", (4, 4), np.uint8, (2, 2), zarr_format)
        with pytest.raises(ValueError, match="array node"):
            read_group_attributes(store_path, "img", zarr_format=zarr_format)


def _write_sample_hcs_plate(store, ngff_version, multiscales):
    """Write a small plate: two wells in one row, two fields in well A/1."""
    from ngff_zarr.hcs import HCSPlate, to_hcs_zarr, write_hcs_well_image
    from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell

    plate_metadata = Plate(
        columns=[PlateColumn(name="1"), PlateColumn(name="2")],
        rows=[PlateRow(name="A")],
        wells=[
            PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        ],
        name="Engine Equivalence Plate",
        field_count=2,
        version=ngff_version,
    )
    to_hcs_zarr(HCSPlate(None, plate_metadata), store)
    for row_name, column_name, field_index in (
        ("A", "1", 0),
        ("A", "1", 1),  # second field exercises the existing-well attrs merge
        ("A", "2", 0),
    ):
        write_hcs_well_image(
            store=store,
            multiscales=multiscales,
            plate_metadata=plate_metadata,
            row_name=row_name,
            column_name=column_name,
            field_index=field_index,
            version=ngff_version,
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b2"),
    reason="the legacy-engine comparison requires zarr-python >= 3",
)
@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_hcs_write_path_store_matches_zarr_python(tmp_path, monkeypatch, ngff_version):
    """HCS plate writes via zarrista match the legacy engine's store exactly.

    Both plates target path stores through the identical hcs.py code path;
    the reference run forces the backend-dispatch helper off so the legacy
    zarr-python engine writes it.
    """
    pytest.importorskip("zarrista")
    import importlib

    import ngff_zarr.hcs as hcs_module
    from ngff_zarr.hcs import from_hcs_zarr

    # The package re-export shadows the module name, so importlib is needed.
    to_ngff_zarr_module = importlib.import_module("ngff_zarr.to_ngff_zarr")

    multiscales = _sample_multiscales(shape=(2, 32, 32), chunks=16)
    zarrista_path = tmp_path / "zarrista.ome.zarr"
    _write_sample_hcs_plate(str(zarrista_path), ngff_version, multiscales)

    monkeypatch.setattr(hcs_module, "use_zarrista_for", lambda _store: False)
    monkeypatch.setattr(to_ngff_zarr_module, "use_zarrista_for", lambda _store: False)
    legacy_path = tmp_path / "legacy.ome.zarr"
    _write_sample_hcs_plate(str(legacy_path), ngff_version, multiscales)

    _assert_stores_identical(zarrista_path, legacy_path)

    plate = from_hcs_zarr(str(zarrista_path))
    well = plate.get_well("A", "1")
    assert [img.path for img in well.images] == ["0", "1"]
    image = well.get_image(1)
    np.testing.assert_array_equal(
        np.asarray(image.images[0].data), np.asarray(multiscales.images[0].data)
    )


def test_hcs_plate_writer_ozx_uses_path_engine(tmp_path):
    """The .ozx flow writes its temp store through the path (zarrista) engine:
    no hcs.py write may fall back to zarr-python group APIs."""
    pytest.importorskip("zarrista")
    import zipfile
    from unittest.mock import patch

    from ngff_zarr.hcs import HCSPlateWriter
    from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell

    plate_metadata = Plate(
        columns=[PlateColumn(name="1")],
        rows=[PlateRow(name="A")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        name="OZX Plate",
        field_count=1,
        version="0.5",
    )
    multiscales = _sample_multiscales(shape=(2, 32, 32), chunks=16)
    ozx_path = tmp_path / "plate.ozx"

    def _fail_open_group(*args, **kwargs):
        raise AssertionError("legacy zarr.open_group used for a path store")

    with patch("ngff_zarr.hcs.zarr.open_group", side_effect=_fail_open_group):
        with HCSPlateWriter(str(ozx_path), plate_metadata) as writer:
            writer.write_well_image(
                multiscales=multiscales,
                row_name="A",
                column_name="1",
                field_index=0,
            )

    with zipfile.ZipFile(ozx_path) as zf:
        names = zf.namelist()
    assert names[0] == "zarr.json"  # RFC-9: root metadata first
    assert "A/1/zarr.json" in names
    assert "A/1/0/zarr.json" in names
