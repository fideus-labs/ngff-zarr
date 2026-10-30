# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The Pyodide bridge's pure-Python parts; the JS half runs in test/pyodide."""

import asyncio
import hashlib
import importlib
import inspect
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import (
    Methods,
    PlateWell,
    Well,
    WellImage,
    _itkwasm_pipelines,
    _js_zarr,
    _zarrista_utils,
    config,
    hcs,
    to_multiscales,
)
from ngff_zarr._js_zarr import (
    TYPED_ARRAYS,
    JsArrayAdapter,
    JsZarrStore,
    _array_layout,
    _block_on,
    _PyStore,
)
from ngff_zarr._zarrista_utils import (
    LocalZarrGroup,
    _array_doc,
    create_zarrista_array,
    create_zarrista_subgroup,
    open_array,
    open_lazy_array,
    open_local_node,
    open_zarrista_array,
    open_zarrista_lazy,
    write_dask_array,
)
from ngff_zarr.rfc9_zip import ZipReadStore
from ngff_zarr.to_ngff_zarr import _numpy_to_zarr_dtype

pytest.importorskip("zarrista")

SHAPE = (4, 6, 8)
CHUNKS = (2, 3, 4)
DOC_CASES = {
    "u16_blosc": {
        "dtype": "uint16",
        "zarr_format": 3,
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1},
    },
    "u8_blosc_sharded_dims": {
        "dtype": "uint8",
        "zarr_format": 3,
        "compressor": {"id": "blosc"},
        "chunks_per_shard": 2,
        "dimension_names": ("z", "y", "x"),
    },
    "u8_shards": {"dtype": "uint8", "zarr_format": 3, "shards": (4, 6, 8)},
    # A shard equal to the chunk is still sharded (the writer collapses shards
    # onto small pyramid levels).
    "u8_one_chunk_shards": {"dtype": "uint8", "zarr_format": 3, "chunks_per_shard": 1},
    "f32_zstd_fill": {
        "dtype": "float32",
        "zarr_format": 3,
        "compressor": {"id": "zstd"},
        "fill_value": 1.5,
    },
    "f32_nan_fill": {"dtype": "float32", "zarr_format": 3, "fill_value": np.nan},
    "f32_nan_fill_v2": {"dtype": "float32", "zarr_format": 2, "fill_value": np.nan},
    "i64_gzip": {"dtype": "int64", "zarr_format": 3, "compressor": {"id": "gzip"}},
    "u16_chain": {
        "dtype": "uint16",
        "zarr_format": 3,
        "compressors": [{"id": "gzip", "level": 2}],
    },
    "bool_none": {"dtype": "bool", "zarr_format": 3},
    "u8_cke_attrs": {
        "dtype": "uint8",
        "zarr_format": 3,
        "attributes": {"k": [1, 2]},
        "chunk_key_encoding": {"name": "v2", "configuration": {"separator": "/"}},
    },
    "u16_v2_blosc": {
        "dtype": "uint16",
        "zarr_format": 2,
        "compressor": {"id": "blosc"},
    },
    "u16_v2_none_sep": {
        "dtype": "uint16",
        "zarr_format": 2,
        "dimension_separator": ".",
    },
}


@pytest.mark.parametrize("case", DOC_CASES.values(), ids=list(DOC_CASES))
def test_array_doc_is_what_zarrista_writes(tmp_path, case):
    create_zarrista_array(tmp_path, "a", SHAPE, chunks=CHUNKS, **case)
    doc_name = "zarr.json" if case["zarr_format"] == 3 else ".zarray"
    written = json.loads((tmp_path / "a" / doc_name).read_text())

    assert _array_doc(SHAPE, chunks=CHUNKS, **case) == written


BAD_DOC_ARGS = {
    "chunks_rank": ({"chunks": (2, 3)}, "chunks must have 3 dimensions"),
    "v2_sharded": ({"zarr_format": 2, "chunks_per_shard": 2}, "only supported for"),
    "v2_dims": ({"zarr_format": 2, "dimension_names": "zyx"}, "requires zarr format 3"),
    "v2_chain": ({"zarr_format": 2, "compressors": []}, "requires zarr format 3"),
    "v2_cke": ({"zarr_format": 2, "chunk_key_encoding": {}}, "requires zarr format 3"),
    "format": ({"zarr_format": 4}, "Unsupported zarr format"),
    "both_shards": ({"shards": SHAPE, "chunks_per_shard": 1}, "not both"),
    "shards_rank": ({"shards": (4, 6)}, "shards must have 3 dimensions"),
    "cps_rank": ({"chunks_per_shard": (1, 1)}, "chunks_per_shard must have 3"),
    "codec": ({"compressor": {"id": "lzma"}}, "not supported by the zarrista"),
}


@pytest.mark.parametrize("args", BAD_DOC_ARGS.values(), ids=list(BAD_DOC_ARGS))
def test_array_doc_rejects_bad_arguments(args):
    kwargs, message = args
    kwargs = {"dtype": "uint8", "zarr_format": 3, "chunks": CHUNKS, **kwargs}

    with pytest.raises(ValueError, match=message):
        _array_doc(SHAPE, **kwargs)


def test_array_doc_broadcasts_an_int_chunk():
    doc = _array_doc(SHAPE, "uint8", 2, 3)

    assert doc["chunk_grid"]["configuration"]["chunk_shape"] == [2, 2, 2]


def test_py_store_js_face(tmp_path, monkeypatch):
    # Uint8Array needs Pyodide; the JS face is the dispatch around read/write.
    monkeypatch.setattr(_js_zarr, "_uint8array", bytes)
    store = _PyStore(tmp_path)
    store.set("/k", SimpleNamespace(to_bytes=lambda: b"abcdef"))

    assert store.get("/k") == b"abcdef"
    assert store.get("/missing") is None
    assert store.getRange("/k", SimpleNamespace(offset=1, length=2)) == b"bc"
    assert store.getRange("/k", SimpleNamespace(suffixLength=2)) == b"ef"
    assert store.getRange("/missing", SimpleNamespace(suffixLength=2)) is None


@pytest.mark.parametrize("kind", ["path", "mapping"])
def test_py_store_reads_ranges_and_writes(tmp_path, kind):
    store = _PyStore(tmp_path if kind == "path" else {})
    assert store.read("/a/b") is None

    store.write("/a/b", b"0123456789")

    assert store.read("a/b") == b"0123456789"
    assert store.read("/a/b", offset=2, length=3) == b"234"
    assert store.read("/a/b", suffix=4) == b"6789"
    assert store.read("/a/b", suffix=0) == b""


def test_js_zarr_store_mapping_face(tmp_path):
    (tmp_path / "zarr.json").write_text("{}")
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "zarr.json").write_text('{"a": 1}')
    store = JsZarrStore(tmp_path)

    assert json.loads(store["zarr.json"]) == {}
    assert "0/zarr.json" in store
    assert "1/zarr.json" not in store
    assert set(store) == {"zarr.json", "0/zarr.json"}
    assert store.node_path() == "/"

    view = store.with_prefix("0")

    assert json.loads(view["zarr.json"]) == {"a": 1}
    assert set(view) == {"zarr.json"}
    assert view.node_path("c/0") == "/0/c/0"
    with pytest.raises(KeyError):
        view["missing"]
    assert list(JsZarrStore("https://example.org/a.zarr")) == []  # cannot list

    store.close()  # no proxy yet: nothing to destroy
    store._handle["js"] = proxy = Mock()
    assert view.js is proxy
    view.close()

    proxy.destroy.assert_called_once_with()
    assert store._handle["js"] is None


def test_block_on_refuses_without_jspi():
    coroutine = asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="runPythonAsync"):
        _block_on(coroutine)

    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_array_layout_reads_inner_chunks_and_attributes(tmp_path):
    attrs = {"k": [1, 2]}
    create_zarrista_array(tmp_path, "s", SHAPE, "uint8", CHUNKS, 3, chunks_per_shard=2)
    create_zarrista_array(tmp_path, "p", SHAPE, "uint8", CHUNKS, 3, attributes=attrs)
    create_zarrista_array(tmp_path, "v2", SHAPE, "uint8", CHUNKS, 2, attributes=attrs)
    store = JsZarrStore(tmp_path)

    assert _array_layout(store, "s") == (CHUNKS, SHAPE, {})
    assert _array_layout(store, "p") == (CHUNKS, CHUNKS, attrs)
    assert _array_layout(store, "v2") == (CHUNKS, CHUNKS, attrs)
    assert _array_layout(store.with_prefix("s"), None) == (CHUNKS, SHAPE, {})


def test_js_zarr_store_reads_http_urls_only():
    with pytest.raises(ValueError, match="HTTP"):
        JsZarrStore("s3://bucket/a.zarr")


@pytest.fixture
def use_js(monkeypatch):
    monkeypatch.setattr(_zarrista_utils, "_USE_JS", True)


def test_group_creation_needs_no_zarrista_on_pyodide(tmp_path, use_js, monkeypatch):
    monkeypatch.setitem(sys.modules, "zarrista", None)  # import would fail

    group = create_zarrista_subgroup(tmp_path, "a/b", {"k": 1}, 3)

    assert isinstance(group, LocalZarrGroup)
    assert group.attrs.asdict() == {"k": 1}
    assert (
        json.loads((tmp_path / "a" / "zarr.json").read_text())["node_type"] == "group"
    )


def test_write_dask_array_writes_whole_shards_on_pyodide(use_js):
    class Target(JsArrayAdapter):
        def __init__(self):
            self.shape, self.ndim, self.dtype = SHAPE, 3, np.dtype("uint8")
            self.write_chunks, self.writes = (4, 3, 8), []

        def __setitem__(self, selection, value):
            self.writes.append((selection, value.shape))

    target = Target()
    write_dask_array(da.ones(SHAPE, dtype="uint8", chunks=CHUNKS), target)

    assert sorted(shape for _, shape in target.writes) == [(4, 3, 8), (4, 3, 8)]


class _Opened(JsArrayAdapter):
    """What the stubbed bridge hands back: enough for the seams' return paths."""

    def __init__(self):
        self.shape, self.ndim, self.dtype = SHAPE, 3, np.dtype("uint8")
        self.chunks = self.write_chunks = CHUNKS

    def to_dask(self):
        return "lazy"


@pytest.fixture
def js_opens(use_js, monkeypatch):
    """Record every array open the seams route to the bridge."""
    opened = []

    def open_js_array(store, component=None, store_path=None, node_path=None):
        opened.append((store.source, store.prefix, component, store_path, node_path))
        if component == "missing":
            raise KeyError(component)
        return _Opened()

    monkeypatch.setattr(_js_zarr, "open_js_array", open_js_array)
    return opened


def test_create_array_passes_every_argument_to_the_bridge_on_pyodide(
    tmp_path, use_js, monkeypatch
):
    calls = []
    signature = inspect.signature(_js_zarr.create_js_array)
    monkeypatch.setattr(_js_zarr, "create_js_array", lambda *a: calls.append(a))
    kwargs = {
        "compressor": {"id": "gzip"},
        "chunks_per_shard": 2,
        "dimension_names": ("z", "y", "x"),
        "fill_value": 1,
        "attributes": {"k": 1},
        "chunk_key_encoding": {"name": "v2", "configuration": {"separator": "."}},
    }

    create_zarrista_array(tmp_path, "a", SHAPE, "uint8", CHUNKS, 3, **kwargs)

    bound = signature.bind(*calls[0]).arguments
    assert bound == {
        "path": tmp_path,
        "name": "a",
        "shape": SHAPE,
        "dtype": "uint8",
        "chunks": CHUNKS,
        "zarr_format": 3,
        "dimension_separator": "/",
        "compressors": None,
        "shards": None,
        **kwargs,
    }


def test_local_opens_route_through_the_bridge_on_pyodide(tmp_path, js_opens):
    adapter = open_zarrista_array(tmp_path, "0")
    assert isinstance(adapter, _Opened)
    assert open_zarrista_lazy(tmp_path, "0") == "lazy"
    assert open_lazy_array(JsZarrStore(tmp_path).with_prefix("w"), "0") == "lazy"
    # Creation writes the doc then opens it; a local handle opens it again
    # and reads its chunk grid from the adapter.
    create_zarrista_array(tmp_path, "a", SHAPE, "uint8", CHUNKS, 3)
    node = open_local_node(tmp_path, "a", zarr_format=3)
    assert node.chunks == CHUNKS
    assert node.to_dask() == "lazy"
    # .ozx archives go through the bridge over the zip mapping.
    archive = tmp_path / "img.ozx"
    zipfile.ZipFile(archive, "w").close()
    assert open_lazy_array(ZipReadStore(archive), "0") == "lazy"

    sources = [
        (src if isinstance(src, Path) else type(src), p, c)
        for src, p, c, *_ in js_opens
    ]
    assert sources == [
        (tmp_path, "", "0"),
        (tmp_path, "", "0"),
        (tmp_path, "w", "0"),
        (tmp_path, "", "a"),  # create
        (tmp_path, "", "a"),  # .chunks
        (tmp_path, "", "a"),  # .to_dask()
        (ZipReadStore, "", "0"),
    ]
    assert js_opens[0][3:] == (tmp_path, "0")  # store_path/node_path for the writer


def test_open_array_on_a_url_returns_the_adapter_on_pyodide(js_opens):
    assert isinstance(open_array("https://h/a.zarr", "0"), _Opened)
    assert js_opens == [("https://h/a.zarr", "", "0", None, None)]
    with pytest.warns(UserWarning, match="storage_options are ignored"):
        open_array("https://h/a.zarr", "0", storage_options={"headers": {}})
    with pytest.raises(FileNotFoundError, match="missing"):
        open_array("https://h/a.zarr", "missing")


def test_from_ome_zarr_routes_urls_to_the_js_store_on_pyodide(use_js, monkeypatch):
    from ngff_zarr import from_ome_zarr

    # The package re-exports the function under the module's name.
    reader = importlib.import_module("ngff_zarr.from_ngff_zarr")
    monkeypatch.setattr(reader, "_USE_JS", True)
    asked = []

    def getitem(self, key):
        asked.append((self.source, key))
        raise KeyError(key)

    monkeypatch.setattr(JsZarrStore, "__getitem__", getitem)

    with pytest.raises(ValueError, match="group"):
        from_ome_zarr("https://example.org/a.zarr")

    assert ("https://example.org/a.zarr", "zarr.json") in asked


def test_hcs_plates_read_urls_through_the_js_store_on_pyodide(use_js, monkeypatch):
    monkeypatch.setattr(hcs, "_USE_JS", True)
    asked = []

    def getitem(self, key):
        asked.append((self.source, key))
        raise KeyError(key)

    monkeypatch.setattr(JsZarrStore, "__getitem__", getitem)

    with pytest.raises(ValueError, match="group"):
        hcs.from_hcs_zarr("https://example.org/plate.zarr")

    assert ("https://example.org/plate.zarr", "zarr.json") in asked


def test_hcs_wells_open_fields_as_prefixed_js_store_views(monkeypatch):
    opened = []
    monkeypatch.setattr(
        hcs, "from_ome_zarr", lambda store: opened.append(store) or store
    )
    well = hcs.HCSWell(
        JsZarrStore("https://example.org/plate.zarr"),
        "A/1",
        PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        Well(images=[WellImage(path="0")]),
    )

    assert well.get_image(0) is opened[0]

    assert (opened[0].source, opened[0].prefix) == (
        "https://example.org/plate.zarr",
        "A/1/0",
    )


def test_every_zarr_v3_dtype_has_a_typed_array():
    for name in ("bool", "int8", "uint16", "int64", "float16", "float64"):
        assert _numpy_to_zarr_dtype(np.dtype(name)) in TYPED_ARRAYS
    assert TYPED_ARRAYS["int64"] == "BigInt64Array"
    assert TYPED_ARRAYS["bool"] == "Uint8Array"


@pytest.mark.parametrize(
    ("method", "twins"),
    [
        (Methods.ITKWASM_GAUSSIAN, {"gaussian_kernel_radius", "downsample"}),
        (Methods.ITKWASM_BIN_SHRINK, {"downsample_bin_shrink"}),
        (
            Methods.ITKWASM_LABEL_IMAGE,
            {"gaussian_kernel_radius", "downsample_label_image"},
        ),
    ],
)
def test_itkwasm_downsampling_awaits_the_async_twins_on_pyodide(
    method, twins, monkeypatch
):
    import itkwasm_downsample

    volume = np.arange(16**3, dtype=np.uint16).reshape(16, 16, 16) % 7
    native = [
        image.data.compute()
        for image in to_multiscales(volume, [2], method=method, chunks=8).images
    ]
    awaited = set()

    def twin(name):
        async def call(*args, **kwargs):
            awaited.add(name)
            return getattr(itkwasm_downsample, name)(*args, **kwargs)

        return call

    for name in twins:
        monkeypatch.setattr(itkwasm_downsample, f"{name}_async", twin(name))
    monkeypatch.setattr(_itkwasm_pipelines, "_USE_JS", True)
    monkeypatch.setattr(_js_zarr, "_block_on", asyncio.run)

    images = to_multiscales(volume, [2], method=method, chunks=8).images

    for image, expected in zip(images, native):
        np.testing.assert_array_equal(image.data.compute(), expected)
    assert awaited == twins


#: sha256 prefixes of the uint16 pyramid below, computed with the WASI build of
#: itkwasm-downsample 2.0.2; test/pyodide/test_pyodide.py checks the same
#: values against the Emscripten build.
PYRAMID_DIGESTS = ["257df0ff4ff55204", "7bd266d79dec4343", "fd60f9720d24832c"]


def test_itkwasm_pyramid_matches_the_pyodide_reference():
    volume = np.random.default_rng(0).integers(0, 4096, (64, 64, 64), dtype=np.uint16)

    images = to_multiscales(volume, scale_factors=[2, 4], chunks=32).images

    digests = [
        hashlib.sha256(i.data.compute().tobytes()).hexdigest()[:16] for i in images
    ]
    assert digests == PYRAMID_DIGESTS


def test_js_package_url_pins_the_release_the_harness_tests():
    """docs/pyodide.md: the esm.sh default is the release test/pyodide ran."""
    harness = json.loads((Path(__file__).parent / "pyodide/package.json").read_text())
    version = harness["dependencies"]["@fideus-labs/ngff-zarr"]

    assert f"/@fideus-labs/ngff-zarr@{version}/" in config.js_package_url
