# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The Pyodide bridge's pure-Python parts; the JS half runs in test/pyodide."""

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from ngff_zarr import _js_zarr
from ngff_zarr._js_zarr import (
    TYPED_ARRAYS,
    JsZarrStore,
    _block_on,
    _chunk_layout,
    _PyStore,
)
from ngff_zarr._zarrista_utils import _array_doc, create_zarrista_array
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


def test_chunk_layout_reads_the_inner_chunks_of_a_sharded_array(tmp_path):
    create_zarrista_array(tmp_path, "s", SHAPE, "uint8", CHUNKS, 3, chunks_per_shard=2)
    create_zarrista_array(tmp_path, "p", SHAPE, "uint8", CHUNKS, 3)
    create_zarrista_array(tmp_path, "v2", SHAPE, "uint8", CHUNKS, 2)
    store = JsZarrStore(tmp_path)

    assert _chunk_layout(store, "s") == (CHUNKS, SHAPE)
    assert _chunk_layout(store, "p") == (CHUNKS, CHUNKS)
    assert _chunk_layout(store, "v2") == (CHUNKS, CHUNKS)
    assert _chunk_layout(store.with_prefix("s"), None) == (CHUNKS, SHAPE)


def test_every_zarr_v3_dtype_has_a_typed_array():
    for name in ("bool", "int8", "uint16", "int64", "float16", "float64"):
        assert _numpy_to_zarr_dtype(np.dtype(name)) in TYPED_ARRAYS
    assert TYPED_ARRAYS["int64"] == "BigInt64Array"
    assert TYPED_ARRAYS["bool"] == "Uint8Array"
