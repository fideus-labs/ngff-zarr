# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Zarr array I/O through the ``@fideus-labs/ngff-zarr`` JS package, for Pyodide.

Pyodide ships no zarrista, so on Emscripten chunk encoding, decoding and
sharding run in JavaScript: zarrita opens the arrays and ngff-zarr's
TypeScript ``zarrGet``/``zarrSet`` move the blocks. Python keeps composing
and parsing every JSON document; JS only ever sees a store.

Blocking on a JS promise uses ``pyodide.ffi.run_sync`` (JSPI), which works
only when Python was entered through a promising entry point:
``runPythonAsync``, an ``async`` Python function or
``PyProxy.callPromising()``. The store callbacks zarrita invokes from JS
(:class:`_PyStore`) therefore never block on a promise themselves.

The JS package is taken from the host-registered module ``ngff_zarr_js``
(``pyodide.registerJsModule("ngff_zarr_js", {...ngffZarr, zarr})``) when
present, else imported from ``config.js_package_url`` and the pinned zarrita
URL, which browsers can do and Node cannot.
"""

import copy
import json
import posixpath
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import dask.array
import numpy as np

from ._zarrista_utils import (
    _array_doc,
    _native_contiguous,
    _normalize_selection,
    _ZarristaArrayAdapter,
    normalize_store,
)
from .config import config

#: zarrita release imported beside ``config.js_package_url`` (same pin as the
#: JS package's own dependency, so both resolve to one module instance).
ZARRITA_URL = "https://esm.sh/zarrita@0.7.5"

#: JS typed array per numpy dtype name; bool travels as its bytes.
TYPED_ARRAYS = {
    "bool": "Uint8Array",
    "int8": "Int8Array",
    "uint8": "Uint8Array",
    "int16": "Int16Array",
    "uint16": "Uint16Array",
    "int32": "Int32Array",
    "uint32": "Uint32Array",
    "int64": "BigInt64Array",
    "uint64": "BigUint64Array",
    "float16": "Float16Array",
    "float32": "Float32Array",
    "float64": "Float64Array",
}

_namespace = None


def _block_on(awaitable):
    """Wait synchronously for JS promise *awaitable* (JSPI stack switching)."""
    try:
        from pyodide import ffi
    except ImportError:
        ffi = None
    if ffi is None or not ffi.can_run_sync():
        if hasattr(awaitable, "close"):
            awaitable.close()
        msg = (
            "ngff-zarr on Pyodide waits for JavaScript promises with JSPI "
            "(pyodide.ffi.run_sync). That needs Chrome/Edge 137+, Firefox 153+, "
            "Safari 27+ or Node 25+, and Python entered through a promising "
            "entry point: runPythonAsync, an async function, or "
            "PyProxy.callPromising()."
        )
        raise RuntimeError(msg)
    try:
        return ffi.run_sync(awaitable)
    except ffi.JsException as error:
        # JS error classes do not cross the boundary; the readers treat store
        # failures as I/O errors (see resolve_matrix_transforms). An
        # AggregateError's causes are the useful part of its message.
        causes = getattr(error.js_error, "errors", None)
        detail = "".join(f"\n  {cause.message}" for cause in causes) if causes else ""
        raise OSError(f"{error}{detail}") from error


def _js():
    """The JS namespace: ``zarrGet``, ``zarrSet`` and ``zarr`` (zarrita)."""
    global _namespace
    if _namespace is None:
        try:
            import ngff_zarr_js as _namespace
        except ImportError:
            from pyodide.code import run_js

            ngff = _block_on(run_js(f"import({config.js_package_url!r})"))
            _namespace = SimpleNamespace(
                zarrGet=ngff.zarrGet,
                zarrSet=ngff.zarrSet,
                zarr=_block_on(run_js(f"import({ZARRITA_URL!r})")),
            )
    return _namespace


def _to_js(obj):
    """*obj* as a JS value; dicts become plain objects rather than Maps."""
    import js
    from pyodide.ffi import to_js

    return to_js(obj, dict_converter=js.Object.fromEntries)


def _uint8array(data: bytes):
    import js

    view = js.Uint8Array.new(len(data))
    view.assign(data)
    return view


def _c_strides(shape) -> list[int]:
    strides = []
    stride = 1
    for n in reversed(shape):
        strides.append(stride)
        stride *= n
    return strides[::-1]


def _to_chunk(block: np.ndarray):
    """A zarrita chunk ``{data, shape, stride}`` holding a copy of *block*."""
    import js

    data = getattr(js, TYPED_ARRAYS[block.dtype.name]).new(block.size)
    data.assign(block)
    if block.dtype.kind == "b":
        data = _js().zarr.BoolArray.new(data.buffer)
    return _to_js(
        {"data": data, "shape": list(block.shape), "stride": _c_strides(block.shape)}
    )


def _from_chunk(chunk, dtype: np.dtype) -> np.ndarray:
    """A numpy array (one copy) from zarrita chunk ``{data, shape, stride}``."""
    import js

    shape = tuple(int(n) for n in chunk.shape)
    stride = tuple(int(n) * dtype.itemsize for n in chunk.stride)
    data = chunk.data
    if dtype.kind == "b":
        # BoolArray is not a typed array; view its backing bytes as one.
        data = js.Uint8Array.new(data.buffer, data.byteOffset, data.byteLength)
    flat = np.empty(int(np.prod(shape)), dtype=dtype)
    data.assign_to(flat)
    out = np.ndarray(shape, dtype=dtype, buffer=flat, strides=stride)
    return out if out.flags.c_contiguous else np.ascontiguousarray(out)


class _PyStore:
    """zarrita store over a directory (Emscripten FS) or a key-to-bytes mapping.

    The JS-facing ``get``/``getRange``/``set`` take zarrita's absolute keys
    (``"/0/zarr.json"``) and return ``Uint8Array`` or ``None`` (``undefined``
    in JS) for a missing key, so zarrita sees a synchronous store. The
    pure-Python ``read``/``write`` do the work.
    """

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            self.root: Path | None = Path(source)
            self.mapping = None
        else:
            self.root = None
            self.mapping = source

    @staticmethod
    def _key(key) -> str:
        return str(key).strip("/")

    def read(
        self,
        key,
        offset: int = 0,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes | None:
        """Bytes at *key*, or the requested range of them; ``None`` when absent."""
        key = self._key(key)
        if self.root is not None:
            path = self.root / key
            if not path.is_file():
                return None
            with path.open("rb") as handle:
                if suffix is not None:
                    handle.seek(max(path.stat().st_size - suffix, 0))
                else:
                    handle.seek(offset)
                return handle.read(-1 if length is None else length)
        try:
            data = self.mapping[key]
        except KeyError:
            return None
        if suffix is not None:
            return data[max(len(data) - suffix, 0) :]
        return data[offset:] if length is None else data[offset : offset + length]

    def write(self, key, data: bytes) -> None:
        key = self._key(key)
        if self.root is not None:
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        else:
            self.mapping[key] = data

    def get(self, key, *_options):
        data = self.read(key)
        return None if data is None else _uint8array(data)

    def getRange(self, key, query, *_options):
        suffix = getattr(query, "suffixLength", None)
        if suffix is None:
            data = self.read(key, offset=int(query.offset), length=int(query.length))
        else:
            data = self.read(key, suffix=int(suffix))
        return None if data is None else _uint8array(data)

    def set(self, key, value):
        self.write(key, value.to_bytes())


class JsZarrStore(Mapping):
    """A zarrita store with the ``Mapping`` face the pure-Python readers navigate.

    *source* is an ``http(s)`` URL (zarrita ``FetchStore``: fetch and decode
    stay in JS), a directory path, or a key-to-bytes mapping (both served to
    zarrita by :class:`_PyStore`). *prefix* scopes keys and node paths to a
    subtree, as the HCS well and field views need. Keys are relative to the
    prefix like every other store mapping here, and a missing key raises
    ``KeyError``.
    """

    def __init__(self, source, prefix: str = ""):
        self.source = source
        self.prefix = "/".join(
            part for part in str(prefix).split("/") if part not in ("", ".")
        )
        is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
        self._py = None if is_url else _PyStore(source)
        # A mutable cell so with_prefix views (shallow copies) share one JS store.
        self._handle = {"js": None}

    @property
    def js(self):
        """The zarrita store object, made on first use."""
        if self._handle["js"] is None:
            if self._py is None:
                self._handle["js"] = _js().zarr.FetchStore.new(self.source)
            else:
                from pyodide.ffi import create_proxy

                self._handle["js"] = create_proxy(self._py)
        return self._handle["js"]

    def close(self) -> None:
        """Release the proxy handed to zarrita; ``with_prefix`` views share it."""
        js = self._handle["js"]
        if js is not None and self._py is not None:
            js.destroy()
        self._handle["js"] = None

    def with_prefix(self, prefix: str) -> "JsZarrStore":
        """A view narrowed to *prefix* under this one; the JS store is shared."""
        view = copy.copy(self)
        view.prefix = posixpath.join(self.prefix, str(prefix).strip("/"))
        return view

    def node_path(self, component=None) -> str:
        """zarrita's absolute path of *component* under the prefix."""
        parts = [
            part
            for source in (self.prefix, component)
            if source
            for part in str(source).split("/")
            if part not in ("", ".")
        ]
        return "/" + "/".join(parts)

    def __getitem__(self, key: str) -> bytes:
        if self._py is not None:
            data = self._py.read(self.node_path(key))
        else:
            data = _block_on(self.js.get(self.node_path(key)))
            if data is not None:
                data = data.to_bytes()
        if data is None:
            raise KeyError(key)
        return data

    def __iter__(self):
        # shortcut: a FetchStore cannot list keys, so an HTTP store without
        # consolidated metadata cannot enumerate children; upgrade to reading
        # the group document's children if that case ever matters.
        if self._py is None:
            return iter(())
        if self._py.root is not None:
            root = (
                self._py.root.joinpath(*self.prefix.split("/"))
                if self.prefix
                else self._py.root
            )
            return (
                p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
            )
        prefix = f"{self.prefix}/" if self.prefix else ""
        return (k[len(prefix) :] for k in self._py.mapping if k.startswith(prefix))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __str__(self) -> str:
        return f"{self.source}/{self.prefix}" if self.prefix else str(self.source)


class JsArrayAdapter(_ZarristaArrayAdapter):
    """dask adapter over a zarrita ``Array`` (a JS object) via zarrGet/zarrSet.

    ``chunks`` is the read granularity (the inner chunk shape of a sharded
    array) and ``write_chunks`` the write one (the shard shape); both come
    from the array document because zarrita exposes only the outer grid.
    """

    def __init__(self, arr, chunks, write_chunks, store_path=None, node_path=None):
        self._arr = arr
        self.shape = tuple(int(n) for n in arr.shape)
        self.dtype = np.dtype(str(arr.dtype))
        self.ndim = len(self.shape)
        self.chunks = tuple(int(c) for c in chunks)
        self.write_chunks = tuple(int(c) for c in write_chunks)
        self.store_path = store_path
        self.node_path = node_path

    @property
    def is_sharded(self) -> bool:
        return self.chunks != self.write_chunks

    def _js_selection(self, normalized):
        zarr = _js().zarr
        bounds = [s.indices(n)[:2] for s, n in zip(normalized, self.shape)]
        return _to_js([zarr.slice(start, stop) for start, stop in bounds])

    def _fetch(self, normalized) -> np.ndarray:
        result = _block_on(_js().zarrGet(self._arr, self._js_selection(normalized)))
        return _from_chunk(result, self.dtype)

    def __setitem__(self, selection, value):
        normalized, dropped, residual = _normalize_selection(self.shape, selection)
        if residual is not None:
            msg = "stepped selections cannot be written"
            raise ValueError(msg)
        region = tuple(
            len(range(*s.indices(n))) for s, n in zip(normalized, self.shape)
        )
        # numpy semantics, as LocalZarrArray: an integer index drops its axis.
        numpy_shape = tuple(n for axis, n in enumerate(region) if axis not in dropped)
        value = np.asarray(value, dtype=self.dtype)
        block = np.broadcast_to(value, numpy_shape).reshape(region)
        _block_on(
            _js().zarrSet(
                self._arr,
                self._js_selection(normalized),
                _to_chunk(_native_contiguous(block)),
            )
        )

    def to_dask(self) -> dask.array.Array:
        return dask.array.from_array(self, chunks=self.chunks)


def _chunk_layout(store: JsZarrStore, component) -> tuple[tuple, tuple]:
    """``(read chunks, write chunks)`` of the array at *component*, from its doc."""
    node = str(component or "").strip("/")
    key = f"{node}/" if node else ""
    try:
        doc = json.loads(store[f"{key}zarr.json"])
    except KeyError:
        chunks = tuple(json.loads(store[f"{key}.zarray"])["chunks"])
        return chunks, chunks
    shard = tuple(doc["chunk_grid"]["configuration"]["chunk_shape"])
    first = (doc.get("codecs") or [{}])[0]
    if first.get("name") == "sharding_indexed":
        return tuple(first["configuration"]["chunk_shape"]), shard
    return shard, shard


def open_js_array(
    store: JsZarrStore, component=None, store_path=None, node_path=None
) -> JsArrayAdapter:
    """Open the array at *component* within *store*."""
    js = _js()
    location = js.zarr.root(store.js).resolve(store.node_path(component))
    arr = _block_on(js.zarr.open(location, _to_js({"kind": "array"})))
    chunks, write_chunks = _chunk_layout(store, component)
    return JsArrayAdapter(arr, chunks, write_chunks, store_path, node_path)


def open_js_lazy(store: JsZarrStore, component=None) -> dask.array.Array:
    """Open the array at *component* within *store* as a lazy dask array."""
    return open_js_array(store, component).to_dask()


def create_js_array(
    path,
    name: str,
    shape: tuple[int, ...],
    dtype,
    chunks: int | tuple[int, ...],
    zarr_format: int,
    compressor=None,
    chunks_per_shard: int | tuple[int, ...] | None = None,
    dimension_names: tuple[str, ...] | None = None,
    dimension_separator: str = "/",
    compressors=None,
    shards: tuple[int, ...] | None = None,
    fill_value=None,
    attributes: dict | None = None,
    chunk_key_encoding: dict | None = None,
) -> JsArrayAdapter:
    """Create the array *name* under the directory *path*; see ``create_zarrista_array``.

    Python writes the metadata document, then zarrita opens the array for
    chunk writes through :meth:`JsArrayAdapter.__setitem__`.
    """
    doc = _array_doc(
        shape,
        dtype,
        chunks,
        zarr_format,
        compressor,
        chunks_per_shard,
        dimension_names,
        dimension_separator,
        compressors,
        shards,
        fill_value,
        attributes,
        chunk_key_encoding,
    )
    path = normalize_store(path)
    name = str(name).strip("/")
    node = path.joinpath(*name.split("/"))
    node.mkdir(parents=True, exist_ok=True)
    if zarr_format == 2:
        (node / ".zarray").write_text(json.dumps(doc, indent=4))
        (node / ".zattrs").write_text(json.dumps(dict(attributes or {}), indent=4))
    else:
        (node / "zarr.json").write_text(json.dumps(doc, indent=4))
    return open_js_array(JsZarrStore(path), name, path, name)
