# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Minimal pure-Python reader for zarr ``MutableMapping`` stores.

zarrista only opens ``FilesystemStore``/``MemoryStore``/``ZipStore`` — its
``SyncStore`` is a closed union with no way to feed it an arbitrary
key-to-bytes mapping (see the Phase 01 capability matrix). This module covers
the store types that therefore cannot go through zarrista: zip archives read
as name-to-bytes mappings, tifffile's ``aszarr`` virtual store, and plain
in-memory dicts.

The zarr format 2 reader parses ``.zgroup``/``.zattrs``/``.zarray`` documents
(consolidated ``.zmetadata`` is used for all metadata lookups when present,
matching zarr-python's consolidated semantics) and decodes chunks with
numcodecs (compressor + filters, C/F order, fill_value for missing chunks,
``.`` and ``/`` dimension separators). A companion zarr format 3 reader
parses ``zarr.json`` documents for unsharded arrays whose codec chain is the
``bytes`` codec plus numcodecs-decodable compression (what tifffile's
``aszarr`` store serves); sharded v3 stores must be read through zarrista.
Arrays are exposed as lazy dask arrays with one delayed chunk fetch per
stored chunk. This module intentionally imports neither zarr-python nor
zarrista.
"""

import asyncio
import base64
import concurrent.futures
import json
import math
from collections.abc import Iterator, Mapping

import dask
import dask.array
import numpy as np
from numcodecs import get_codec
from numcodecs.compat import ensure_bytes, ensure_ndarray

__all__ = [
    "AsyncStoreMapping",
    "AttrsDict",
    "V2Array",
    "V2Group",
    "V3Array",
    "V3Group",
    "open_store_array",
    "open_store_node",
    "open_v2",
    "open_v2_array",
    "open_v2_group",
    "open_v3",
]

_FLOAT_FILLS = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}


class AttrsDict(dict):
    """Plain dict exposing zarr-python's ``Attributes.asdict()`` surface.

    Lets the read path treat node attributes uniformly across backends
    (zarr-python groups, this reader, the compat-layer local reader).
    """

    def asdict(self) -> dict:
        return dict(self)


def _normalize_path(path) -> str:
    """Normalize a node path to the store-key form (no leading/trailing /)."""
    if path is None:
        return ""
    return "/".join(p for p in str(path).split("/") if p not in ("", "."))


def _decode_dtype(spec) -> np.dtype:
    """Decode a ``.zarray`` dtype: a numpy string or the structured list form."""
    if isinstance(spec, str):
        return np.dtype(spec)

    def to_tuple(item):
        if isinstance(item, list):
            return tuple(to_tuple(entry) for entry in item)
        return item

    return np.dtype([to_tuple(field) for field in spec])


def _parse_fill_value(fill, dtype: np.dtype):
    """Decode a JSON ``fill_value`` to a numpy scalar of *dtype*.

    ``None`` (JSON null, "undefined" in the v2 spec) maps to zero, matching
    what zarr-python materializes for missing chunks in that case. Float
    specials arrive as the strings ``"NaN"``/``"Infinity"``/``"-Infinity"``;
    fixed-length bytes and structured fills arrive base64-encoded.
    """
    if fill is None:
        return np.zeros((), dtype=dtype)[()]
    if dtype.kind == "f" and isinstance(fill, str):
        return np.array(_FLOAT_FILLS[fill], dtype=dtype)[()]
    if dtype.kind == "c" and isinstance(fill, (list, tuple)):
        parts = [_FLOAT_FILLS[part] if isinstance(part, str) else part for part in fill]
        return np.array(complex(parts[0], parts[1]), dtype=dtype)[()]
    if dtype.kind in "SV" and isinstance(fill, str):
        decoded = base64.standard_b64decode(fill)
        return np.frombuffer(decoded, dtype=dtype)[0]
    if dtype.kind == "U" and isinstance(fill, str):
        return np.array(fill, dtype=dtype)[()]
    return np.array(fill, dtype=dtype)[()]


def _run_coroutine(coro):
    """Run *coro* to completion from synchronous code.

    Chunk fetches happen inside dask worker threads, which have no running
    event loop; when one is running (e.g. the caller's thread in Jupyter),
    the coroutine runs on a private loop in a helper thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _BytesBufferShim:
    """Buffer factory returning plain ``bytes``, ducking zarr's Buffer API."""

    @staticmethod
    def from_bytes(data) -> bytes:
        return bytes(data)

    @staticmethod
    def from_array_like(array_like) -> bytes:
        return np.asarray(array_like).tobytes()


class _BytesPrototypeShim:
    """The ``prototype.buffer`` surface async stores build results through."""

    buffer = _BytesBufferShim


class AsyncStoreMapping(Mapping):
    """Read-only ``Mapping[str, bytes]`` view over an async byte store.

    Adapts stores implementing the zarr-python 3 async ``Store`` interface
    (``await get(key, prototype)``, ``async for key in list()``) -- e.g.
    tifffile's ``aszarr`` ``ZarrTiffStore`` -- to the bytes-mapping surface
    :func:`open_v2` consumes, without importing zarr-python: the prototype
    handed to ``get`` is a minimal shim producing plain ``bytes``.

    Iteration yields the store's listable keys (for tifffile these are the
    metadata documents; chunk keys are computed on demand), which is exactly
    the surface :class:`_V2Source` needs for metadata lookups and child
    listing. Missing keys -- including chunks the store reports as absent --
    raise ``KeyError``, which the reader materializes as ``fill_value``.
    """

    def __init__(self, store):
        self._store = store
        self._keys: list[str] | None = None

    def __getitem__(self, key: str) -> bytes:
        value = _run_coroutine(self._store.get(key, _BytesPrototypeShim))
        if value is None:
            raise KeyError(key)
        if hasattr(value, "to_bytes"):
            value = value.to_bytes()
        return bytes(value)

    def _list_keys(self) -> list[str]:
        if self._keys is None:

            async def collect() -> list[str]:
                return [key async for key in self._store.list()]

            self._keys = _run_coroutine(collect())
        return self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._list_keys())

    def __len__(self) -> int:
        return len(self._list_keys())


class _V2Source:
    """A ``MutableMapping[str, bytes]`` plus its consolidated metadata.

    Shared by every node opened from one store so ``.zmetadata`` is parsed
    once. When consolidated metadata is present it is the sole source for
    metadata documents and child listing (zarr-python's consolidated
    semantics); chunk payloads always come from the mapping.
    """

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        self.consolidated: dict | None = None
        raw = self.raw(".zmetadata")
        if raw is not None:
            doc = json.loads(ensure_bytes(raw))
            if isinstance(doc, dict) and isinstance(doc.get("metadata"), dict):
                self.consolidated = doc["metadata"]

    def raw(self, key: str) -> bytes | None:
        try:
            return self.mapping[key]
        except KeyError:
            return None

    def doc(self, key: str) -> dict | None:
        """The metadata JSON document at *key*, or ``None`` when absent."""
        if self.consolidated is not None:
            return self.consolidated.get(key)
        raw = self.raw(key)
        return None if raw is None else json.loads(ensure_bytes(raw))

    def metadata_keys(self) -> Iterator[str]:
        if self.consolidated is not None:
            return iter(self.consolidated)
        return iter(self.mapping)


def _doc_key(path: str, doc_name: str) -> str:
    return f"{path}/{doc_name}" if path else doc_name


class _LazyChunkedArray:
    """Shared lazy-dask surface for chunked array nodes read from a mapping.

    Subclasses set ``shape``/``chunks``/``dtype``/``ndim`` and implement
    ``_read_chunk(chunk_index) -> np.ndarray`` returning the in-bounds extent
    at that grid position.
    """

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def _trim_edge_chunk(
        self, chunk: np.ndarray, chunk_index: tuple[int, ...]
    ) -> np.ndarray:
        """Trim a full-shape stored chunk to its in-bounds extent.

        Both zarr formats store edge chunks padded to the full chunk shape.
        """
        if not chunk_index:
            return chunk
        bounds = tuple(
            slice(0, min(c, s - i * c))
            for i, s, c in zip(chunk_index, self.shape, self.chunks)
        )
        return chunk[bounds]

    def to_dask(self) -> dask.array.Array:
        """The array as a lazy dask array chunked on the stored chunk grid.

        One delayed chunk fetch per stored chunk; nothing is read until
        compute. ``pure=False`` keeps dask from tokenizing the fetch
        arguments, which would hash (or choke on) the backing store object.
        """
        if 0 in self.shape:
            return dask.array.empty(self.shape, dtype=self.dtype, chunks=self.chunks)
        fetch = dask.delayed(self._read_chunk, pure=False)
        if self.ndim == 0:
            return dask.array.from_delayed(fetch(()), shape=(), dtype=self.dtype)
        grid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

        def build(index: tuple[int, ...]):
            axis = len(index)
            if axis == self.ndim:
                chunk_shape = tuple(
                    min(c, s - i * c) for i, s, c in zip(index, self.shape, self.chunks)
                )
                return dask.array.from_delayed(
                    fetch(index), shape=chunk_shape, dtype=self.dtype
                )
            return [build((*index, i)) for i in range(grid[axis])]

        return dask.array.block(build(()))


class V2Array(_LazyChunkedArray):
    """A zarr format 2 array node read lazily from a bytes mapping."""

    def __init__(self, source: _V2Source, path: str, meta: dict):
        self._source = source
        self.path = path
        self.shape = tuple(int(s) for s in meta["shape"])
        self.chunks = tuple(int(c) for c in meta["chunks"])
        self.dtype = _decode_dtype(meta["dtype"])
        self.ndim = len(self.shape)
        self.fill_value = _parse_fill_value(meta.get("fill_value"), self.dtype)
        self._order = meta.get("order", "C")
        self._separator = meta.get("dimension_separator") or "."
        compressor_config = meta.get("compressor")
        self._compressor = (
            get_codec(compressor_config) if compressor_config is not None else None
        )
        self._filters = [get_codec(config) for config in meta.get("filters") or []]
        attrs = source.doc(_doc_key(path, ".zattrs"))
        self.attrs = AttrsDict(attrs) if attrs else AttrsDict()

    def _chunk_key(self, chunk_index: tuple[int, ...]) -> str:
        prefix = f"{self.path}/" if self.path else ""
        if not chunk_index:
            return prefix + "0"
        return prefix + self._separator.join(str(i) for i in chunk_index)

    def _decode_chunk(self, raw) -> np.ndarray:
        chunk = ensure_bytes(raw)
        if self._compressor is not None:
            chunk = self._compressor.decode(chunk)
        for codec in reversed(self._filters):
            chunk = codec.decode(chunk)
        arr = ensure_ndarray(chunk).view(self.dtype)
        return arr.reshape(self.chunks, order=self._order)

    def _read_chunk(self, chunk_index: tuple[int, ...]) -> np.ndarray:
        raw = self._source.raw(self._chunk_key(chunk_index))
        if raw is None:
            chunk = np.full(self.chunks, self.fill_value, dtype=self.dtype)
        else:
            chunk = self._decode_chunk(raw)
        return self._trim_edge_chunk(chunk, chunk_index)


class V2Group:
    """A zarr format 2 group node read from a bytes mapping."""

    def __init__(self, source: _V2Source, path: str):
        self._source = source
        self.path = path
        attrs = source.doc(_doc_key(path, ".zattrs"))
        self.attrs = AttrsDict(attrs) if attrs else AttrsDict()

    def _child_path(self, name) -> str:
        child = _normalize_path(name)
        return f"{self.path}/{child}" if self.path else child

    def __contains__(self, name) -> bool:
        path = self._child_path(name)
        return (
            self._source.doc(_doc_key(path, ".zarray")) is not None
            or self._source.doc(_doc_key(path, ".zgroup")) is not None
        )

    def __getitem__(self, name):
        path = self._child_path(name)
        meta = self._source.doc(_doc_key(path, ".zarray"))
        if meta is not None:
            return V2Array(self._source, path, meta)
        if self._source.doc(_doc_key(path, ".zgroup")) is not None:
            return V2Group(self._source, path)
        raise KeyError(f"No zarr format 2 node found at {path!r}")

    def open_array(self, name) -> "V2Array":
        node = self[name]
        if not isinstance(node, V2Array):
            raise ValueError(f"Node at {self._child_path(name)!r} is a group")
        return node

    def open_group(self, name) -> "V2Group":
        node = self[name]
        if not isinstance(node, V2Group):
            raise ValueError(f"Node at {self._child_path(name)!r} is an array")
        return node

    def _child_names(self, doc_name: str) -> list[str]:
        prefix = f"{self.path}/" if self.path else ""
        names = set()
        for key in self._source.metadata_keys():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].split("/")
            if len(parts) == 2 and parts[1] == doc_name:
                names.add(parts[0])
        return sorted(names)

    def array_keys(self) -> list[str]:
        """Names of the immediate child arrays, sorted."""
        return self._child_names(".zarray")

    def group_keys(self) -> list[str]:
        """Names of the immediate child groups, sorted."""
        return self._child_names(".zgroup")

    def keys(self) -> list[str]:
        """Names of all immediate child nodes (arrays and groups), sorted."""
        return sorted({*self.array_keys(), *self.group_keys()})


def open_v2(store: Mapping, path=None):
    """Open the node at *path* within a v2 bytes-mapping *store*.

    Returns a :class:`V2Array` or :class:`V2Group` depending on the node
    type (arrays win when both documents exist, matching zarr-python), or
    raises ``KeyError`` when neither metadata document is present.
    """
    source = _V2Source(store)
    node_path = _normalize_path(path)
    meta = source.doc(_doc_key(node_path, ".zarray"))
    if meta is not None:
        return V2Array(source, node_path, meta)
    if source.doc(_doc_key(node_path, ".zgroup")) is not None:
        return V2Group(source, node_path)
    raise KeyError(
        f"No zarr format 2 array or group found at path {node_path!r} "
        "(missing .zarray/.zgroup)"
    )


def open_v2_array(store: Mapping, path=None) -> V2Array:
    """Open the array node at *path* within a v2 bytes-mapping *store*."""
    node = open_v2(store, path)
    if not isinstance(node, V2Array):
        raise ValueError(f"Node at {_normalize_path(path)!r} is a group, not an array")
    return node


def open_v2_group(store: Mapping, path=None) -> V2Group:
    """Open the group node at *path* within a v2 bytes-mapping *store*."""
    node = open_v2(store, path)
    if not isinstance(node, V2Group):
        raise ValueError(f"Node at {_normalize_path(path)!r} is an array, not a group")
    return node


# -- zarr format 3 ----------------------------------------------------------

# Zarr v3 bytes-to-bytes codec name -> numcodecs config for decoding. The
# decode direction ignores compression-level knobs, so only structurally
# relevant configuration is translated.
_V3_BLOSC_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def _v3_bytes_codec(codec: dict):
    """A numcodecs codec that decodes the given v3 bytes-to-bytes codec."""
    name = codec.get("name")
    config = codec.get("configuration") or {}
    if name == "gzip":
        return get_codec({"id": "gzip", "level": config.get("level", 5)})
    if name == "zstd":
        return get_codec({"id": "zstd", "level": config.get("level", 0)})
    if name == "blosc":
        return get_codec(
            {
                "id": "blosc",
                "cname": config.get("cname", "lz4"),
                "clevel": config.get("clevel", 5),
                "shuffle": _V3_BLOSC_SHUFFLE.get(config.get("shuffle", "shuffle"), 1),
                "blocksize": config.get("blocksize", 0),
            }
        )
    raise ValueError(
        f"Unsupported zarr format 3 codec {name!r} in a mapping store. "
        "Supported bytes-to-bytes codecs: gzip, zstd, blosc; sharded stores "
        "must be read through zarrista."
    )


class V3Array(_LazyChunkedArray):
    """An unsharded zarr format 3 array node read lazily from a bytes mapping.

    Supports the ``bytes`` array-to-bytes codec (either endian) followed by
    numcodecs-decodable compression. The sharding and transpose codecs are
    rejected: stores using them should be read through zarrista instead.
    """

    def __init__(self, mapping: Mapping, path: str, doc: dict):
        self._mapping = mapping
        self.path = path
        self.shape = tuple(int(s) for s in doc["shape"])
        self.ndim = len(self.shape)
        grid = doc.get("chunk_grid") or {}
        if grid.get("name") != "regular":
            raise ValueError(
                f"Unsupported zarr format 3 chunk grid {grid.get('name')!r}"
            )
        self.chunks = tuple(
            int(c) for c in (grid.get("configuration") or {})["chunk_shape"]
        )

        codecs = list(doc.get("codecs") or [])
        if not codecs or codecs[0].get("name") != "bytes":
            first = codecs[0].get("name") if codecs else None
            raise ValueError(
                f"Unsupported zarr format 3 array-to-bytes codec {first!r} in "
                "a mapping store; only the 'bytes' codec is supported. "
                "Sharded stores must be read through zarrista."
            )
        endian = (codecs[0].get("configuration") or {}).get("endian", "little")
        dtype = np.dtype(doc["data_type"])
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder("<" if endian == "little" else ">")
        self.dtype = dtype
        self._decoders = [_v3_bytes_codec(codec) for codec in codecs[1:]]

        key_encoding = doc.get("chunk_key_encoding") or {"name": "default"}
        self._key_encoding = key_encoding.get("name", "default")
        default_separator = "/" if self._key_encoding == "default" else "."
        self._separator = (key_encoding.get("configuration") or {}).get(
            "separator", default_separator
        )
        self.fill_value = _parse_fill_value(doc.get("fill_value"), self.dtype)
        self.attrs = AttrsDict(doc.get("attributes") or {})

    def _chunk_key(self, chunk_index: tuple[int, ...]) -> str:
        prefix = f"{self.path}/" if self.path else ""
        parts = [str(i) for i in chunk_index]
        if self._key_encoding == "default":
            parts = ["c", *parts]
        elif not parts:
            parts = ["0"]
        return prefix + self._separator.join(parts)

    def _read_chunk(self, chunk_index: tuple[int, ...]) -> np.ndarray:
        try:
            raw = self._mapping[self._chunk_key(chunk_index)]
        except KeyError:
            raw = None
        if raw is None:
            chunk = np.full(self.chunks, self.fill_value, dtype=self.dtype)
        else:
            data = ensure_bytes(raw)
            for codec in reversed(self._decoders):
                data = codec.decode(data)
            chunk = ensure_ndarray(data).view(self.dtype).reshape(self.chunks)
        return self._trim_edge_chunk(chunk, chunk_index)


class V3Group:
    """A zarr format 3 group node read from a bytes mapping."""

    def __init__(self, mapping: Mapping, path: str, doc: dict):
        self._mapping = mapping
        self.path = path
        self.attrs = AttrsDict(doc.get("attributes") or {})

    def _child_path(self, name) -> str:
        child = _normalize_path(name)
        return f"{self.path}/{child}" if self.path else child

    def _child_doc(self, path: str) -> dict | None:
        try:
            raw = self._mapping[_doc_key(path, "zarr.json")]
        except KeyError:
            return None
        return json.loads(ensure_bytes(raw))

    def __contains__(self, name) -> bool:
        doc = self._child_doc(self._child_path(name))
        return doc is not None and doc.get("node_type") in ("array", "group")

    def __getitem__(self, name):
        path = self._child_path(name)
        doc = self._child_doc(path)
        if doc is None or doc.get("node_type") not in ("array", "group"):
            raise KeyError(f"No zarr format 3 node found at {path!r}")
        if doc["node_type"] == "array":
            return V3Array(self._mapping, path, doc)
        return V3Group(self._mapping, path, doc)

    def _child_names(self, node_type: str) -> list[str]:
        prefix = f"{self.path}/" if self.path else ""
        names = set()
        for key in self._mapping:
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].split("/")
            if len(parts) != 2 or parts[1] != "zarr.json":
                continue
            doc = self._child_doc(self._child_path(parts[0]))
            if doc is not None and doc.get("node_type") == node_type:
                names.add(parts[0])
        return sorted(names)

    def array_keys(self) -> list[str]:
        """Names of the immediate child arrays, sorted."""
        return self._child_names("array")

    def group_keys(self) -> list[str]:
        """Names of the immediate child groups, sorted."""
        return self._child_names("group")

    def keys(self) -> list[str]:
        """Names of all immediate child nodes (arrays and groups), sorted."""
        return sorted({*self.array_keys(), *self.group_keys()})


def open_v3(store: Mapping, path=None):
    """Open the node at *path* within a v3 bytes-mapping *store*.

    Returns a :class:`V3Array` or :class:`V3Group`, or raises ``KeyError``
    when no ``zarr.json`` document is present at that path.
    """
    node_path = _normalize_path(path)
    try:
        raw = store[_doc_key(node_path, "zarr.json")]
    except KeyError:
        raw = None
    doc = None if raw is None else json.loads(ensure_bytes(raw))
    if doc is None or doc.get("node_type") not in ("array", "group"):
        raise KeyError(
            f"No zarr format 3 array or group found at path {node_path!r} "
            "(missing zarr.json)"
        )
    if doc["node_type"] == "array":
        return V3Array(store, node_path, doc)
    return V3Group(store, node_path, doc)


def open_store_node(store: Mapping, path=None):
    """Open the node at *path* within *store*, auto-detecting the zarr format.

    Zarr format 3 (``zarr.json``) wins when both formats are present,
    matching zarr-python's auto-detection order.
    """
    try:
        return open_v3(store, path)
    except KeyError:
        return open_v2(store, path)


def open_store_array(store: Mapping, path=None):
    """Open the array node at *path* within *store* (either zarr format)."""
    node = open_store_node(store, path)
    if isinstance(node, (V2Group, V3Group)):
        raise ValueError(f"Node at {_normalize_path(path)!r} is a group, not an array")
    return node
