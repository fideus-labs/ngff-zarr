# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Minimal pure-Python reader for zarr format 2 ``MutableMapping`` stores.

zarrista only opens ``FilesystemStore``/``MemoryStore``/``ZipStore`` — its
``SyncStore`` is a closed union with no way to feed it an arbitrary
key-to-bytes mapping (see the Phase 01 capability matrix). This module covers
the store types that therefore cannot go through zarrista: zip archives read
as name-to-bytes mappings, tifffile's ``aszarr`` virtual store, and plain
in-memory dicts.

The reader parses ``.zgroup``/``.zattrs``/``.zarray`` documents (consolidated
``.zmetadata`` is used for all metadata lookups when present, matching
zarr-python's consolidated semantics), decodes chunks with numcodecs
(compressor + filters, C/F order, fill_value for missing chunks, ``.`` and
``/`` dimension separators), and exposes arrays as lazy dask arrays with one
delayed chunk fetch per stored chunk. It intentionally imports neither
zarr-python nor zarrista.
"""

import base64
import json
from collections.abc import Iterator, Mapping

import dask
import dask.array
import numpy as np
from numcodecs import get_codec
from numcodecs.compat import ensure_bytes, ensure_ndarray

__all__ = [
    "V2Array",
    "V2Group",
    "open_v2",
    "open_v2_array",
    "open_v2_group",
]

_FLOAT_FILLS = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}


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


class V2Array:
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
        self.attrs = dict(attrs) if attrs else {}

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
        if not chunk_index:
            return chunk
        # v2 edge chunks are stored padded to the full chunk shape; trim to
        # the in-bounds extent expected at this grid position.
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


class V2Group:
    """A zarr format 2 group node read from a bytes mapping."""

    def __init__(self, source: _V2Source, path: str):
        self._source = source
        self.path = path
        attrs = source.doc(_doc_key(path, ".zattrs"))
        self.attrs = dict(attrs) if attrs else {}

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
