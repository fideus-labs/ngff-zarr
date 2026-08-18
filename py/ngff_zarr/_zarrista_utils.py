# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Compatibility layer between ngff-zarr and zarrista (Rust/zarrs-backed Zarr).

zarrista is v3-focused and its API surface differs from zarr-python in ways
that matter for OME-Zarr writing. The empirically verified constraints this
module works around (see the Phase 01 capability spike):

- No group-create API for either zarr format: group metadata (``zarr.json``
  for format 3, ``.zgroup``/``.zattrs`` for format 2) is hand-written JSON.
- ``ArrayBuilder`` emits only zarr format 3 metadata: format 2 arrays are
  created via ``Array.from_metadata`` with a ``.zarray``-style dict, which
  writes real v2 stores that zarr-python reads back correctly.
- Consolidated ``.zmetadata`` cannot be written by zarrista for format 2:
  it is hand-written by aggregating the per-node JSON documents.
- Every write requires C-contiguous input, and arrays expose ``shape`` as a
  ``list`` and ``dtype`` as zarrista's own ``DataType``: dask interop goes
  through a thin adapter (:class:`_ZarristaArrayAdapter`).
- Supported stores are ``FilesystemStore``/``MemoryStore``/read-only
  ``ZipStore`` only: :func:`normalize_store` restricts targets to local
  directory paths.

Sharding vocabulary differs from zarr-python: the builder chunk grid is the
*shard* (outer chunk) grid and ``subchunk_shape`` is the read chunk, so
ngff-zarr's ``chunks_per_shard`` multiplies ``chunks`` into the builder grid
shape while ``chunks`` itself becomes the subchunk shape.

zarrista requires Python >= 3.11 and is an optional dependency, so it is
imported lazily inside each function.
"""

import json
import os
from pathlib import Path

import dask.array
import numpy as np
import zarr.storage

from .rfc9_zip import is_ozx_path

__all__ = [
    "consolidate_metadata",
    "create_zarrista_array",
    "create_zarrista_group",
    "normalize_store",
    "open_zarrista_array",
    "open_zarrista_lazy",
    "write_dask_array",
]

_BLOSC_SHUFFLE_TO_INT = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
_BLOSC_SHUFFLE_TO_STR = {v: k for k, v in _BLOSC_SHUFFLE_TO_INT.items()}


class _ZarristaArrayAdapter:
    """Minimal numpy-protocol surface over a zarrista ``Array`` for dask.

    zarrista exposes ``shape`` as a ``list``, ``dtype`` as its own
    ``DataType``, has no ``.chunks`` attribute, and rejects non-C-contiguous
    write input. dask needs a tuple ``shape``, a numpy ``dtype``, ndarray
    ``__getitem__`` results, and contiguity-tolerant ``__setitem__``.
    """

    def __init__(self, arr):
        self._arr = arr
        self.shape = tuple(arr.shape)
        self.dtype = np.dtype(arr.dtype.name)
        self.ndim = arr.ndim

    def __getitem__(self, selection):
        return np.asarray(self._arr[selection])

    def __setitem__(self, selection, value):
        self._arr[selection] = np.ascontiguousarray(value)


def _canonical_compressor(compressor) -> tuple[str, dict] | None:
    """Normalize any accepted compressor form to a ``(name, params)`` pair.

    Accepts a numcodecs codec object, a zarr v3 codec object, a bare codec
    name string, a numcodecs-style config dict (``{"id": ...}``), or a zarr
    v3 codec dict (``{"name": ..., "configuration": ...}``). Returns ``None``
    for ``None``.
    """
    if compressor is None:
        return None
    if isinstance(compressor, str):
        return compressor, {}
    if hasattr(compressor, "codec_id"):
        params = dict(compressor.get_config())
        return params.pop("id"), params
    if hasattr(compressor, "to_dict"):
        as_dict = compressor.to_dict()
        return as_dict["name"], dict(as_dict.get("configuration") or {})
    if isinstance(compressor, dict):
        params = dict(compressor)
        if "id" in params:
            return params.pop("id"), params
        if "name" in params:
            return params["name"], dict(params.get("configuration") or {})
    raise TypeError(
        f"Unsupported compressor specification: {compressor!r}. Pass a "
        "numcodecs codec, a zarr v3 codec, or their config dict forms."
    )


def _blosc_shuffle_int(value, itemsize: int = 4) -> int:
    if value is None or (not isinstance(value, str) and int(value) == -1):
        # numcodecs AUTOSHUFFLE / an unresolved zarr v3 default: bitshuffle
        # for single-byte dtypes, byte shuffle otherwise, matching how
        # zarr-python evolves the codec from the dtype at write time.
        return 2 if itemsize == 1 else 1
    if isinstance(value, str):
        return _BLOSC_SHUFFLE_TO_INT[value]
    return int(value)


def _compressor_to_v2_config(compressor, dtype: np.dtype) -> dict | None:
    """Map a compressor to the numcodecs config dict stored in ``.zarray``."""
    canonical = _canonical_compressor(compressor)
    if canonical is None:
        return None
    name, params = canonical
    if name == "blosc":
        return {
            "id": "blosc",
            "cname": params.get("cname", "lz4"),
            "clevel": params.get("clevel", 5),
            "shuffle": _blosc_shuffle_int(params.get("shuffle", 1), dtype.itemsize),
            "blocksize": params.get("blocksize", 0) or 0,
        }
    if name == "gzip":
        return {"id": "gzip", "level": params.get("level", 5)}
    if name == "zstd":
        return {"id": "zstd", "level": params.get("level", 3)}
    raise ValueError(
        f"Compressor {name!r} is not supported by the zarrista "
        "compatibility layer (supported: blosc, gzip, zstd)"
    )


def _compressor_to_zarrista_codec(compressor, dtype: np.dtype):
    """Map a compressor to a zarrista bytes-to-bytes codec (zarr format 3)."""
    from zarrista import codec as zarrista_codec

    canonical = _canonical_compressor(compressor)
    if canonical is None:
        return None
    name, params = canonical
    if name == "blosc":
        shuffle = _BLOSC_SHUFFLE_TO_STR[
            _blosc_shuffle_int(params.get("shuffle", 1), dtype.itemsize)
        ]
        return zarrista_codec.blosc(
            params.get("cname", "lz4"),
            params.get("clevel", 5),
            shuffle,
            typesize=params.get("typesize") or dtype.itemsize,
        )
    if name == "gzip":
        return zarrista_codec.gzip(params.get("level", 5))
    if name == "zstd":
        return zarrista_codec.zstd(
            params.get("level", 3), params.get("checksum", False)
        )
    raise ValueError(
        f"Compressor {name!r} is not supported by the zarrista "
        "compatibility layer (supported: blosc, gzip, zstd)"
    )


def normalize_store(store) -> Path:
    """Normalize *store* to a local directory path zarrista can target.

    Accepts ``str``/``pathlib.Path``/``os.PathLike`` and local zarr-python
    stores that wrap a directory path. Raises ``TypeError`` for store objects
    zarrista cannot handle (in-memory mappings, remote stores, ...) and
    ``ValueError`` for zip targets, which zarrista can only read.
    """
    local_store_cls = getattr(zarr.storage, "LocalStore", None)
    if local_store_cls is not None and isinstance(store, local_store_cls):
        store = store.root
    directory_store_cls = getattr(zarr.storage, "DirectoryStore", None)
    if directory_store_cls is not None and isinstance(store, directory_store_cls):
        store = store.path
    if not isinstance(store, (str, Path, os.PathLike)):
        raise TypeError(
            "The zarrista compatibility layer requires a local path store; "
            f"got {type(store).__name__}. Pass a str, pathlib.Path, or "
            "os.PathLike directory path. In-memory mappings and other zarr "
            "store objects must use the zarr-python engine."
        )
    path = Path(os.fspath(store))
    if path.suffix.lower() == ".zip" or is_ozx_path(path):
        raise ValueError(
            "zarrista cannot write into zip archives (.zip/.ozx). Write to a "
            "directory and zip it afterwards (see ngff_zarr.rfc9_zip), or use "
            "the zarr-python engine."
        )
    return path


def create_zarrista_group(path, attributes: dict | None, zarr_format: int):
    """Create group metadata at *path* and return the opened zarrista group.

    zarrista has no group-create API, so the metadata documents are written
    directly: ``.zgroup`` + ``.zattrs`` for zarr format 2, ``zarr.json`` for
    format 3. The returned ``zarrista.Group`` reads them back, hiding the
    hand-written JSON from callers.
    """
    from zarrista import Group
    from zarrista.store import FilesystemStore

    path = normalize_store(path)
    path.mkdir(parents=True, exist_ok=True)
    attributes = attributes or {}
    if zarr_format == 2:
        (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=4))
        (path / ".zattrs").write_text(json.dumps(attributes, indent=4))
    elif zarr_format == 3:
        metadata = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attributes,
        }
        (path / "zarr.json").write_text(json.dumps(metadata, indent=4))
    else:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
    return Group.open(FilesystemStore(path), "/")


def create_zarrista_array(
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
):
    """Create a zarr array under the store at *path* and return it.

    *name* is the array's path within the store (e.g. ``"0"``). *compressor*
    accepts the same forms as ngff-zarr's writer: a numcodecs codec, a zarr
    v3 codec, or their config dicts; ``None`` means no compression.
    *compressors* (format 3 only) is an ordered sequence of bytes-to-bytes
    codecs written after the ``bytes`` codec; it overrides *compressor*, and
    an empty sequence means no compression. Sharding (format 3 only) is
    requested either via ``chunks_per_shard`` — an int or per-dimension tuple
    of chunks per shard, matching ``to_ngff_zarr`` — or via ``shards``, the
    explicit outer shard shape whose subchunks are *chunks*. The metadata is
    stored immediately; the returned ``zarrista.Array`` is ready for chunk
    writes.
    """
    path = normalize_store(path)
    path.mkdir(parents=True, exist_ok=True)
    node_path = "/" + str(name).strip("/")
    dtype = np.dtype(dtype)
    shape = tuple(int(s) for s in shape)
    if isinstance(chunks, int):
        chunks = (chunks,) * len(shape)
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) != len(shape):
        raise ValueError(f"chunks must have {len(shape)} dimensions, got {chunks}")

    from zarrista import Array, ArrayBuilder, ChunkGrid, DataType, FillValue
    from zarrista.store import FilesystemStore

    store = FilesystemStore(path)

    if zarr_format == 2:
        if chunks_per_shard is not None or shards is not None:
            raise ValueError("Sharding is only supported for zarr format 3")
        if dimension_names is not None:
            raise ValueError("dimension_names requires zarr format 3")
        if compressors is not None:
            raise ValueError(
                "compressors (a zarr v3 codec chain) requires zarr format 3; "
                "pass a single compressor for zarr format 2"
            )
        metadata = {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": dtype.str,
            "compressor": _compressor_to_v2_config(compressor, dtype),
            "fill_value": np.zeros((), dtype=dtype).item(),
            "order": "C",
            "filters": None,
            "dimension_separator": dimension_separator,
        }
        arr = Array.from_metadata(metadata, store, node_path)
        arr.store_metadata()
        return arr

    if zarr_format != 3:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")

    # ngff-zarr's dtype normalization also yields valid zarr v3 dtype names.
    from .to_ngff_zarr import _numpy_to_zarr_dtype

    if shards is not None:
        if chunks_per_shard is not None:
            raise ValueError("Pass either chunks_per_shard or shards, not both")
        shards = tuple(int(s) for s in shards)
        if len(shards) != len(shape):
            raise ValueError(f"shards must have {len(shape)} dimensions, got {shards}")
        grid_chunk_shape = shards
    elif chunks_per_shard is None:
        grid_chunk_shape = chunks
    else:
        if isinstance(chunks_per_shard, int):
            chunks_per_shard = (chunks_per_shard,) * len(shape)
        if len(chunks_per_shard) != len(shape):
            raise ValueError(
                f"chunks_per_shard must have {len(shape)} dimensions, "
                f"got {chunks_per_shard}"
            )
        grid_chunk_shape = tuple(c * int(f) for c, f in zip(chunks, chunks_per_shard))

    grid = ChunkGrid.regular(list(shape), chunk_shape=list(grid_chunk_shape))
    data_type = DataType.from_string(_numpy_to_zarr_dtype(dtype))
    fill_value = FillValue(np.zeros((), dtype=dtype).tobytes())
    builder = ArrayBuilder(grid, data_type, fill_value)
    if chunks_per_shard is not None or shards is not None:
        builder = builder.subchunk_shape(list(chunks))
    if compressors is not None:
        codecs = [
            codec
            for codec in (
                _compressor_to_zarrista_codec(entry, dtype) for entry in compressors
            )
            if codec is not None
        ]
    else:
        codec = _compressor_to_zarrista_codec(compressor, dtype)
        codecs = [codec] if codec is not None else []
    if codecs:
        builder = builder.compressors(codecs)
    if dimension_names is not None:
        builder = builder.dimension_names(list(dimension_names))
    return builder.create(store, node_path)


def write_dask_array(darr, zarrista_array, region=None) -> None:
    """Write dask array *darr* into an existing zarrista array.

    Writes are chunk-aligned: *darr* is rechunked to the target's write
    granularity (the shard shape for sharded arrays) so concurrent block
    writes never touch the same stored chunk, then streamed with
    ``dask.array.store``. A *region* (tuple of slices) selects the target
    subset and should be aligned to the same granularity. Empty arrays
    (a zero-length dimension) are a no-op.
    """
    darr = dask.array.asarray(darr)
    if darr.size == 0:
        return
    write_chunks = tuple(zarrista_array.chunk_shape([0] * zarrista_array.ndim))
    if darr.chunksize != write_chunks:
        darr = darr.rechunk(write_chunks)
    target = _ZarristaArrayAdapter(zarrista_array)
    store_kwargs = {"lock": False}
    if region is not None:
        store_kwargs["regions"] = region
    dask.array.store(darr, target, **store_kwargs)


def open_zarrista_array(path, component: str | None = None):
    """Open the existing array at *component* within the store at *path*.

    Returns the raw ``zarrista.Array``, ready for region ``__setitem__``
    writes into an already-created array (either zarr format). Write input
    must be C-contiguous; :class:`_ZarristaArrayAdapter` or
    :func:`write_dask_array` handle that for adapter-mediated writes.
    """
    from zarrista import Array
    from zarrista.store import FilesystemStore

    path = normalize_store(path)
    node_path = "/" + str(component).strip("/") if component else "/"
    return Array.open(FilesystemStore(path), node_path)


def open_zarrista_lazy(path, component: str | None = None) -> dask.array.Array:
    """Open the array at *component* within *path* as a lazy dask array.

    Chunks follow the stored chunk grid, using the subchunk (inner chunk)
    shape for sharded arrays so reads stay at the efficient granularity.
    """
    arr = open_zarrista_array(path, component)
    if arr.is_sharded:
        chunks = tuple(arr.subchunk_shape)
    else:
        chunks = tuple(arr.chunk_shape([0] * arr.ndim))
    return dask.array.from_array(_ZarristaArrayAdapter(arr), chunks=chunks)


def consolidate_metadata(path, zarr_format: int) -> None:
    """Write consolidated metadata for the store at *path*.

    zarrista cannot write zarr format 2 consolidated metadata, so
    ``.zmetadata`` is hand-written by aggregating the per-node JSON documents
    (what OME-Zarr 0.4 stores carry). Format 3 consolidation is inlined into
    the root ``zarr.json`` through zarrista's native API, matching
    ``zarr.consolidate_metadata``.
    """
    path = normalize_store(path)
    if zarr_format == 2:
        metadata = {}
        for doc in sorted(path.rglob(".z*")):
            if doc.name not in (".zgroup", ".zattrs", ".zarray"):
                continue
            metadata[doc.relative_to(path).as_posix()] = json.loads(doc.read_text())
        consolidated = {"metadata": metadata, "zarr_consolidated_format": 1}
        (path / ".zmetadata").write_text(json.dumps(consolidated, indent=4))
    elif zarr_format == 3:
        from zarrista import Group
        from zarrista.store import FilesystemStore

        metadata = {}
        for doc in sorted(path.rglob("zarr.json")):
            if doc == path / "zarr.json":
                continue
            key = doc.parent.relative_to(path).as_posix()
            metadata[key] = json.loads(doc.read_text())
        group = Group.open(FilesystemStore(path), "/")
        group = group.with_consolidated_metadata(
            {"kind": "inline", "must_understand": False, "metadata": metadata}
        )
        group.store_metadata()
    else:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
