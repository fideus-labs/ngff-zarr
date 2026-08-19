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

import importlib.util
import json
import os
import shutil
import threading
from collections.abc import MutableMapping
from pathlib import Path

import dask.array
import numpy as np
import zarr.storage

from .rfc9_zip import is_ozx_path

__all__ = [
    "consolidate_metadata",
    "create_zarrista_array",
    "create_zarrista_group",
    "create_zarrista_subgroup",
    "normalize_store",
    "open_zarrista_array",
    "open_zarrista_lazy",
    "read_group_attributes",
    "resolve_store_path",
    "use_zarrista_for",
    "write_dask_array",
]

_BLOSC_SHUFFLE_TO_INT = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
_BLOSC_SHUFFLE_TO_STR = {v: k for k, v in _BLOSC_SHUFFLE_TO_INT.items()}


def _native_contiguous(value) -> np.ndarray:
    """Return *value* as a C-contiguous, native-byte-order ndarray.

    zarrista ingests write input through DLPack, which rejects both
    non-contiguous buffers and non-native byte order (e.g. big-endian
    ``>u2`` input); the target array's stored byte order comes from its
    metadata, so byteswapping the in-memory values is safe.
    """
    value = np.ascontiguousarray(value)
    if not value.dtype.isnative:
        value = value.astype(value.dtype.newbyteorder("="))
    # numpy's C_CONTIGUOUS flag ignores the strides of size-1 dimensions, so
    # ascontiguousarray can return a view whose singleton-dim strides still
    # point into a larger parent array. DLPack consumers (zarrs) validate the
    # raw strides and reject such views; copy to canonical strides.
    expected_strides = []
    stride = value.itemsize
    for dim in reversed(value.shape):
        expected_strides.append(stride)
        stride *= dim
    if value.strides != tuple(reversed(expected_strides)):
        value = value.copy()
    return value


class _ZarristaArrayAdapter:
    """Minimal numpy-protocol surface over a zarrista ``Array`` for dask.

    zarrista exposes ``shape`` as a ``list``, ``dtype`` as its own
    ``DataType``, has no ``.chunks`` attribute, and rejects non-C-contiguous
    or non-native-byte-order write input. dask needs a tuple ``shape``, a
    numpy ``dtype``, ndarray ``__getitem__`` results, and a tolerant
    ``__setitem__`` (see :func:`_native_contiguous`).
    """

    def __init__(self, arr):
        self._arr = arr
        self.shape = tuple(arr.shape)
        self.dtype = np.dtype(arr.dtype.name)
        self.ndim = arr.ndim

    def __getitem__(self, selection):
        return np.asarray(self._arr[selection])

    def __setitem__(self, selection, value):
        self._arr[selection] = _native_contiguous(value)


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


def _evolve_blosc_params(compressor, params: dict, itemsize: int) -> dict:
    """Resolve constructor-defaulted blosc typesize/shuffle from the dtype.

    zarr-python's ``BloscCodec`` serializes unset attrs as ``typesize=1`` /
    ``shuffle="bitshuffle"``, records which were defaulted in
    ``_tunable_attrs``, and evolves those from the dtype at write time.
    Mirror that so the stored metadata matches what zarr-python would write;
    explicitly chosen values are kept.
    """
    params = dict(params)
    tunable = getattr(compressor, "_tunable_attrs", None)
    if tunable is None:
        # Fallback for compressor forms without _tunable_attrs.
        if params.get("typesize") in (None, 1):
            params["typesize"] = itemsize
    else:
        if "typesize" in tunable:
            params["typesize"] = itemsize
        if "shuffle" in tunable:
            params["shuffle"] = "bitshuffle" if itemsize == 1 else "shuffle"
    return params


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
        params = _evolve_blosc_params(compressor, params, dtype.itemsize)
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
        params = _evolve_blosc_params(compressor, params, dtype.itemsize)
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


_ZARRISTA_AVAILABLE: bool | None = None


def _zarrista_available() -> bool:
    global _ZARRISTA_AVAILABLE
    if _ZARRISTA_AVAILABLE is None:
        _ZARRISTA_AVAILABLE = importlib.util.find_spec("zarrista") is not None
    return _ZARRISTA_AVAILABLE


def use_zarrista_for(store) -> bool:
    """Return True when writes targeting *store* should use zarrista.

    This is the single backend-dispatch point: every write site branches
    through it so the zarr-python fallback can later be removed in one place.
    zarrista handles ``str``/``pathlib.Path``/``os.PathLike`` targets; store
    objects and ``MutableMapping``s keep the zarr-python engine, as do
    ``.zip``/``.ozx`` targets (zarrista can only read zip archives). Also
    False when zarrista itself is unavailable (it requires Python >= 3.11).
    """
    if not isinstance(store, (str, Path, os.PathLike)) or isinstance(
        store, MutableMapping
    ):
        return False
    path = Path(os.fspath(store))
    if path.suffix.lower() == ".zip" or is_ozx_path(path):
        return False
    return _zarrista_available()


def resolve_store_path(store) -> Path | None:
    """Resolve *store* to the local directory path backing it, or ``None``.

    Accepts plain ``str``/``pathlib.Path``/``os.PathLike`` targets and the
    directory-backed zarr-python stores (``LocalStore``/``DirectoryStore``,
    e.g. the default ``config.cache_store``). Returns ``None`` for stores
    with no local directory (in-memory mappings, remote stores, ...), which
    must keep the zarr-python engine.
    """
    local_store_cls = getattr(zarr.storage, "LocalStore", None)
    if local_store_cls is not None and isinstance(store, local_store_cls):
        return Path(store.root)
    directory_store_cls = getattr(zarr.storage, "DirectoryStore", None)
    if directory_store_cls is not None and isinstance(store, directory_store_cls):
        return Path(store.path)
    if isinstance(store, (str, Path, os.PathLike)) and not isinstance(
        store, MutableMapping
    ):
        return Path(os.fspath(store))
    return None


def normalize_store(store) -> Path:
    """Normalize *store* to a local directory path zarrista can target.

    Accepts ``str``/``pathlib.Path``/``os.PathLike`` and local zarr-python
    stores that wrap a directory path. Raises ``TypeError`` for store objects
    zarrista cannot handle (in-memory mappings, remote stores, ...) and
    ``ValueError`` for zip targets, which zarrista can only read.
    """
    path = resolve_store_path(store)
    if path is None:
        raise TypeError(
            "The zarrista compatibility layer requires a local path store; "
            f"got {type(store).__name__}. Pass a str, pathlib.Path, or "
            "os.PathLike directory path. In-memory mappings and other zarr "
            "store objects must use the zarr-python engine."
        )
    if path.suffix.lower() == ".zip" or is_ozx_path(path):
        raise ValueError(
            "zarrista cannot write into zip archives (.zip/.ozx). Write to a "
            "directory and zip it afterwards (see ngff_zarr.rfc9_zip), or use "
            "the zarr-python engine."
        )
    return path


def read_group_attributes(store, group_path=None, *, zarr_format: int) -> dict | None:
    """Attributes of the group at *group_path* within *store*.

    Returns ``None`` when no group node exists there and ``{}`` for a group
    with no attributes, so callers can distinguish "absent" from "empty"
    (the compat-layer equivalent of a zarr-python membership test plus
    ``attrs.asdict()``). Raises ``ValueError`` when the path holds an array
    node, mirroring zarr-python's refusal to open a group over an array.
    """
    path = normalize_store(store)
    if group_path is not None:
        parts = [p for p in str(group_path).split("/") if p not in ("", ".")]
        path = path.joinpath(*parts)
    if zarr_format == 2:
        if (path / ".zarray").exists():
            raise ValueError(f"An array node already exists at {path}")
        if not (path / ".zgroup").exists():
            return None
        doc = path / ".zattrs"
        if not doc.exists():
            return {}
        loaded = json.loads(doc.read_text())
        return loaded if isinstance(loaded, dict) else {}
    if zarr_format != 3:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
    doc = path / "zarr.json"
    if not doc.exists():
        return None
    loaded = json.loads(doc.read_text())
    if loaded.get("node_type") != "group":
        raise ValueError(f"An array node already exists at {path}")
    return dict(loaded.get("attributes") or {})


def _existing_group_attributes(path: Path, zarr_format: int) -> dict:
    """Attributes of the group already at *path*, or ``{}`` when absent."""
    attributes = read_group_attributes(path, zarr_format=zarr_format)
    return {} if attributes is None else attributes


def _write_group_doc(path: Path, doc: dict) -> None:
    """Serialize *doc* to *path* atomically (temp file + rename).

    Group documents on shared ancestors (e.g. an HCS row group) can be read
    and rewritten by concurrent well/field writers; the atomic replace keeps
    readers from ever observing a truncated document.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc, indent=4))
    os.replace(tmp, path)


def create_zarrista_group(
    path, attributes: dict | None, zarr_format: int, *, overwrite: bool = False
):
    """Create group metadata at *path* and return the opened zarrista group.

    zarrista has no group-create API, so the metadata documents are written
    directly: ``.zgroup`` + ``.zattrs`` for zarr format 2, ``zarr.json`` for
    format 3. With ``overwrite`` any existing content below *path* is removed
    first, matching ``zarr.open_group(mode="w")``; otherwise an existing
    group's attributes are preserved and updated with *attributes*, matching
    ``mode="a"`` followed by per-key attribute assignment. The returned
    ``zarrista.Group`` reads the documents back, hiding the hand-written JSON
    from callers.
    """
    from zarrista import Group
    from zarrista.store import FilesystemStore

    path = normalize_store(path)
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    attributes = {
        **_existing_group_attributes(path, zarr_format),
        **(attributes or {}),
    }
    if zarr_format == 2:
        _write_group_doc(path / ".zgroup", {"zarr_format": 2})
        _write_group_doc(path / ".zattrs", attributes)
    elif zarr_format == 3:
        metadata = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attributes,
        }
        _write_group_doc(path / "zarr.json", metadata)
    else:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
    return Group.open(FilesystemStore(path), "/")


def create_zarrista_subgroup(store, group_path, attributes: dict | None, zarr_format):
    """Create the group at *group_path* within the store at *store*.

    Missing ancestor groups along *group_path* are created with empty
    attributes, matching zarr-python's implicit-parent creation; an existing
    leaf group keeps its other attributes (see :func:`create_zarrista_group`).
    Returns the zarrista group for the leaf.
    """
    root = normalize_store(store)
    parts = [part for part in str(group_path).split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Invalid group path: {group_path!r}")
    for depth in range(1, len(parts)):
        ancestor = root.joinpath(*parts[:depth])
        # Leave existing ancestors untouched so concurrent writers sharing
        # them (e.g. HCS wells in one row) never rewrite the same document.
        if read_group_attributes(ancestor, zarr_format=zarr_format) is None:
            create_zarrista_group(ancestor, None, zarr_format)
    return create_zarrista_group(root.joinpath(*parts), attributes, zarr_format)


def _normalize_array_metadata_doc(doc_path: Path, zarr_format: int, itemsize: int):
    """Rewrite a zarrista-written array metadata document for engine parity.

    zarrs' serialization differs from zarr-python's in ways that are spec-legal
    but break byte-level store equivalence: it stamps ``node_type`` into v2
    ``.zarray`` docs and a ``_zarrs`` provenance attribute into v3 docs, omits
    the always-materialized ``storage_transformers``/``attributes`` fields, and
    writes an explicit ``endian`` for the ``bytes`` codec even for single-byte
    dtypes (where zarr-python omits the configuration). Only the document is
    rewritten; the already-open zarrista array keeps its in-memory metadata.
    """

    def _strip_bytes_endian(codecs: list) -> None:
        for codec in codecs:
            if not isinstance(codec, dict):
                continue
            if codec.get("name") == "bytes":
                codec.pop("configuration", None)
            elif codec.get("name") == "sharding_indexed":
                # The inner chain encodes the same dtype; the index codecs
                # encode uint64 offsets and keep their endian.
                _strip_bytes_endian(codec.get("configuration", {}).get("codecs", []))

    doc = json.loads(doc_path.read_text())
    if zarr_format == 2:
        doc.pop("node_type", None)
        # zarrs stamps its provenance into a sibling ``.zattrs``; zarr-python
        # writes an empty one.
        zattrs_path = doc_path.with_name(".zattrs")
        zattrs = json.loads(zattrs_path.read_text()) if zattrs_path.exists() else {}
        zattrs.pop("_zarrs", None)
        zattrs_path.write_text(json.dumps(zattrs, indent=4))
    else:
        attributes = dict(doc.get("attributes") or {})
        attributes.pop("_zarrs", None)
        doc["attributes"] = attributes
        doc.setdefault("storage_transformers", [])
        if itemsize == 1:
            _strip_bytes_endian(doc.get("codecs", []))
    doc_path.write_text(json.dumps(doc, indent=4))


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
        _normalize_array_metadata_doc(
            path.joinpath(*node_path.strip("/").split("/"), ".zarray"),
            zarr_format,
            dtype.itemsize,
        )
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
    arr = builder.create(store, node_path)
    _normalize_array_metadata_doc(
        path.joinpath(*node_path.strip("/").split("/"), "zarr.json"),
        zarr_format,
        dtype.itemsize,
    )
    return arr


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
    (what OME-Zarr 0.4 stores carry). Format 3 consolidation is likewise
    hand-inlined into the root ``zarr.json``: routing it through zarrista's
    consolidation API would re-serialize the child documents with zarrs
    conventions (e.g. the bare-string shorthand for configuration-less codecs)
    that zarr-python's reader rejects, so the raw documents are embedded
    verbatim instead, matching ``zarr.consolidate_metadata``.
    """

    def _empty_consolidated() -> dict:
        # zarr-python's consolidation marks every non-root child group entry
        # as (vacuously) consolidated; mirror it so consolidated output is
        # indistinguishable from zarr-python's.
        return {"kind": "inline", "must_understand": False, "metadata": {}}

    path = normalize_store(path)
    if zarr_format == 2:
        metadata = {}
        for doc in sorted(path.rglob(".z*")):
            if doc.name not in (".zgroup", ".zattrs", ".zarray"):
                continue
            key = doc.relative_to(path).as_posix()
            entry = json.loads(doc.read_text())
            if doc.name == ".zarray":
                # zarrs stamps a non-standard node_type into v2 array
                # metadata; zarr-python's consolidation drops it.
                entry.pop("node_type", None)
            elif doc.name == ".zgroup" and key != ".zgroup":
                entry["consolidated_metadata"] = _empty_consolidated()
            metadata[key] = entry
        consolidated = {"metadata": metadata, "zarr_consolidated_format": 1}
        (path / ".zmetadata").write_text(json.dumps(consolidated, indent=4))
    elif zarr_format == 3:
        metadata = {}
        for doc in sorted(path.rglob("zarr.json")):
            if doc == path / "zarr.json":
                continue
            key = doc.parent.relative_to(path).as_posix()
            entry = json.loads(doc.read_text())
            # zarr-python's consolidation re-serializes entries through its
            # metadata model, which always materializes these optional fields.
            entry.setdefault("attributes", {})
            if entry.get("node_type") == "group":
                entry["consolidated_metadata"] = _empty_consolidated()
            else:
                entry.setdefault("storage_transformers", [])
            metadata[key] = entry
        root_doc = json.loads((path / "zarr.json").read_text())
        root_doc["consolidated_metadata"] = {
            "kind": "inline",
            "must_understand": False,
            "metadata": metadata,
        }
        (path / "zarr.json").write_text(json.dumps(root_doc, indent=4))
    else:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
