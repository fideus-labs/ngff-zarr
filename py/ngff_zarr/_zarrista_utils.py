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

zarrista is imported lazily inside each function to keep module import cheap.
"""

import ast
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping, MutableMapping
from pathlib import Path

import dask.array
import numpy as np

from ._v2_store_reader import AttrsDict
from .rfc9_zip import ZipReadStore, is_ozx_path

__all__ = [
    "LocalZarrArray",
    "LocalZarrGroup",
    "consolidate_metadata",
    "create_zarrista_array",
    "create_zarrista_group",
    "create_zarrista_subgroup",
    "has_consolidated_metadata",
    "is_zarrista_array",
    "normalize_store",
    "open_lazy_array",
    "open_local_node",
    "open_ozx_store",
    "open_zarrista_array",
    "open_zarrista_lazy",
    "open_zip_lazy",
    "read_group_attributes",
    "resolve_store_path",
    "write_dask_array",
    "zarrista_array_to_dask",
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

    def _fetch(self, selection) -> np.ndarray:
        """Materialize *selection*; the async remote adapter overrides this."""
        return np.asarray(self._arr[selection])

    def __getitem__(self, selection):
        # zarrista treats an integer index like a length-1 slice, keeping the
        # axis; dask's fused getters rely on numpy semantics where integer
        # indices drop it. Normalize integers to slices and squeeze after.
        if not isinstance(selection, tuple):
            selection = (selection,)
        if Ellipsis in selection:
            explicit = [index for index in selection if index is not Ellipsis]
            fill = (slice(None),) * (self.ndim - len(explicit))
            at = selection.index(Ellipsis)
            selection = selection[:at] + fill + selection[at + 1 :]
            selection = tuple(index for index in selection if index is not Ellipsis)
        drop_axes = []
        normalized = []
        for axis, index in enumerate(selection):
            if isinstance(index, (int, np.integer)):
                index = int(index)
                if index < 0:
                    index += self.shape[axis]
                normalized.append(slice(index, index + 1))
                drop_axes.append(axis)
            else:
                normalized.append(index)
        result = self._fetch(tuple(normalized))
        if drop_axes:
            result = np.squeeze(result, axis=tuple(drop_axes))
        return result

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


def _is_local_path(store) -> bool:
    """Whether *store* is a local directory path zarrista can target directly.

    True for ``str``/``pathlib.Path``/``os.PathLike`` targets that are neither
    remote URL strings nor ``.zip``/``.ozx`` archive paths (zarrista can only
    read zip archives); ``MutableMapping``s and store objects are excluded.
    """
    if not isinstance(store, (str, Path, os.PathLike)) or isinstance(
        store, MutableMapping
    ):
        return False
    target = os.fspath(store)
    if isinstance(target, str):
        # Imported lazily: from_ngff_zarr must stay importable without pulling
        # in the write stack this module belongs to.
        from .from_ngff_zarr import REMOTE_URL_SCHEMES

        if target.startswith(REMOTE_URL_SCHEMES):
            return False
    path = Path(target)
    return not (path.suffix.lower() == ".zip" or is_ozx_path(path))


_FILESYSTEM_STORE_REPR = re.compile(r"\AFilesystemStore\(path=(.*)\)\Z", re.DOTALL)


def _zarrista_filesystem_store_path(store) -> Path | None:
    """Root path of a zarrista ``FilesystemStore``, or ``None`` otherwise.

    zarrista's stores are pyo3 classes with no Python-visible attributes; the
    only channel for the root path is the round-trippable ``repr``
    (``FilesystemStore(path='...')``, quoted per Python's string ``repr``),
    parsed with ``ast.literal_eval``. Tests pin the format so upstream drift
    surfaces as a failure rather than a silently unresolvable store.
    """
    from zarrista.store import FilesystemStore

    if not isinstance(store, FilesystemStore):
        return None
    match = _FILESYSTEM_STORE_REPR.match(repr(store))
    if match is None:
        return None
    try:
        path = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    return Path(path) if isinstance(path, str) else None


def resolve_store_path(store) -> Path | None:
    """Resolve *store* to the local directory path backing it, or ``None``.

    Accepts plain ``str``/``pathlib.Path``/``os.PathLike`` targets (e.g. the
    default ``config.cache_store``) and zarrista ``FilesystemStore`` handles.
    Returns ``None`` for anything with no local directory (in-memory mappings,
    remote URL strings, store objects, ...).
    """
    zarrista_path = _zarrista_filesystem_store_path(store)
    if zarrista_path is not None:
        return zarrista_path
    if isinstance(store, (str, Path, os.PathLike)) and not isinstance(
        store, MutableMapping
    ):
        target = os.fspath(store)
        if isinstance(target, str):
            from .from_ngff_zarr import REMOTE_URL_SCHEMES

            if target.startswith(REMOTE_URL_SCHEMES):
                return None
        return Path(target)
    return None


def normalize_store(store) -> Path:
    """Normalize *store* to a local directory path zarrista can target.

    Accepts ``str``/``pathlib.Path``/``os.PathLike`` directory paths and
    zarrista ``FilesystemStore`` handles. Raises ``TypeError`` for anything
    else (in-memory mappings, remote URL strings, zarr store objects, ...)
    and ``ValueError`` for zip targets, which zarrista can only read.
    """
    path = resolve_store_path(store)
    if path is None:
        raise TypeError(
            "ngff-zarr writes to local directory paths; got "
            f"{type(store).__name__}. Pass a str, pathlib.Path, or "
            "os.PathLike directory path (write to a local directory and "
            "upload afterwards for remote targets). In-memory mappings and "
            "zarr-python store objects are no longer accepted."
        )
    if path.suffix.lower() == ".zip" or is_ozx_path(path):
        raise ValueError(
            "zarrista cannot write into zip archives (.zip/.ozx). Write to a "
            "directory and zip it afterwards (see ngff_zarr.rfc9_zip)."
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


def has_consolidated_metadata(store, zarr_format: int) -> bool:
    """Whether the store at *store* carries consolidated metadata.

    Checks the root ``zarr.json`` for an inline ``consolidated_metadata``
    block (zarr format 3) or for a ``.zmetadata`` sidecar (format 2), so
    in-place metadata rewrites can refresh consolidation exactly when
    zarr-python would.
    """
    path = normalize_store(store)
    if zarr_format == 2:
        return (path / ".zmetadata").exists()
    if zarr_format != 3:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
    doc = path / "zarr.json"
    if not doc.exists():
        return False
    return "consolidated_metadata" in json.loads(doc.read_text())


def _existing_group_attributes(path: Path, zarr_format: int) -> dict:
    """Attributes of the group already at *path*, or ``{}`` when absent."""
    attributes = read_group_attributes(path, zarr_format=zarr_format)
    return {} if attributes is None else attributes


def _write_group_doc(path: Path, doc: dict) -> None:
    """Serialize *doc* to *path* atomically (temp file + rename).

    Group documents on shared ancestors (e.g. an HCS row group) can be read
    and rewritten by concurrent well/field writers; the atomic replace keeps
    readers from ever observing a truncated document.

    On Windows ``os.replace`` raises ``PermissionError`` when another writer
    or reader momentarily holds the destination open, so the replace is
    retried briefly. POSIX renames onto an open file without complaint.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc, indent=4))
    delay = 0.001
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.05)


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
    fill_value=None,
    attributes: dict | None = None,
    chunk_key_encoding: dict | None = None,
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
    explicit outer shard shape whose subchunks are *chunks*. *fill_value*
    ``None`` means the dtype's zero, matching zarr-python. *attributes* are
    the array's user attributes. *chunk_key_encoding* (format 3 only) is the
    zarr v3 metadata form (e.g. ``{"name": "v2", "configuration":
    {"separator": "/"}}``), used by in-place v2->v3 upgrades to keep existing
    chunk keys resolvable. The metadata is stored immediately; the returned
    ``zarrista.Array`` is ready for chunk writes.
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
    fill_scalar = (
        np.zeros((), dtype=dtype)
        if fill_value is None
        else np.asarray(fill_value, dtype=dtype)
    )

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
        if chunk_key_encoding is not None:
            raise ValueError(
                "chunk_key_encoding requires zarr format 3; pass "
                "dimension_separator for zarr format 2"
            )
        metadata = {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": dtype.str,
            "compressor": _compressor_to_v2_config(compressor, dtype),
            "fill_value": fill_scalar.item(),
            "order": "C",
            "filters": None,
            "dimension_separator": dimension_separator,
        }
        arr = Array.from_metadata(metadata, store, node_path)
        arr.store_metadata()
        node_dir = path.joinpath(*node_path.strip("/").split("/"))
        _normalize_array_metadata_doc(node_dir / ".zarray", zarr_format, dtype.itemsize)
        if attributes:
            zattrs_path = node_dir / ".zattrs"
            merged = {**json.loads(zattrs_path.read_text()), **attributes}
            zattrs_path.write_text(json.dumps(merged, indent=4))
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
    builder = ArrayBuilder(
        grid, data_type, FillValue(_native_contiguous(fill_scalar).tobytes())
    )
    if chunks_per_shard is not None or shards is not None:
        builder = builder.subchunk_shape(list(chunks))
    if chunk_key_encoding is not None:
        from zarrista import ChunkKeyEncoding

        builder = builder.chunk_key_encoding(
            ChunkKeyEncoding.from_metadata(chunk_key_encoding)
        )
    if attributes:
        builder = builder.attrs(dict(attributes))
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


def is_zarrista_array(obj) -> bool:
    """Whether *obj* is a raw ``zarrista.Array`` handle."""
    from zarrista import Array

    return isinstance(obj, Array)


def zarrista_array_to_dask(arr) -> dask.array.Array:
    """Wrap zarrista ``Array`` *arr* as a lazy dask array.

    Chunks follow the stored chunk grid, using the subchunk (inner chunk)
    shape for sharded arrays so reads stay at the efficient granularity.
    """
    if arr.is_sharded:
        chunks = tuple(arr.subchunk_shape)
    else:
        chunks = tuple(arr.chunk_shape([0] * arr.ndim))
    return dask.array.from_array(_ZarristaArrayAdapter(arr), chunks=chunks)


def open_zarrista_lazy(path, component: str | None = None) -> dask.array.Array:
    """Open the array at *component* within *path* as a lazy dask array."""
    return zarrista_array_to_dask(open_zarrista_array(path, component))


def open_ozx_store(path):
    """Store handle for reading the zip archive (``.ozx``/``.zip``) at *path*.

    A :class:`~.rfc9_zip.ZipReadStore`: metadata and group navigation go
    through the pure-Python mapping reader while pixel data reads through
    zarrista's native zip store (see :func:`open_zip_lazy`).
    """
    return ZipReadStore(path)


def open_zip_lazy(store: ZipReadStore, component: str | None = None):
    """Open the array at *component* within zip-archive *store* lazily.

    Routes through zarrista's read-only zip store rather than the mapping
    reader because ``.ozx`` archives hold sharded zarr v3 arrays by default,
    which only zarrista can decode. The node path combines the store's
    prefix (HCS well/field views) with *component*.
    """
    from zarrista import Array
    from zarrista.store import FilesystemStore, ZipStore

    zip_store = ZipStore(FilesystemStore(store.path.parent), store.path.name)
    parts = [
        part
        for source in (store.prefix, component)
        if source
        for part in str(source).split("/")
        if part not in ("", ".")
    ]
    return zarrista_array_to_dask(Array.open(zip_store, "/" + "/".join(parts)))


def _is_bytes_mapping(store) -> bool:
    """Whether *store* is a key-to-bytes mapping for the pure-Python reader.

    ``str``/``Path`` are excluded even though paths are iterable: only real
    mapping objects (plain dicts, zip-archive views, tifffile's aszarr store)
    go through :mod:`._v2_store_reader`.
    """
    return isinstance(store, Mapping) and not isinstance(
        store, (str, Path, os.PathLike)
    )


def open_lazy_array(store, component: str | None = None) -> dask.array.Array:
    """Open the array at *component* within *store* as a lazy dask array.

    The single read-dispatch point: zip-archive views read pixels through
    zarrista's zip store, remote store handles through zarrista's async API
    over obstore, other bytes mappings go through the pure-Python store
    reader (either zarr format), and local paths through zarrista. Any other
    store object raises ``TypeError``.
    """
    if isinstance(store, ZipReadStore):
        return open_zip_lazy(store, component)
    from ._remote_reader import RemoteZarrStore, open_remote_lazy

    if isinstance(store, RemoteZarrStore):
        return open_remote_lazy(store, component)
    if _is_bytes_mapping(store):
        from ._v2_store_reader import open_store_array

        return open_store_array(store, component).to_dask()
    if _is_local_path(store):
        return open_zarrista_lazy(store, component)
    raise TypeError(
        "ngff-zarr reads from local directory paths, remote URL strings, "
        f"zip archive paths, and key-to-bytes mappings; got "
        f"{type(store).__name__}. Pass a path instead of a zarr-python "
        "store object."
    )


def _load_local_doc(doc_path: Path) -> dict:
    loaded = json.loads(doc_path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _local_node_document(dirpath: Path, zarr_format: int) -> tuple[str, dict] | None:
    """``(node_type, metadata document)`` for the node at *dirpath*.

    Returns ``None`` when no node of *zarr_format* exists there. For format 2
    an array wins when both ``.zarray`` and ``.zgroup`` are present, matching
    zarr-python.
    """
    if zarr_format == 3:
        doc_path = dirpath / "zarr.json"
        if not doc_path.exists():
            return None
        doc = _load_local_doc(doc_path)
        node_type = doc.get("node_type")
        return (node_type, doc) if node_type in ("array", "group") else None
    if zarr_format != 2:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")
    if (dirpath / ".zarray").exists():
        return "array", _load_local_doc(dirpath / ".zarray")
    if (dirpath / ".zgroup").exists():
        return "group", _load_local_doc(dirpath / ".zgroup")
    return None


def _local_node_attrs(dirpath: Path, doc: dict, zarr_format: int) -> AttrsDict:
    """User attributes for the node at *dirpath* (``.zattrs`` or inline)."""
    if zarr_format == 2:
        attrs_path = dirpath / ".zattrs"
        attrs = _load_local_doc(attrs_path) if attrs_path.exists() else {}
    else:
        attrs = doc.get("attributes") or {}
    return AttrsDict(attrs if isinstance(attrs, dict) else {})


class LocalZarrArray:
    """Read-only handle for an array node within a local directory store.

    Exposes ``shape``/``attrs`` for node inspection and :meth:`to_dask` for
    lazy pixel access through zarrista.
    """

    def __init__(self, store_root: Path, path: str, zarr_format: int, doc: dict):
        self._store_root = store_root
        self.path = path
        self.zarr_format = zarr_format
        self.shape = tuple(int(s) for s in doc.get("shape", ()))
        self.ndim = len(self.shape)
        dirpath = store_root.joinpath(*path.split("/")) if path else store_root
        self.attrs = _local_node_attrs(dirpath, doc, zarr_format)

    def to_dask(self) -> dask.array.Array:
        return open_zarrista_lazy(self._store_root, self.path or None)


class LocalZarrGroup:
    """Read-only handle for a group node within a local directory store.

    Provides the slice of the zarr-python ``Group`` surface the read path
    uses -- ``attrs`` (a dict with ``asdict()``), ``__contains__`` /
    ``__getitem__`` path navigation, and ``keys()`` -- backed by direct
    metadata-document reads (``.zgroup``/``.zattrs`` for zarr format 2,
    ``zarr.json`` for format 3). Child arrays open their pixel data lazily
    through zarrista.
    """

    def __init__(self, store_root: Path, path: str, zarr_format: int, doc: dict):
        self._store_root = store_root
        self.path = path
        self.zarr_format = zarr_format
        dirpath = store_root.joinpath(*path.split("/")) if path else store_root
        self._dirpath = dirpath
        self.attrs = _local_node_attrs(dirpath, doc, zarr_format)

    def _child_path(self, name) -> str:
        child = "/".join(p for p in str(name).split("/") if p not in ("", "."))
        return f"{self.path}/{child}" if self.path else child

    def __getitem__(self, name):
        child_path = self._child_path(name)
        node = open_local_node(
            self._store_root, child_path, zarr_format=self.zarr_format
        )
        if node is None:
            raise KeyError(child_path)
        return node

    def __contains__(self, name) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True

    def keys(self) -> list[str]:
        """Names of all immediate child nodes (arrays and groups), sorted."""
        if not self._dirpath.is_dir():
            return []
        return sorted(
            child.name
            for child in self._dirpath.iterdir()
            if child.is_dir()
            and _local_node_document(child, self.zarr_format) is not None
        )


def open_local_node(store, node_path=None, *, zarr_format: int):
    """Open the node at *node_path* within the local directory store *store*.

    The compat-layer read equivalent of ``zarr.open``: metadata documents are
    read directly from the filesystem, no zarr-python involved. Returns a
    :class:`LocalZarrGroup` or :class:`LocalZarrArray`, or ``None`` when no
    node of *zarr_format* exists at that path.
    """
    root = normalize_store(store)
    parts = [p for p in str(node_path or "").split("/") if p not in ("", ".")]
    path = "/".join(parts)
    found = _local_node_document(root.joinpath(*parts), zarr_format)
    if found is None:
        return None
    node_type, doc = found
    if node_type == "array":
        return LocalZarrArray(root, path, zarr_format, doc)
    return LocalZarrGroup(root, path, zarr_format, doc)


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
