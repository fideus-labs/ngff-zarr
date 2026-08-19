# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import copy
import os
import sys
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .methods._metadata import get_method_metadata

if sys.version_info < (3, 10):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

import dask.array
import numpy as np
import zarr
import zarr.storage
from dask import __version__ as dask_version
from itkwasm import array_like_to_numpy_array
from packaging.version import Version

from ._store_types import StoreLike
from ._supported_versions import NgffVersion
from ._zarrista_utils import (
    consolidate_metadata as _zarrista_consolidate_metadata,
)
from ._zarrista_utils import (
    create_zarrista_group,
    create_zarrista_subgroup,
    use_zarrista_for,
)
from .config import config
from .memory_usage import memory_usage
from .methods._support import _dim_scale_factors
from .multiscales import NgffMultiscales
from .rfc9_zip import is_ozx_path, write_store_to_zip
from .rich_dask_progress import NgffProgress, NgffProgressCallback
from .to_multiscales import to_multiscales
from .v04.zarr_metadata import Metadata as Metadata_v04
from .v05.zarr_metadata import Metadata as Metadata_v05

zarr_version = Version(zarr.__version__)
IS_ZARR_V3_PLUS = zarr_version.major >= 3
DASK_SUPPORTS_SHARDING = Version(dask_version) >= Version("2025.12.0")

# Legacy zarr-python engine fallback (store objects / MutableMapping targets);
# removed together with the ``use_zarrista`` False-branches.
if IS_ZARR_V3_PLUS:
    from zarr.api.synchronous import open_array

    zarr_kwargs = {"chunk_key_encoding": {"name": "default", "separator": "/"}}
else:
    from zarr.creation import open_array

    zarr_kwargs = {"dimension_separator": "/"}

# Detect whether dask.array.to_zarr uses Group.create_array() (which rejects
# ``zarr_format`` but inherits it from the group) or top-level
# ``zarr.create_array()`` (which needs ``zarr_format`` to avoid defaulting to
# format 3).  Changed in dask ~2026.3.0.
_DASK_USES_GROUP_CREATE = False
if DASK_SUPPORTS_SHARDING:
    import inspect as _inspect
    import re as _re

    _src = _inspect.getsource(dask.array.core.to_zarr)
    _DASK_USES_GROUP_CREATE = bool(_re.search(r"root\s*=\s*zarr\.open_group", _src))

ScaleStrategy = Literal["pad", "exact"]


def _numcodecs_to_zarr_v3_codec(compressor):
    """Translate a *numcodecs* compressor to its zarr v3 codec equivalent.

    Returns a zarr v3 codec object suitable for use inside a
    :class:`~zarr.codecs.sharding.ShardingCodec` or as a standalone
    bytes-to-bytes codec.  Returns ``None`` when *compressor* is ``None``
    or the codec type is not recognised.
    """
    if compressor is None:
        return None

    codec_id = getattr(compressor, "codec_id", None)
    if codec_id is None:
        # Already a native Zarr v3 codec: pass it through unchanged. The
        # sharding path otherwise swaps it for the zstd default.
        if hasattr(compressor, "to_dict"):
            return compressor
        return None

    try:
        if codec_id == "blosc":
            from zarr.codecs.blosc import BloscCodec

            shuffle_val = getattr(compressor, "shuffle", 1)
            if shuffle_val == 0:
                from zarr.codecs.blosc import BloscShuffle

                shuffle = BloscShuffle.noshuffle
            elif shuffle_val == 2:
                from zarr.codecs.blosc import BloscShuffle

                shuffle = BloscShuffle.bitshuffle
            else:
                from zarr.codecs.blosc import BloscShuffle

                shuffle = BloscShuffle.shuffle
            return BloscCodec(
                cname=getattr(compressor, "cname", "lz4"),
                clevel=getattr(compressor, "clevel", 5),
                shuffle=shuffle,
            )
        if codec_id == "gzip":
            from zarr.codecs.gzip import GzipCodec

            return GzipCodec(level=getattr(compressor, "level", 6))
        if codec_id == "zstd":
            from zarr.codecs.zstd import ZstdCodec

            return ZstdCodec(level=getattr(compressor, "level", 3))
    except ImportError:
        pass

    return None


def _remove_none_values(obj: Any) -> Any:
    """Recursively strip ``None``-valued keys from nested dicts and lists.

    In the OME-Zarr spec the absence of a field carries no significance: only
    fields that are present are meaningful, and an explicit ``null`` is treated
    as equivalent to omitting the field. Culling every ``None`` therefore keeps
    the written metadata clean without enumerating optional fields per spec
    version. Only ``None`` is removed; falsy-but-valid values such as ``0``,
    ``0.0``, ``""`` and empty collections are preserved.
    """
    if isinstance(obj, dict):
        return {
            key: _remove_none_values(value)
            for key, value in obj.items()
            if value is not None
        }
    if isinstance(obj, list):
        return [_remove_none_values(item) for item in obj]
    return obj


def _pop_metadata_optionals(metadata_dict: dict) -> dict:
    """Strip optional metadata fields that must not be serialized.

    Recursively culls all ``None``-valued keys. Axes that carry a populated
    RFC 4 ``orientation`` keep it -- anatomical orientation is written whenever
    it is present -- while axes whose ``orientation`` is ``None`` have the key
    dropped by the ``None`` stripping below.
    """
    # ``Metadata.extra`` is a read-side validation aid (leaked-key capture for
    # the v0.5 namespacing rules), never part of the written multiscale entry.
    # It is an empty dict on every clean read, which ``_remove_none_values``
    # would not strip, so drop it explicitly.
    metadata_dict.pop("extra", None)

    return _remove_none_values(metadata_dict)


def _prep_for_to_zarr(store: StoreLike, arr: dask.array.Array) -> dask.array.Array:
    try:
        importlib_metadata.distribution("kvikio")
        _KVIKIO_AVAILABLE = True
    except importlib_metadata.PackageNotFoundError:
        _KVIKIO_AVAILABLE = False

    if _KVIKIO_AVAILABLE:
        from kvikio.zarr import GDSStore

        if not isinstance(store, GDSStore):
            arr = dask.array.map_blocks(
                array_like_to_numpy_array,
                arr,
                dtype=arr.dtype,
                meta=np.empty(()),
            )
        return arr
    return dask.array.map_blocks(
        array_like_to_numpy_array, arr, dtype=arr.dtype, meta=np.empty(())
    )


def _numpy_to_zarr_dtype(dtype):
    dtype_map = {
        "bool": "bool",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
        "float16": "float16",
        "float32": "float32",
        "float64": "float64",
        "complex64": "complex64",
        "complex128": "complex128",
    }

    # ``np.dtype(...).name`` yields the canonical, byte-order-independent name
    # (e.g. ">u2", "<u2", and "uint16" all resolve to "uint16"), so big-endian
    # and other non-native dtypes map correctly.  ``str(dtype)`` would instead
    # surface a byte-order-prefixed short code such as ">u2" for non-native
    # dtypes, whose stripped form ("u2") is absent from ``dtype_map``.
    dtype_name = np.dtype(dtype).name

    # Look up corresponding zarr dtype
    try:
        return dtype_map[dtype_name]
    except KeyError:
        raise ValueError(f"dtype {dtype} cannot be mapped to Zarr v3 core dtype")


def _write_with_zarrista(
    store_path: str,
    array,
    region,
    chunks,
    zarr_format,
    dimension_names=None,
    internal_chunk_shape=None,
    full_array_shape=None,
    create_dataset=True,
    compression_chain=None,
    **kwargs,
) -> None:
    """Write array using the zarrista backend.

    *store_path* addresses the array node itself (``<store>/<path>``).
    ``compression_chain`` carries the ordered compression codec chain (for
    zarr format 2 metadata only its first entry is stored): ``None`` requests
    the engine default, an empty sequence requests no compression.
    """
    from ._zarrista_utils import (
        _native_contiguous,
        create_zarrista_array,
        open_zarrista_array,
        write_dask_array,
    )

    # Use full array shape if provided, otherwise use the region array shape
    dataset_shape = tuple(
        full_array_shape if full_array_shape is not None else array.shape
    )
    scale_path = Path(store_path)
    store_root = scale_path.parent
    node_name = scale_path.name

    if create_dataset:
        create_kwargs = {}
        if zarr_format == 2:
            if compression_chain is None:
                # Mirror zarr-python's zarr format 2 default compressor so the
                # zarrista engine's output is indistinguishable from what the
                # legacy engine wrote.
                compressor = {"id": "zstd", "level": 0}
            elif compression_chain:
                compressor = compression_chain[0]
            else:
                compressor = None
            create_kwargs["compressor"] = compressor
        elif zarr_format == 3:
            default_compression = {
                "name": "zstd",
                "configuration": {"level": 0, "checksum": False},
            }
            if compression_chain is None:
                # Mirror the zarr-python default codec chain.
                create_kwargs["compressors"] = [default_compression]
            else:
                # An explicit chain is preserved in order; an explicit empty
                # chain means no compression, matching zarr-python.
                create_kwargs["compressors"] = list(compression_chain)
            if dimension_names:
                create_kwargs["dimension_names"] = tuple(dimension_names)
            if internal_chunk_shape:
                # ``chunks`` is the shard (outer chunk) grid; the subchunks
                # inside each shard are the internal chunk shape.
                create_kwargs["shards"] = tuple(chunks)
                chunks = tuple(internal_chunk_shape)
        else:
            raise ValueError(f"Unsupported zarr format: {zarr_format}")

        # Try to create the dataset first, open existing only if needed
        try:
            dataset = create_zarrista_array(
                store_root,
                node_name,
                dataset_shape,
                array.dtype,
                chunks,
                zarr_format,
                **create_kwargs,
            )
        except Exception as create_error:
            # Dataset may already exist: fall back to opening it, surfacing
            # the create failure when there is nothing to open.
            try:
                dataset = open_zarrista_array(store_root, node_name)
            except Exception:
                raise create_error from None
    else:
        dataset = open_zarrista_array(store_root, node_name)

    # Try to write the dask array directly first
    try:
        write_dask_array(array, dataset, region=region)
    except Exception as e:
        # If we encounter dimension mismatch or shape-related errors,
        # compute the array and try again with corrective action
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in [
                "dimension",
                "shape",
                "mismatch",
                "size",
                "extent",
                "rank",
                "invalid",
            ]
        ):
            # Compute the array to get the actual shape
            computed_array = _native_contiguous(array)

            # Adjust region to match the actual computed array shape if needed
            if len(region) == len(computed_array.shape):
                adjusted_region = tuple(
                    slice(
                        region[i].start or 0,
                        (region[i].start or 0) + computed_array.shape[i],
                    )
                    if isinstance(region[i], slice)
                    else region[i]
                    for i in range(len(region))
                )
            else:
                adjusted_region = region

            # Try writing the computed array with adjusted region
            dataset[adjusted_region] = computed_array
        else:
            # Re-raise the exception if it's not related to dimension/shape issues
            raise


def _validate_ngff_parameters(
    version: str | NgffVersion,
    chunks_per_shard: int | tuple[int, ...] | dict[str, int] | None,
    use_tensorstore: bool,
    store: StoreLike,
) -> None:
    """Validate the parameters for the NGFF Zarr generation."""
    if isinstance(version, str):
        version = NgffVersion(version)

    if version not in [NgffVersion.V04, NgffVersion.V05, NgffVersion.V06]:
        raise ValueError(f"Unsupported version: {version}")

    if chunks_per_shard is not None:
        if version == NgffVersion.V04:
            raise ValueError(
                "Sharding is only supported for OME-Zarr version 0.5 and later"
            )
        if not use_tensorstore and not IS_ZARR_V3_PLUS:
            raise ValueError(
                "Sharding requires zarr-python version >= 3.0.0b1 for OME-Zarr version >= 0.5"
            )

    if use_tensorstore and not isinstance(store, (str, Path)):
        raise ValueError(
            "The zarrista writer (use_tensorstore=True) currently requires a "
            "path-like store"
        )


def _prepare_metadata(
    multiscales: NgffMultiscales, version: str
) -> tuple[Metadata_v04 | Metadata_v05, tuple[str, ...], dict]:
    """Prepare and convert metadata to the proper version format."""
    metadata = multiscales.metadata

    # Convert method enum to lowercase string for the type field
    method_type = None
    method_metadata = None
    if multiscales.method is not None:
        method_type = multiscales.method.value
        method_metadata = get_method_metadata(multiscales.method)

    metadata = metadata.to_version(version)
    metadata.type = method_type
    metadata.metadata = method_metadata

    dimension_names = metadata.dimension_names
    dimension_names_kwargs = (
        {"dimension_names": dimension_names} if version != "0.4" else {}
    )

    return metadata, dimension_names, dimension_names_kwargs


def _root_ome_attrs(metadata_dict: dict, version: str) -> dict:
    """Build the OME-Zarr root-group attribute entries for ``metadata_dict``.

    Returns the ``ome``/``multiscales`` attribute mapping (hoisting ``omero``
    to its version-specific location) exactly as the writer persists it --
    including mapping the API version ``"0.6"`` to the ``"0.6.dev4"`` string
    stored on disk. ``metadata_dict`` is mutated in place: its ``omero`` entry
    is popped so it lives only in its hoisted location, matching historical
    behavior.
    """
    if version != "0.4":
        # RFC 2, Zarr 3 - omero goes inside ome namespace
        if version == "0.6":
            version = "0.6.dev4"
        ome_dict = {"version": version, "multiscales": [metadata_dict]}
        if "omero" in metadata_dict:
            ome_dict["omero"] = metadata_dict.pop("omero")
        return {"ome": ome_dict}
    # v0.4 - omero is at root level
    attrs = {}
    if "omero" in metadata_dict:
        attrs["omero"] = metadata_dict.pop("omero")
    attrs["multiscales"] = [metadata_dict]
    return attrs


def _write_root_ome_attrs(root: zarr.Group, metadata_dict: dict, version: str) -> None:
    """Serialize the OME-Zarr root-group attributes for ``metadata_dict``.

    Sets the :func:`_root_ome_attrs` entries on an already-open
    ``zarr.Group``. Shared by :func:`_create_zarr_root` and
    :func:`ngff_zarr.upgrade_ome_zarr` so both emit byte-identical root
    metadata.
    """
    for key, value in _root_ome_attrs(metadata_dict, version).items():
        root.attrs[key] = value


def _create_zarr_root(
    store: StoreLike,
    chunk_store: StoreLike | None,
    version: str,
    overwrite: bool,
    metadata_dict: dict,
    use_zarrista: bool = False,
):
    """Create and configure the root Zarr group with proper attributes.

    With ``use_zarrista`` the group metadata is written through the zarrista
    compatibility layer and the returned handle is a ``zarrista.Group``;
    otherwise it is a ``zarr.Group`` opened by zarr-python.
    """
    zarr_format = 2 if version == "0.4" else 3
    if version != "0.4" and not IS_ZARR_V3_PLUS:
        raise ValueError(
            "zarr-python version >= 3.0.0b2 required for OME-Zarr version >= 0.5"
        )

    if use_zarrista:
        return create_zarrista_group(
            store,
            _root_ome_attrs(metadata_dict, version),
            zarr_format,
            overwrite=overwrite,
        )

    format_kwargs = {"zarr_format": zarr_format} if IS_ZARR_V3_PLUS else {}
    root = zarr.open_group(
        store,
        mode="w" if overwrite else "a",
        chunk_store=chunk_store,
        **format_kwargs,
    )
    _write_root_ome_attrs(root, metadata_dict, version)

    return root


def _configure_sharding(
    arr: dask.array.Array,
    chunks_per_shard: int | tuple[int, ...] | dict[str, int] | None,
    dims: tuple[str, ...],
    kwargs: dict,
) -> tuple[dict, tuple[int, ...] | None, dask.array.Array]:
    """Configure sharding parameters if sharding is enabled."""
    if chunks_per_shard is None:
        return {}, None, arr

    c0 = tuple([c[0] for c in arr.chunks])

    if isinstance(chunks_per_shard, int):
        shards = tuple([c * chunks_per_shard for c in c0])
    elif isinstance(chunks_per_shard, (tuple, list)):
        if len(chunks_per_shard) != arr.ndim:
            raise ValueError(f"chunks_per_shard must be a tuple of length {arr.ndim}")
        shards = tuple([c * c0[i] for i, c in enumerate(chunks_per_shard)])
    elif isinstance(chunks_per_shard, dict):
        shards = {d: c * chunks_per_shard.get(d, 1) for d, c in zip(dims, c0)}
        shards = tuple([shards[d] for d in dims])
    else:
        raise ValueError("chunks_per_shard must be an int, tuple, or dict")

    internal_chunk_shape = c0
    arr = arr.rechunk(shards)

    # Configure sharding parameters differently for v2 vs v3
    sharding_kwargs = {}
    if IS_ZARR_V3_PLUS:
        # For Zarr v3, configure sharding as a codec
        # Use chunk_shape for internal chunks and configure sharding via codecs
        sharding_kwargs["chunk_shape"] = internal_chunk_shape
        # Note: sharding codec will be configured separately in the codecs parameter
        # We'll pass the shard shape through a separate key to be handled later
        sharding_kwargs["_shard_shape"] = shards
    else:
        # For zarr v2, use the older API
        sharding_kwargs = {
            "shards": shards,
            "chunks": internal_chunk_shape,
        }

    return sharding_kwargs, internal_chunk_shape, arr


def _is_bytes_codec(codec) -> bool:
    """Is this the zarr v3 array-to-bytes codec rather than a compressor?"""
    if isinstance(codec, str):
        return codec == "bytes"
    name = getattr(codec, "name", None)
    if name is None and hasattr(codec, "to_dict"):
        name = codec.to_dict().get("name")
    if name is None and isinstance(codec, dict):
        name = codec.get("name")
    return name == "bytes"


def _write_array_with_zarrista(
    store_path: str,
    path: str,
    arr: dask.array.Array,
    chunks: tuple[int, ...] | list[int],
    shards: tuple[int, ...] | None,
    internal_chunk_shape: tuple[int, ...] | None,
    zarr_format: int,
    dimension_names: tuple[str, ...] | None,
    region: tuple[slice, ...],
    full_array_shape: tuple[int, ...] | None = None,
    create_dataset: bool = True,
    **kwargs,
) -> None:
    """Write an array using the zarrista backend."""
    # Translate the public compression kwargs into a single ordered chain:
    # ``None`` selects the engine default and an empty chain selects no
    # compression. Mirroring the legacy engine's kwargs handling, an explicit
    # ``compressor=None`` disables compression while ``compressors=None`` is
    # indistinguishable from unset (default compression); the unset cases are
    # detected with a sentinel. Only the array-to-bytes ("bytes") codec is
    # dropped from a supplied chain since the writer always leads with one.
    filters = kwargs.pop("filters", None)
    if filters:
        raise ValueError(
            "filters are not supported by the zarrista write engine; pass a "
            "zarr store object as the target to use the zarr-python engine"
        )
    _unset = object()
    compressor = kwargs.pop("compressor", _unset)
    compressors = kwargs.pop("compressors", None)
    if compressors is not None:
        if isinstance(compressors, (list, tuple)):
            compression_chain = [c for c in compressors if not _is_bytes_codec(c)]
        else:
            compression_chain = [compressors]
    elif compressor is not _unset:
        compression_chain = [] if compressor is None else [compressor]
    else:
        compression_chain = None
    if zarr_format == 3 and compression_chain:
        # numcodecs compressor objects go through the same zarr v3 codec
        # conversion as the legacy engine so both write identical metadata.
        compression_chain = [
            (_numcodecs_to_zarr_v3_codec(codec) or codec)
            if getattr(codec, "codec_id", None) is not None
            else codec
            for codec in compression_chain
        ]
    kwargs.pop("chunks", None)  # Remove chunks from kwargs since it's a positional arg

    scale_path = f"{store_path}/{path}"
    if shards is None:
        _write_with_zarrista(
            scale_path,
            arr,
            region,
            chunks,
            zarr_format=zarr_format,
            dimension_names=dimension_names,
            full_array_shape=full_array_shape,
            create_dataset=create_dataset,
            compression_chain=compression_chain,
            **kwargs,
        )
    else:  # Sharding
        _write_with_zarrista(
            scale_path,
            arr,
            region,
            chunks,
            zarr_format=zarr_format,
            dimension_names=dimension_names,
            internal_chunk_shape=internal_chunk_shape,
            full_array_shape=full_array_shape,
            create_dataset=create_dataset,
            compression_chain=compression_chain,
            **kwargs,
        )


def _prepare_zarr_kwargs(to_zarr_kwargs: dict):
    """Prepare zarr kwargs for dask.array.to_zarr.

    This helper function ensures that correct kwargs are passed on based on which version of zarr
    and dask is being used. The different versions support different sets of arguments. The zarr_kwargs
    are adjusted in place and thus the original is overwritten. This is not a problem given that the
    arguments being adjusted are the same for the zarr store in use.
    """
    is_zarr_f2 = to_zarr_kwargs.get("zarr_format") == 2

    # The zarr v2 case does not have to be checked here as this is done by
    # the module-level ``zarr_kwargs`` fallback definition.
    # The reason for not doing it here is that it only has one option whereas zarr v3 depends on zarr format being used.
    if IS_ZARR_V3_PLUS and is_zarr_f2:
        if DASK_SUPPORTS_SHARDING:
            # New dask uses chunk_key_encoding
            to_zarr_kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
            to_zarr_kwargs.pop("dimension_separator", None)
        else:
            # Old dask uses dimension_separator
            to_zarr_kwargs["dimension_separator"] = "/"
            to_zarr_kwargs.pop("chunk_key_encoding", None)

    # Old dask (< 2025.12) passes kwargs to zarr.create() which only accepts
    # 'compressor' (singular).  Newer dask uses Group.create_array() which
    # accepts 'compressors' (plural).
    if IS_ZARR_V3_PLUS and not DASK_SUPPORTS_SHARDING:
        _sentinel = object()
        compressors_val = to_zarr_kwargs.pop("compressors", _sentinel)
        if compressors_val is not _sentinel:
            if compressors_val is None:
                to_zarr_kwargs["compressor"] = None
            elif isinstance(compressors_val, (list, tuple)):
                to_zarr_kwargs["compressor"] = (
                    compressors_val[0] if compressors_val else None
                )
            else:
                to_zarr_kwargs["compressor"] = compressors_val
    # Dask versions that use Group.create_array() (< ~2026.3) reject
    # zarr_format (it's inherited from the group).  Newer dask uses
    # zarr.create_array() directly and NEEDS zarr_format to avoid
    # defaulting to format 3.
    if DASK_SUPPORTS_SHARDING and _DASK_USES_GROUP_CREATE:
        to_zarr_kwargs.pop("zarr_format", None)


def _array_write_error(path: str, error: Exception) -> OSError:
    """Build an actionable error for failures while writing a zarr array.

    Writing an array triggers lazy evaluation of the dask graph, which reads
    the source data (e.g. a TIFF/SVS file or its on-disk slab cache). A
    corrupt source can surface here as a cryptic ``OSError`` (such as the
    ``[Errno 22] Invalid argument`` reported in gh-issue-343) only after a long
    conversion. Wrap it with context identifying the likely cause so the
    failure is diagnosable rather than mysterious.
    """
    msg = (
        f"Failed to write data to the zarr array at path '{path}'. "
        f"If converting a TIFF/SVS file, this can indicate a corrupted or "
        f"truncated source file with invalid page offsets or structure. "
        f"Original error: {type(error).__name__}: {error}"
    )
    return OSError(msg)


def _write_array_direct(
    arr: dask.array.Array,
    store: StoreLike,
    path: str,
    sharding_kwargs: dict,
    zarr_kwargs: dict,
    format_kwargs: dict,
    dimension_names_kwargs: dict,
    region: tuple[slice, ...] | None = None,
    zarr_array=None,
    **kwargs,
) -> None:
    """Write an array directly using dask.array.to_zarr."""
    arr = _prep_for_to_zarr(store, arr)

    zarr_fmt = format_kwargs.get("zarr_format")

    # Intercept compressor/compressors from kwargs so they don't leak into
    # zarr.create_array() or dask.array.to_zarr() as unexpected keyword args.
    _sentinel = object()
    user_compressor = kwargs.pop("compressor", _sentinel)
    has_compressor = user_compressor is not _sentinel
    if not has_compressor:
        user_compressor = None
    user_compressors = kwargs.pop("compressors", None)
    user_filters = kwargs.pop("filters", None)

    # Handle sharding kwargs for direct writing
    cleaned_sharding_kwargs = {}

    if sharding_kwargs and "_shard_shape" in sharding_kwargs:
        # For Zarr v3 direct writes, use shards and chunks parameters
        shard_shape = sharding_kwargs["_shard_shape"]
        internal_chunk_shape = sharding_kwargs.get("chunk_shape")

        # Ensure internal_chunk_shape is available
        if internal_chunk_shape is None:
            # Use chunks from arr if available, or default
            internal_chunk_shape = tuple(arr.chunks[i][0] for i in range(arr.ndim))

        # For direct Zarr v3 writes, use shards and chunks
        cleaned_sharding_kwargs["shards"] = shard_shape
        cleaned_sharding_kwargs["chunks"] = internal_chunk_shape

        # Remove internal kwargs
        cleaned_sharding_kwargs.update(
            {
                k: v
                for k, v in sharding_kwargs.items()
                if k not in ["_shard_shape", "chunk_shape"]
            }
        )
    else:
        cleaned_sharding_kwargs = sharding_kwargs

    # Translate user-supplied compression settings into format-appropriate kwargs.
    # zarr v3's ``zarr.create_array`` uses ``compressors`` (plural).  When
    # older dask is used (< 2025.12, calls ``zarr.create()``),
    # ``_prepare_zarr_kwargs`` converts back to singular ``compressor``.
    compression_kwargs = {}
    if IS_ZARR_V3_PLUS:
        if user_compressors is not None:
            compression_kwargs["compressors"] = user_compressors
        elif has_compressor:
            if user_compressor is None:
                compression_kwargs["compressors"] = None
            elif zarr_fmt == 3:
                # zarr format 3 needs zarr v3 codec objects
                v3_codec = _numcodecs_to_zarr_v3_codec(user_compressor)
                compression_kwargs["compressors"] = (
                    [v3_codec] if v3_codec is not None else [user_compressor]
                )
            else:
                # zarr format 2: numcodecs objects are accepted by
                # Group.create_array() (used by new dask) when the group
                # is format 2.
                compression_kwargs["compressors"] = [user_compressor]
        if user_filters is not None:
            compression_kwargs["filters"] = user_filters
    else:
        # zarr v2 library uses 'compressor' (singular)
        if has_compressor:
            compression_kwargs["compressor"] = user_compressor
        if user_filters is not None:
            compression_kwargs["filters"] = user_filters

    to_zarr_kwargs = {
        **cleaned_sharding_kwargs,
        **zarr_kwargs,
        **format_kwargs,
        **dimension_names_kwargs,
        **compression_kwargs,
        **kwargs,
    }

    # For Zarr v3 direct writes without sharding, ``zarr.create_array`` picks
    # its own chunk shape if we do not pass one — that overrides the chunking
    # the caller selected via ``to_multiscales(chunks=...)``.  Forward the
    # dask array's canonical chunk shape (``chunksize`` = per-dim max, which
    # tolerates non-uniform leading chunks) so user intent is respected.
    if zarr_fmt == 3 and zarr_array is None and "chunks" not in to_zarr_kwargs:
        to_zarr_kwargs["chunks"] = arr.chunksize

    if zarr_fmt == 3 and zarr_array is None:
        # Zarr v3, use zarr.create_array and assign (whole array or region)
        array = zarr.create_array(
            store=store,
            name=path,
            shape=arr.shape,
            dtype=arr.dtype,
            **to_zarr_kwargs,
        )
        try:
            if region is not None:
                array[region] = arr.compute()
            else:
                array[:] = arr.compute()
        except (OSError, ValueError) as e:
            raise _array_write_error(path, e) from e
    else:
        _prepare_zarr_kwargs(to_zarr_kwargs)

        target = (
            zarr_array if (region is not None and zarr_array is not None) else store
        )

        try:
            dask.array.to_zarr(
                arr,
                target,
                region=region
                if (region is not None and zarr_array is not None)
                else None,
                component=path,
                overwrite=False,
                compute=True,
                return_stored=False,
                **to_zarr_kwargs,
            )
        except (OSError, ValueError) as e:
            raise _array_write_error(path, e) from e


def _handle_large_array_writing(
    image,
    arr: dask.array.Array,
    store: StoreLike,
    path: str,
    dims: tuple[str, ...],
    dim_factors: dict[str, int],
    chunks: tuple[int, ...],
    sharding_kwargs: dict,
    zarr_kwargs: dict,
    format_kwargs: dict,
    dimension_names_kwargs: dict,
    use_zarrista: bool,
    store_path: str | None,
    zarr_format: int,
    dimension_names: tuple[str, ...],
    internal_chunk_shape: tuple[int, ...] | None,
    shards: tuple[int, ...] | None,
    progress: NgffProgress | NgffProgressCallback | None,
    index: int,
    nscales: int,
    **kwargs,
) -> None:
    """Handle writing large arrays by splitting them into manageable pieces."""
    # Copy zarr_kwargs to avoid mutating the caller's dict across loop iterations
    zarr_kwargs = zarr_kwargs.copy()

    shrink_factors = []
    for dim in dims:
        if dim in dim_factors:
            shrink_factors.append(dim_factors[dim])
        else:
            shrink_factors.append(1)

    # Region writes and shard/array chunk sizes do not need to divide the
    # dimension evenly: Zarr v3 permits a partial final chunk. Region slabs are
    # chunk-aligned and the final slab is truncated to the axis, so a partial
    # final chunk is safe.
    def _find_optimal_chunk_size(first_chunk, dim_size):
        """Target the requested chunk size, clamped to the axis length.

        Never snap to an axis divisor: for dimensions with large prime factors
        (e.g. 320095 = 5 * 64019) the nearest divisor collapses to a tiny chunk
        (5 px), causing extreme chunk-count and storage inflation.
        """
        return max(1, min(int(first_chunk), int(dim_size)))

    def _inner_chunk_for_shard(first_chunk, shard_size):
        """Inner (sub-shard) chunk that divides the shard exactly.

        Zarr's sharding codec requires the shard chunk_shape to be divisible by
        the inner chunk_shape. Return the largest shard divisor <= the target
        that is at or above a small floor, so an awkwardly factored shard (e.g.
        2*prime, with divisors {2, prime, 2*prime}) does not collapse to a tiny
        inner chunk and recreate the very collapse this clamp prevents. When no
        divisor at or above the floor exists (a prime or 2*prime shard), fall
        back to the whole shard as a single inner chunk. A unit-length shard (a
        t or z dim of size 1) returns 1, which is that whole shard rather than a
        degenerate sub-chunk.

        Divisors are enumerated in O(sqrt(shard_size)) rather than scanned
        linearly from the target.
        """
        shard_size = int(shard_size)
        target = max(1, min(int(first_chunk), shard_size))
        min_inner = min(target, 16)
        best = 0
        divisor = 1
        while divisor * divisor <= shard_size:
            if shard_size % divisor == 0:
                for candidate in (divisor, shard_size // divisor):
                    if min_inner <= candidate <= target and candidate > best:
                        best = candidate
            divisor += 1
        return best or shard_size

    # If sharding is enabled, configure it properly
    chunk_kwargs = {}
    codecs_kwargs = {}

    if sharding_kwargs and "_shard_shape" in sharding_kwargs:
        # For Zarr v3 with sharding, clamp the shard shape to the axis length
        shard_shape = sharding_kwargs.pop("_shard_shape")
        internal_chunk_shape = sharding_kwargs.get(
            "chunk_shape"
        )  # This is the inner chunk shape

        # Clamp each shard dim to its axis; Zarr v3 permits a partial final shard,
        # and the inner chunk is chosen to divide the clamped shard below.
        optimized_shard_shape = tuple(
            [
                _find_optimal_chunk_size(s, arr.shape[i])
                for i, s in enumerate(shard_shape)
            ]
        )

        # Ensure internal_chunk_shape divides evenly into optimized_shard_shape
        if internal_chunk_shape is not None:
            # Adjust each internal chunk to be a divisor of the corresponding shard dimension
            adjusted_internal_chunks = []
            for shard_dim, internal_dim in zip(
                optimized_shard_shape, internal_chunk_shape
            ):
                # Find the best divisor of shard_dim that's close to internal_dim
                if shard_dim % internal_dim == 0:
                    # Already divides evenly
                    adjusted_internal_chunks.append(internal_dim)
                else:
                    # Find closest divisor
                    best_divisor = _inner_chunk_for_shard(internal_dim, shard_dim)
                    adjusted_internal_chunks.append(best_divisor)
            internal_chunk_shape = tuple(adjusted_internal_chunks)
        else:
            # No internal chunks specified, use defaults based on array chunks
            internal_chunk_shape = tuple(
                [
                    _inner_chunk_for_shard(c[0], s)
                    for c, s in zip(arr.chunks, optimized_shard_shape)
                ]
            )

        # Configure the sharding codec, respecting user-provided compression
        from zarr.codecs.bytes import BytesCodec
        from zarr.codecs.sharding import ShardingCodec
        from zarr.codecs.zstd import ZstdCodec

        # Determine inner codecs: honour user-supplied compressor/compressors
        user_compressors = kwargs.get("compressors")
        user_compressor = kwargs.get("compressor")
        if user_compressors is not None:
            from collections.abc import Iterable as IterableABC

            from zarr.abc.codec import Codec

            if isinstance(user_compressors, (Codec, dict)):
                compressor_list = [user_compressors]
            elif isinstance(user_compressors, IterableABC):
                compressor_list = list(user_compressors)
            else:
                compressor_list = [user_compressors]
            if any(isinstance(c, BytesCodec) for c in compressor_list):
                inner_codecs = list(compressor_list)
            else:
                inner_codecs = [BytesCodec()] + compressor_list
        elif user_compressor is not None:
            v3_codec = _numcodecs_to_zarr_v3_codec(user_compressor)
            inner_codecs = (
                [BytesCodec(), v3_codec]
                if v3_codec is not None
                else [BytesCodec(), ZstdCodec()]
            )
        else:
            inner_codecs = [BytesCodec(), ZstdCodec()]

        # The array's chunk_shape should be the shard shape
        # The sharding codec's chunk_shape should be the internal chunk shape
        sharding_codec = ShardingCodec(
            chunk_shape=internal_chunk_shape,  # Internal chunk shape within shards
            codecs=inner_codecs,
        )

        # Set up codecs with sharding
        existing_codecs = zarr_kwargs.get("codecs", [])
        if not isinstance(existing_codecs, list):
            existing_codecs = []
        codecs_kwargs["codecs"] = [sharding_codec] + existing_codecs

        # Set the array's chunk_shape to the optimized shard shape
        chunk_kwargs["chunk_shape"] = optimized_shard_shape

        # For region computation, use the optimized shard shape (actual zarr chunk shape)
        zarr_chunk_shape = optimized_shard_shape

        # The zarrista engine creates the array from these values; use the
        # optimized shapes so both engines produce identical chunk grids.
        chunks = optimized_shard_shape
        shards = optimized_shard_shape

        # Clean up remaining kwargs (remove chunk_shape since we're setting it explicitly)
        remaining_kwargs = {
            k: v
            for k, v in sharding_kwargs.items()
            if k not in ["_shard_shape", "chunk_shape"]
        }
        sharding_kwargs_clean = remaining_kwargs
    elif sharding_kwargs:
        # For Zarr v2 or other cases with sharding but no _shard_shape
        chunks = tuple(
            [
                _find_optimal_chunk_size(c[0], arr.shape[i])
                for i, c in enumerate(arr.chunks)
            ]
        )
        zarr_chunk_shape = chunks
        sharding_kwargs_clean = sharding_kwargs
    else:
        # No sharding
        chunks = tuple(
            [
                _find_optimal_chunk_size(c[0], arr.shape[i])
                for i, c in enumerate(arr.chunks)
            ]
        )
        chunk_kwargs = {"chunks": chunks}
        zarr_chunk_shape = chunks
        sharding_kwargs_clean = {}

    if format_kwargs["zarr_format"] == 2:
        if IS_ZARR_V3_PLUS:
            zarr_kwargs["dimension_separator"] = zarr_kwargs["chunk_key_encoding"][
                "separator"
            ]
            del zarr_kwargs["chunk_key_encoding"]
        else:
            zarr_kwargs["dimension_separator"] = "/"

    # For non-sharding paths (codecs_kwargs is empty), propagate user-supplied
    # compression settings that were not captured by the sharding setup above.
    if not codecs_kwargs:
        user_compressor = kwargs.get("compressor")
        user_compressors = kwargs.get("compressors")
        zarr_fmt = format_kwargs["zarr_format"]
        if zarr_fmt == 3:
            from zarr.codecs.bytes import BytesCodec

            if user_compressors is not None:
                from collections.abc import Iterable as IterableABC

                from zarr.abc.codec import Codec

                if isinstance(user_compressors, (Codec, dict)):
                    comp_list = [user_compressors]
                elif isinstance(user_compressors, IterableABC):
                    comp_list = list(user_compressors)
                else:
                    comp_list = [user_compressors]
                if not any(isinstance(c, BytesCodec) for c in comp_list):
                    comp_list = [BytesCodec()] + comp_list
                codecs_kwargs["codecs"] = comp_list
            elif user_compressor is not None:
                v3_codec = _numcodecs_to_zarr_v3_codec(user_compressor)
                if v3_codec is not None:
                    codecs_kwargs["codecs"] = [BytesCodec(), v3_codec]
                else:
                    codecs_kwargs["codecs"] = [BytesCodec(), user_compressor]
        elif zarr_fmt == 2 and user_compressor is not None:
            # open_array() (unlike dask's create_array) accepts 'compressor'
            # singular for format 2 even in zarr v3.
            zarr_kwargs["compressor"] = user_compressor
            user_filters = kwargs.get("filters")
            if user_filters is not None:
                zarr_kwargs["filters"] = user_filters

    # The zarrista engine creates the array itself on the first region write;
    # opening through zarr-python here would write the array metadata with the
    # legacy engine first.
    zarr_array = None
    if not use_zarrista:
        zarr_array = open_array(
            shape=arr.shape,
            dtype=arr.dtype,
            store=store,
            path=path,
            mode="a",
            **chunk_kwargs,
            **sharding_kwargs_clean,
            **zarr_kwargs,
            **codecs_kwargs,
            **dimension_names_kwargs,
            **format_kwargs,
        )

    shape = image.data.shape
    x_index = dims.index("x")
    y_index = dims.index("y")

    regions = _compute_write_regions(
        image, dims, arr, shape, x_index, y_index, zarr_chunk_shape, shrink_factors
    )

    for region_index, region in enumerate(regions):
        if isinstance(progress, NgffProgressCallback):
            progress.add_callback_task(
                f"[green]Writing scale {index + 1} of {nscales}, region {region_index + 1} of {len(regions)}"
            )

        arr_region = arr[region]
        arr_region = _prep_for_to_zarr(store, arr_region)
        optimized = dask.array.Array(
            dask.array.optimize(
                arr_region.__dask_graph__(), arr_region.__dask_keys__()
            ),
            arr_region.name,
            arr_region.chunks,
            meta=arr_region,
        )

        if use_zarrista:
            _write_array_with_zarrista(
                store_path,
                path,
                optimized,
                chunks,  # Use original array chunks, not region chunks
                shards,
                internal_chunk_shape,
                zarr_format,
                dimension_names,
                region,
                full_array_shape=arr.shape,
                create_dataset=(region_index == 0),  # Only create on first region
                **kwargs,
            )
        else:
            _write_array_direct(
                optimized,
                store,
                path,
                sharding_kwargs,
                zarr_kwargs,
                format_kwargs,
                dimension_names_kwargs,
                region,
                zarr_array,
                **kwargs,
            )


def _compute_write_regions(
    image,
    dims: tuple[str, ...],
    arr: dask.array.Array,
    shape: tuple[int, ...],
    x_index: int,
    y_index: int,
    chunks: tuple[int, ...],
    shrink_factors: list[int],
) -> list[tuple[slice, ...]]:
    """Compute the regions for writing a large array in chunks."""
    regions = []

    # If z dimension exists, handle 3D data
    if "z" in dims:
        z_index = dims.index("z")
        slice_bytes = memory_usage(image, {"z"})
        slab_slices = min(
            int(np.ceil(config.memory_target / slice_bytes)), arr.shape[z_index]
        )
        z_chunks = chunks[z_index]
        slice_planes = False
        if slab_slices < z_chunks:
            slab_slices = z_chunks
            slice_planes = True
        if slab_slices > arr.shape[z_index]:
            slab_slices = arr.shape[z_index]
        slab_slices = int(slab_slices / z_chunks) * z_chunks
        num_z_splits = int(np.ceil(shape[z_index] / slab_slices))
        while num_z_splits % shrink_factors[z_index] > 1:
            num_z_splits += 1

        # num_y_splits = 1
        # num_x_splits = 1

        for slab_index in range(num_z_splits):
            # Process individual slabs
            if slice_planes:
                regions.extend(
                    _compute_plane_regions(
                        image,
                        dims,
                        arr,
                        shape,
                        x_index,
                        y_index,
                        z_index,
                        chunks,
                        shrink_factors,
                        slab_index,
                    )
                )
            else:
                region = [slice(arr.shape[i]) for i in range(arr.ndim)]
                region[z_index] = slice(
                    slab_index * slab_slices,
                    min((slab_index + 1) * slab_slices, arr.shape[z_index]),
                )
                regions.append(tuple(region))
    else:
        # 2D data - one region covering the whole array
        regions.append(tuple([slice(arr.shape[i]) for i in range(arr.ndim)]))

    return regions


def _compute_plane_regions(
    image,
    dims: tuple[str, ...],
    arr: dask.array.Array,
    shape: tuple[int, ...],
    x_index: int,
    y_index: int,
    z_index: int,
    chunks: tuple[int, ...],
    shrink_factors: list[int],
    slab_index: int,
) -> list[tuple[slice, ...]]:
    """Compute regions for a single z-slab, dividing into planes and strips if needed."""
    plane_regions = []
    z_chunks = chunks[z_index]
    y_chunks = chunks[y_index]
    x_chunks = chunks[x_index]

    # Calculate how to divide planes
    plane_bytes = memory_usage(image, {"z", "y"})
    plane_slices = min(
        int(np.ceil(config.memory_target / plane_bytes)),
        arr.shape[y_index],
    )
    slice_strips = False
    if plane_slices < y_chunks:
        plane_slices = y_chunks
        slice_strips = True
    if plane_slices > arr.shape[y_index]:
        plane_slices = arr.shape[y_index]
    plane_slices = int(plane_slices / y_chunks) * y_chunks
    num_y_splits = int(np.ceil(shape[y_index] / plane_slices))
    while num_y_splits % shrink_factors[y_index] > 1:
        num_y_splits += 1

    if slice_strips:
        # Need to subdivide further into strips
        strip_bytes = memory_usage(image, {"z", "y", "x"})
        strip_slices = min(
            int(np.ceil(config.memory_target / strip_bytes)),
            arr.shape[x_index],
        )
        strip_slices = max(strip_slices, x_chunks)
        if strip_slices > arr.shape[x_index]:
            strip_slices = arr.shape[x_index]
        strip_slices = int(strip_slices / x_chunks) * x_chunks
        num_x_splits = int(np.ceil(shape[x_index] / strip_slices))
        while num_x_splits % shrink_factors[x_index] > 1:
            num_x_splits += 1

        for plane_index in range(num_y_splits):
            for strip_index in range(num_x_splits):
                region = [slice(arr.shape[i]) for i in range(arr.ndim)]
                region[z_index] = slice(
                    slab_index * z_chunks,
                    min((slab_index + 1) * z_chunks, arr.shape[z_index]),
                )
                region[y_index] = slice(
                    plane_index * plane_slices,
                    min((plane_index + 1) * plane_slices, arr.shape[y_index]),
                )
                region[x_index] = slice(
                    strip_index * strip_slices,
                    min((strip_index + 1) * strip_slices, arr.shape[x_index]),
                )
                plane_regions.append(tuple(region))
    else:
        # Just divide into planes
        for plane_index in range(num_y_splits):
            region = [slice(arr.shape[i]) for i in range(arr.ndim)]
            region[z_index] = slice(
                slab_index * z_chunks,
                min((slab_index + 1) * z_chunks, arr.shape[z_index]),
            )
            region[y_index] = slice(
                plane_index * plane_slices,
                min((plane_index + 1) * plane_slices, arr.shape[y_index]),
            )
            plane_regions.append(tuple(region))

    return plane_regions


def _prepare_next_scale(
    image,
    index: int,
    nscales: int,
    multiscales: NgffMultiscales,
    store: StoreLike,
    path: str,
    progress: NgffProgress | NgffProgressCallback | None,
    scale_strategy: ScaleStrategy = "pad",
) -> object | None:
    """Prepare the next scale for processing if needed.

    :param scale_strategy: Strategy for handling non-power-of-2 scale factors.
        "pad" (default) always downsamples incrementally from the previous
        level, which is memory-efficient but can miss the target sizes; the
        datasets metadata still describes the exact targets, so the store then
        misdescribes its own geometry. "exact" uses pre-computed images from
        the initial to_multiscales() call when incremental downsampling cannot
        achieve the exact target size, so the written arrays match the
        coordinate metadata.
    """
    # No next scale if we're at the last one
    if index >= nscales - 1:
        return None
    # Minimize task graph depth
    if multiscales.scale_factors and multiscales.method and multiscales.chunks:
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []

        image.data = dask.array.from_zarr(store, component=path)

        # Get the absolute scale factor for the next level
        next_absolute_factor = multiscales.scale_factors[index]
        original_image = multiscales.images[0]
        spatial_dims = {"x", "y", "z"}

        # Track what scale factor was applied to reach current image
        previous_dim_factors = dict.fromkeys(image.dims, 1)
        if index > 0:
            prev_absolute_factor = multiscales.scale_factors[index - 1]
            if isinstance(prev_absolute_factor, int):
                for d in image.dims:
                    if d in spatial_dims:
                        previous_dim_factors[d] = prev_absolute_factor
            else:
                for d in prev_absolute_factor:
                    previous_dim_factors[d] = prev_absolute_factor[d]

        # Compute incremental factor from current image to next scale
        dim_factors = _dim_scale_factors(
            image.dims,
            next_absolute_factor,
            previous_dim_factors,
            original_image=original_image,
            previous_image=image,
        )

        # Check whether incremental downsampling would fail to achieve the
        # exact target size (original_size // absolute_factor).  When True,
        # "exact" mode should fall back to the pre-computed image.
        can_use_exact_mode = False
        for dim in dim_factors:
            if dim in spatial_dims:
                dim_index = original_image.dims.index(dim)
                original_size = original_image.data.shape[dim_index]
                # Handle both int and dict scale_factor
                dim_scale_factor = (
                    next_absolute_factor[dim]
                    if isinstance(next_absolute_factor, dict)
                    else next_absolute_factor
                )
                target_size = int(original_size / dim_scale_factor)

                prev_dim_index = image.dims.index(dim)
                previous_size = image.data.shape[prev_dim_index]

                # Check if floor(previous_size / dim_factors[dim]) != target_size
                if int(previous_size / dim_factors[dim]) != target_size:
                    can_use_exact_mode = True
                    break

        if scale_strategy == "exact" and can_use_exact_mode:
            # Cannot achieve exact target sizes via incremental downsampling.
            # Use the already-computed image from multiscales.images[index + 1]
            # which was computed correctly during the initial to_multiscales call.
            return multiscales.images[index + 1]
        # Downsample from current (previous) image.
        # In "pad" mode this always runs (sizes may differ slightly due to
        # floor-division rounding).  In "exact" mode this runs when
        # incremental downsampling achieves the exact target.
        source_image = image
        # Only include spatial dimensions in the scale factors dict
        # to avoid KeyError in methods that look up scale/translation
        next_multiscales_factor = {
            d: f for d, f in dim_factors.items() if d in spatial_dims
        }

        next_multiscales = to_multiscales(
            source_image,
            scale_factors=[
                next_multiscales_factor,
            ],
            method=multiscales.method,
            chunks=multiscales.chunks,
            progress=progress,
            cache=False,
        )
        multiscales.images[index + 1] = next_multiscales.images[1]
        return next_multiscales.images[1]
    return multiscales.images[index + 1]


def to_ome_zarr(
    store: StoreLike,
    multiscales: NgffMultiscales,
    version: str = "0.5",
    overwrite: bool = True,
    use_tensorstore: bool = False,
    chunk_store: StoreLike | None = None,
    progress: NgffProgress | NgffProgressCallback | None = None,
    chunks_per_shard: int | tuple[int, ...] | dict[str, int] | None = None,
    scale_strategy: ScaleStrategy = "pad",
    **kwargs,
) -> None:
    """
    Write an image pixel array and metadata to a Zarr store with the OME-NGFF standard data model.

    :param store: Store or path to directory in file system. If the path ends with .ozx, writes an RFC-9
    compliant zipped OME-Zarr file.
    :type  store: StoreLike

    :param multiscales: NgffMultiscales OME-NGFF image pixel data and metadata. Can be generated with ngff_zarr.to_multiscales.
    :type  multiscales: NgffMultiscales

    :param version: OME-Zarr specification version. For .ozx files, version 0.5 is required.
    :type  version: str, optional

    :param overwrite: If True, delete any pre-existing data in `store` before creating groups.
    :type  overwrite: bool, optional

    :param use_tensorstore: Deprecated. If True, write arrays with the zarrista
        backend's fast direct-write path (tensorstore is no longer used).
    :type  use_tensorstore: bool, optional

    :param chunk_store: Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    :type  chunk_store: StoreLike, optional

    :param progress: Optional progress logger
    :type  progress: RichDaskProgress

    :param chunks_per_shard: Number of chunks along each axis in a shard. If None, no sharding. For .ozx files, defaults to 2 if not specified. Requires OME-Zarr version >= 0.5.
    :type  chunks_per_shard: int, tuple, or dict, optional


    :param scale_strategy: Strategy for handling non-power-of-2 scale factors during
        multiscale writing. "pad" (default) always downsamples incrementally from
        the previous level, which is memory-efficient but can miss the target
        sizes; the datasets metadata still describes the exact targets, so the
        store then misdescribes its own geometry. "exact" uses pre-computed
        images from the initial to_multiscales() call when incremental
        downsampling cannot achieve the exact target size
        (original_size // scale_factor), so the written arrays match the
        coordinate metadata.
    :type  scale_strategy: "pad" or "exact", optional

    :param **kwargs: Passed to the zarr.create_array() or zarr.creation.create() function, e.g., compression options.
    """
    if use_tensorstore:
        warnings.warn(
            "use_tensorstore is deprecated; the fast direct-write path now "
            "uses the zarrista backend instead of tensorstore.",
            DeprecationWarning,
            stacklevel=2,
        )

    # ``enabled_rfcs`` was removed: RFC-4 anatomical orientation is now written
    # automatically whenever it is present. Reject it explicitly so a legacy
    # caller gets a clear migration message instead of an opaque error from the
    # downstream zarr-creation kwargs.
    if "enabled_rfcs" in kwargs:
        raise TypeError(
            "to_ome_zarr()/to_ngff_zarr() no longer accepts 'enabled_rfcs'. "
            "RFC-4 anatomical orientation is now written automatically whenever "
            "it is present. Set NgffImage.axes_orientations, or pass "
            "orientation= to to_multiscales()."
        )

    # RFC-9: Handle .ozx (zipped OME-Zarr) files
    if isinstance(store, (str, Path)) and is_ozx_path(store):
        if version != "0.5":
            raise ValueError(
                "RFC-9 zipped OME-Zarr (.ozx) requires OME-Zarr version 0.5. "
                f"Got version '{version}'. Please set version='0.5'."
            )

        # Default chunks_per_shard to 2 for .ozx files if not specified
        if chunks_per_shard is None:
            chunks_per_shard = 2

        # Determine if we should use memory or disk for intermediate storage
        total_memory_usage = sum(memory_usage(img) for img in multiscales.images)
        use_memory_store = total_memory_usage <= config.memory_target

        if use_memory_store:
            # Small dataset: use memory store
            from zarr.storage import MemoryStore

            temp_store = MemoryStore()
        else:
            # Large dataset: use temporary directory store in cache
            if hasattr(zarr.storage, "DirectoryStore"):
                LocalStore = zarr.storage.DirectoryStore
            else:
                LocalStore = zarr.storage.LocalStore

            temp_dir = tempfile.mkdtemp(
                dir=config.cache_store.path
                if hasattr(config.cache_store, "path")
                else None
            )
            temp_store = LocalStore(temp_dir)

        try:
            # Write to temporary store first
            _to_ngff_zarr_impl(
                temp_store,
                multiscales,
                version=version,
                overwrite=overwrite,
                use_tensorstore=False,  # zarrista writer needs a path-like store, not memory/temp stores
                chunk_store=None,
                progress=progress,
                chunks_per_shard=chunks_per_shard,
                scale_strategy=scale_strategy,
                **kwargs,
            )

            # Write temp store to .ozx file
            write_store_to_zip(temp_store, store, version=version)
        finally:
            # Clean up temporary directory if used
            if not use_memory_store:
                import shutil

                if hasattr(zarr.storage, "DirectoryStore") and isinstance(
                    temp_store, zarr.storage.DirectoryStore
                ):
                    shutil.rmtree(temp_store.dir_path(), ignore_errors=True)
                elif hasattr(zarr.storage, "LocalStore") and isinstance(
                    temp_store, zarr.storage.LocalStore
                ):
                    shutil.rmtree(temp_store.root, ignore_errors=True)

        return

    # Standard (non-.ozx) path
    _to_ngff_zarr_impl(
        store,
        multiscales,
        version=version,
        overwrite=overwrite,
        use_tensorstore=use_tensorstore,
        chunk_store=chunk_store,
        progress=progress,
        chunks_per_shard=chunks_per_shard,
        scale_strategy=scale_strategy,
        **kwargs,
    )


#: Backwards-compatible alias for :func:`to_ome_zarr`.
to_ngff_zarr = to_ome_zarr


def _to_ngff_zarr_impl(
    store: StoreLike,
    multiscales: NgffMultiscales,
    version: str = "0.4",
    overwrite: bool = True,
    use_tensorstore: bool = False,
    chunk_store: StoreLike | None = None,
    progress: NgffProgress | NgffProgressCallback | None = None,
    chunks_per_shard: int | tuple[int, ...] | dict[str, int] | None = None,
    scale_strategy: ScaleStrategy = "pad",
    **kwargs,
) -> None:
    """
    Internal implementation of to_ngff_zarr without .ozx handling.
    """
    # Shallow copy, with per-image computed_callbacks: the write loop swaps
    # image data for on-disk views and regenerates scales, and none of that
    # may reach the caller's multiscales (gh-issue-627).
    multiscales = copy.copy(multiscales)
    multiscales.images = [copy.copy(image) for image in multiscales.images]
    for image in multiscales.images:
        image.computed_callbacks = list(image.computed_callbacks)

    # Setup and validation
    store_path = os.fspath(store) if isinstance(store, (str, os.PathLike)) else None

    _validate_ngff_parameters(version, chunks_per_shard, use_tensorstore, store)
    metadata, dimension_names, dimension_names_kwargs = _prepare_metadata(
        multiscales, version
    )
    metadata_dict = asdict(metadata)
    metadata_dict = _pop_metadata_optionals(metadata_dict)
    metadata_dict["@type"] = "ngff:Image"

    # Route all writes — group/attribute metadata and arrays — through the
    # zarrista compat layer for path-like stores; store objects, mappings, and
    # split chunk_store setups keep the zarr-python engine.
    use_zarrista = chunk_store is None and use_zarrista_for(store)

    # Create Zarr root
    root = _create_zarr_root(
        store,
        chunk_store,
        version,
        overwrite,
        metadata_dict,
        use_zarrista=use_zarrista,
    )

    # Format parameters
    zarr_format = 2 if version == "0.4" else 3
    format_kwargs = {"zarr_format": zarr_format}
    _zarr_kwargs = zarr_kwargs.copy()

    if version == "0.4" and kwargs.get("compressors") is not None:
        raise ValueError(
            "The argument `compressors` is not supported for OME-Zarr version 0.4 "
            "(Zarr v2). Use `compressor` instead."
        )

    # Process each scale level
    nscales = len(multiscales.images)
    if progress:
        progress.add_multiscales_task("[green]Writing scales", nscales)

    next_image = multiscales.images[0]
    dims = next_image.dims
    previous_dim_factors = dict.fromkeys(dims, 1)

    for index in range(nscales):
        if progress:
            progress.update_multiscales_task_completed(index + 1)

        image = next_image
        arr = image.data
        path = metadata.datasets[index].path
        parent = str(PurePosixPath(path).parent)

        # Create parent groups if needed
        if parent not in (".", "/"):
            if use_zarrista:
                create_zarrista_subgroup(
                    store,
                    parent,
                    {"_ARRAY_DIMENSIONS": list(image.dims)},
                    zarr_format,
                )
            else:
                array_dims_group = root.create_group(parent)
                array_dims_group.attrs["_ARRAY_DIMENSIONS"] = image.dims

        # Calculate dimension factors
        if index > 0 and index < nscales - 1 and multiscales.scale_factors:
            dim_factors = _dim_scale_factors(
                dims, multiscales.scale_factors[index], previous_dim_factors
            )
        else:
            dim_factors = dict.fromkeys(dims, 1)
        previous_dim_factors = dim_factors

        # Configure sharding if needed
        # TODO check with recent updates to zarr by Ilan whether sharding can just be configured on zarr side.
        sharding_kwargs, internal_chunk_shape, arr = _configure_sharding(
            arr, chunks_per_shard, dims, kwargs.copy()
        )

        # Get the chunks - these are now the shards if sharding is enabled.
        # Use the canonical per-dim max (``chunksize``): a sliced input can
        # carry a smaller partial leading chunk which must not become the
        # stored chunk shape (gh-issue-488).
        chunks = arr.chunksize

        # For the zarrista writer, shards are the same as chunks when sharding
        # is enabled. Use the true shard grid from _configure_sharding: dask's
        # rechunk clamps arr.chunks to the array shape, but the stored shard
        # shape may exceed it (a partial final shard), and the inner chunk
        # shape must divide the stored shard shape evenly.
        if chunks_per_shard is not None:
            shards = tuple(
                sharding_kwargs.get("_shard_shape")
                or sharding_kwargs.get("shards")
                or chunks
            )
            chunks = shards
        else:
            shards = None

        # Determine write method based on memory requirements
        if memory_usage(image) > config.memory_target:
            _handle_large_array_writing(
                image,
                arr,
                store,
                path,
                dims,
                dim_factors,
                chunks,
                sharding_kwargs,
                _zarr_kwargs,
                format_kwargs,
                dimension_names_kwargs,
                use_tensorstore or use_zarrista,
                store_path,
                zarr_format,
                dimension_names,
                internal_chunk_shape,
                shards,
                progress,
                index,
                nscales,
                **kwargs,
            )
        else:
            if isinstance(progress, NgffProgressCallback):
                progress.add_callback_task(
                    f"[green]Writing scale {index + 1} of {nscales}"
                )

            # For small arrays, write in one go
            region = tuple([slice(arr.shape[i]) for i in range(arr.ndim)])
            if use_tensorstore or use_zarrista:
                _write_array_with_zarrista(
                    store_path,
                    path,
                    arr,
                    chunks,
                    shards,
                    internal_chunk_shape,
                    zarr_format,
                    dimension_names,
                    region,
                    full_array_shape=arr.shape,
                    create_dataset=True,  # Always create for small arrays
                    **kwargs,
                )
            else:
                _write_array_direct(
                    arr,
                    store,
                    path,
                    sharding_kwargs,
                    _zarr_kwargs,
                    format_kwargs,
                    dimension_names_kwargs,
                    None,
                    None,
                    **kwargs,
                )

        # Prepare next scale if needed
        next_image = _prepare_next_scale(
            image,
            index,
            nscales,
            multiscales,
            store,
            path,
            progress,
            scale_strategy=scale_strategy,
        )

    # Clean up callbacks
    for image in multiscales.images:
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []

    # Consolidate metadata
    if use_zarrista:
        _zarrista_consolidate_metadata(store, zarr_format)
    elif IS_ZARR_V3_PLUS:
        with warnings.catch_warnings():
            # Ignore consolidated metadata warning
            warnings.filterwarnings("ignore", category=UserWarning)
            zarr.consolidate_metadata(store, **format_kwargs)
    else:
        # Zarr_format is used elsewhere but for this consolidate_metadata it is not an argument in zarr v2.
        if format_kwargs.get("zarr_format"):
            del format_kwargs["zarr_format"]
        zarr.consolidate_metadata(store, **format_kwargs)
