# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import copy
import json
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import dask.array
import numpy as np
from itkwasm import array_like_to_numpy_array

from ._store_types import StoreLike
from ._supported_versions import V06_ONDISK_VERSION, NgffVersion
from ._zarrista_utils import (
    OME_ROOT_KEYS,
    _ZarristaArrayAdapter,
    create_zarrista_group,
    create_zarrista_subgroup,
    has_consolidated_metadata,
    normalize_store,
    open_lazy_array,
    read_group_attributes,
)
from ._zarrista_utils import (
    consolidate_metadata as _zarrista_consolidate_metadata,
)
from .config import config
from .memory_usage import memory_usage
from .methods._metadata import get_method_metadata
from .methods._support import _dim_scale_factors
from .multiscales import NgffMultiscales
from .rfc9_zip import is_ozx_path, write_store_to_zip
from .rich_dask_progress import NgffProgress, NgffProgressCallback
from .to_multiscales import to_multiscales
from .v04.zarr_metadata import Metadata as Metadata_v04
from .v05.zarr_metadata import Metadata as Metadata_v05

ScaleStrategy = Literal["pad", "exact"]


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


def _prep_for_write(arr: dask.array.Array) -> dask.array.Array:
    """Convert array-like blocks (itkwasm, cupy, ...) to numpy for writing."""
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


def _create_dataset(
    store_path: str,
    shape,
    dtype,
    chunks,
    zarr_format,
    dimension_names=None,
    internal_chunk_shape=None,
    compression_chain=None,
):
    """Create the array node at ``store_path`` (``<store>/<path>``) and return it.

    ``compression_chain`` carries the ordered compression codec chain (for
    zarr format 2 metadata only its first entry is stored): ``None`` requests
    the engine default, an empty sequence requests no compression. An array
    already at the path is opened instead of created.
    """
    from ._zarrista_utils import create_zarrista_array, open_zarrista_array

    scale_path = Path(store_path)
    store_root = scale_path.parent
    node_name = scale_path.name

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
        return create_zarrista_array(
            store_root,
            node_name,
            tuple(shape),
            dtype,
            chunks,
            zarr_format,
            **create_kwargs,
        )
    except Exception as create_error:
        # Dataset may already exist: fall back to opening it, surfacing
        # the create failure when there is nothing to open.
        try:
            return open_zarrista_array(store_root, node_name)
        except Exception:
            raise create_error from None


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

    *store_path* addresses the array node itself (``<store>/<path>``); see
    :func:`_create_dataset` for ``compression_chain``.
    """
    from ._zarrista_utils import (
        _native_contiguous,
        open_zarrista_array,
        write_dask_array,
    )

    # Use full array shape if provided, otherwise use the region array shape
    dataset_shape = tuple(
        full_array_shape if full_array_shape is not None else array.shape
    )

    if create_dataset:
        dataset = _create_dataset(
            store_path,
            dataset_shape,
            array.dtype,
            chunks,
            zarr_format,
            dimension_names,
            internal_chunk_shape,
            compression_chain,
        )
    else:
        scale_path = Path(store_path)
        dataset = open_zarrista_array(scale_path.parent, scale_path.name)

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
) -> None:
    """Validate the parameters for the NGFF Zarr generation."""
    if isinstance(version, str):
        version = NgffVersion(version)

    if version not in [
        NgffVersion.V04,
        NgffVersion.V05,
        NgffVersion.V06,
        NgffVersion.V09dev1,
    ]:
        raise ValueError(f"Unsupported version: {version}")

    if chunks_per_shard is not None and version == NgffVersion.V04:
        raise ValueError(
            "Sharding is only supported for OME-Zarr version 0.5 and later"
        )


def _gate_top_level_transforms(metadata, version: str) -> None:
    """Refuse a multiscale-level transform the 0.6 schema cannot express.

    From 0.6 a transform on the multiscales entry maps between two named
    coordinate systems, and the schema requires both ``input`` and ``output``
    to name one. The writer serializes whatever the model holds, so a missing
    reference would produce a store the validated reader rejects.

    0.9.dev1 is the 0.6 model with the axis restrictions relaxed, so its
    inter-system transforms carry the same requirement and the gate covers it.

    Each transform then runs the reader's ``validate_transform`` against the
    systems it names, so a store this writes is one it can read back.
    """
    if version not in ("0.6", NgffVersion.V09dev1.value):
        return
    if not metadata.coordinateTransformations:
        return
    from .v06.zarr_metadata import validate_transform

    systems = getattr(metadata, "coordinateSystems", None)
    for index, transform in enumerate(metadata.coordinateTransformations):
        for side in ("input", "output"):
            reference = getattr(transform, side, None)
            if reference is None or getattr(reference, "name", None) is None:
                raise ValueError(
                    f"multiscales coordinateTransformations[{index}] "
                    f"({transform.type}) names no {side} coordinate system; "
                    "OME-Zarr 0.6 requires every multiscale-level transformation "
                    "to name both its input and its output coordinate system"
                )
        try:
            validate_transform(transform, systems)
        except ValueError as invalid:
            raise ValueError(
                f"multiscales coordinateTransformations[{index}] "
                f"({transform.type}) would be written as a transform this "
                f"package cannot read back: {invalid}"
            ) from invalid


@dataclass(frozen=True)
class _AxisView:
    """Minimal stand-in exposing only ``axes``.

    The axis rules in :mod:`ngff_zarr.structural_validation` read nothing else
    off the metadata. A v0.6 ``Metadata`` has no ``axes`` attribute, only
    ``coordinateSystems``.
    """

    axes: list


def _axis_views(metadata) -> list[tuple[str, _AxisView]]:
    """Every axis list ``metadata`` will serialize, paired with its location.

    v0.4/v0.5 metadata carry a flat ``axes``; v0.6 and 0.9.dev1 carry
    ``coordinateSystems``, and ``coordinate_systems.schema`` applies
    ``axes.schema`` to each one, so every system is returned.

    ``coordinateSystems`` is checked first: v0.9.dev1 also exposes an ``axes``
    property, which returns only the intrinsic system's axes.
    """
    systems = list(getattr(metadata, "coordinateSystems", None) or [])
    if systems:
        return [
            (f"multiscales[0].coordinateSystems[{i}].axes", _AxisView(list(cs.axes)))
            for i, cs in enumerate(systems)
        ]
    axes = getattr(metadata, "axes", None)
    if axes is not None:
        return [("multiscales[0].axes", _AxisView(list(axes)))]
    return []


def _gate_axis_model(metadata, version) -> None:
    """Refuse to serialize an axis model the target ``version`` cannot express.

    Reuses the axis rules of :mod:`ngff_zarr.structural_validation`, but this
    is a second dispatcher, not the same pass as
    :func:`ngff_zarr.validate_structural`, and it differs from it two ways:

    1. It runs only the five axis rules, not the full image cascade.
    2. It checks **every** coordinate system (see :func:`_axis_views`), while
       ``validate_structural`` reads a single flat ``axes`` list.

    Called after ``to_version``: the converted object is what ``asdict``
    serializes, and for a 0.4/0.5 target it exposes the axes that survive the
    downgrade. The corollary is that this cannot refuse what the downgrade has
    already dropped -- ``Metadata._to_v05`` keeps only the coordinate system
    the datasets reference -- so a model refused at 0.6 may be accepted at 0.4
    with the other systems silently discarded.
    """
    from .structural_validation import (
        SpecRule,
        ValidationError,
        validate_axis_count,
        validate_axis_names_unique,
        validate_axis_order,
        validate_axis_type,
        validate_spatial_axis_order,
    )

    rules = (
        validate_axis_count,
        validate_axis_type,
        validate_axis_order,
        validate_spatial_axis_order,
        validate_axis_names_unique,
    )
    for location, view in _axis_views(metadata):
        for rule in rules:
            try:
                rule(view, version)
            except ValidationError as exc:
                rendered = ", ".join(
                    f"{ax.name!r}(type={ax.type!r})" for ax in view.axes
                )
                if exc.rule is SpecRule.AXIS_NAMES_UNIQUE:
                    # Required at every version, 0.9.dev1 included.
                    raise ValueError(
                        f'Cannot write OME-Zarr version="{version}": {exc.message} '
                        f"Axes at {location}: [{rendered}]."
                    ) from exc
                raise ValueError(
                    f'Cannot write OME-Zarr version="{version}": this axis model '
                    f"violates that version's [{exc.rule.value}] rule. "
                    f"{exc.message} Axes at {location}: [{rendered}]. "
                    f'Pass version="{NgffVersion.V09dev1.value}" to write it: '
                    "0.9.dev1 is the only OME-Zarr version that adopts RFC-3 "
                    "(arbitrary axis count, names, types and ordering)."
                ) from exc


def _lazy_array_leaves(arr: dask.array.Array, seen: set[str] | None = None):
    """Yield the local store arrays embedded in the dask graph of ``arr``.

    An array opened from a local store enters a dask graph as a
    ``_ZarristaArrayAdapter`` held by a materialized layer; blockwise layers
    only reference it by key. Materialized data (``persist()``) holds none.

    ``seen`` carries the layer names already walked. The levels of a
    multiscales are derived from one another, so each level's graph holds
    every layer the levels below it hold, and walking them per level costs
    the same layer once per level that keeps it.
    """
    graph = arr.__dask_graph__()
    layers = getattr(graph, "layers", None)
    if layers is None:
        sources = graph.values()
    else:
        if seen is None:
            seen = set()
        fresh = [name for name in layers if name not in seen]
        seen.update(fresh)
        sources = (
            value
            for name in fresh
            for value in getattr(layers[name], "mapping", {}).values()
        )
    for value in sources:
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, _ZarristaArrayAdapter):
                if item.store_path is not None:
                    yield item
            elif isinstance(item, (tuple, list)):
                stack.extend(item)


def _leaf_array_dir(leaf: _ZarristaArrayAdapter) -> Path:
    """Directory of the array a lazy array leaf reads from."""
    parts = [p for p in (leaf.node_path or "").split("/") if p not in ("", ".")]
    return Path(leaf.store_path).resolve().joinpath(*parts)


def _reads_below(leaf: _ZarristaArrayAdapter, directory: Path) -> bool:
    """Whether a lazy array leaf reads an array at or below ``directory``."""
    array_dir = _leaf_array_dir(leaf)
    return array_dir == directory or directory in array_dir.parents


def _guard_overwrite_of_source_store(
    multiscales: NgffMultiscales, store_path: str
) -> None:
    """Refuse an overwrite that would clear the store the images still read.

    Overwriting removes every node of the store before dask computes a
    chunk. Reads of the removed chunks then return the fill value, so the
    write silently fills the store with zeros.
    """
    destination = Path(store_path).resolve()
    seen: set[str] = set()
    for index, image in enumerate(multiscales.images):
        if not isinstance(image.data, dask.array.Array):
            continue
        for leaf in _lazy_array_leaves(image.data, seen):
            if _reads_below(leaf, destination):
                raise ValueError(
                    f"Cannot overwrite '{store_path}': multiscales.images[{index}] "
                    "reads its data from that store, and overwrite=True removes "
                    "the store contents before the data is computed, which "
                    "would write zeros. Write to a different store, append "
                    "levels with start_level, or materialize the data first, "
                    "e.g. image.data = image.data.persist()."
                )


def _guard_rewrite_of_read_levels(
    multiscales: NgffMultiscales, store_path: str, datasets, start_level: int
) -> None:
    """Refuse to replace a stored level that a level to be written still reads.

    A level at or above ``start_level`` replaces the array at its path, and
    the array is removed before its replacement is computed.
    """
    store_root = Path(store_path).resolve()
    replaced = {
        store_root.joinpath(*_node_parts(dataset.path)): dataset.path
        for dataset in datasets[start_level:]
    }
    seen: set[str] = set()
    for index in range(start_level, len(multiscales.images)):
        data = multiscales.images[index].data
        if not isinstance(data, dask.array.Array):
            continue
        for leaf in _lazy_array_leaves(data, seen):
            array_dir = _leaf_array_dir(leaf)
            if array_dir in replaced:
                raise ValueError(
                    f"multiscales.images[{index}] reads the array at "
                    f"'{replaced[array_dir]}', which start_level={start_level} "
                    "replaces before its data is computed. Materialize that "
                    "level first, e.g. image.data = image.data.persist(), or "
                    f"pass a multiscales whose levels from {start_level} on are "
                    f"derived from the stored level {start_level - 1}."
                )


def _node_parts(path: str) -> list[str]:
    """Path components of a node path within a store."""
    return [part for part in str(path).split("/") if part not in ("", ".")]


def _bind_existing_level(image, store_path: str, path: str, index: int) -> None:
    """Point ``image`` at the array already stored for scale ``index``.

    The array must match the shape and dtype the multiscales describes for
    that level: the levels written after it are derived from it, and the
    datasets metadata is rewritten from the multiscales.
    """
    from ._zarrista_utils import open_zarrista_array

    try:
        existing = open_zarrista_array(store_path, path)
    except Exception as error:
        raise ValueError(
            f"start_level expects scale level {index} at '{path}', but the "
            f"store has no readable array there: {type(error).__name__}: {error}"
        ) from error
    shape = tuple(existing.shape)
    dtype = np.dtype(existing.dtype.name)
    if shape != tuple(image.data.shape) or dtype != np.dtype(image.data.dtype):
        raise ValueError(
            f"The array stored for scale level {index} at '{path}' has shape "
            f"{shape} and dtype {dtype}, but the multiscales describes "
            f"{tuple(image.data.shape)} {image.data.dtype}."
        )
    for callback in image.computed_callbacks:
        callback()
    image.computed_callbacks = []
    image.data = open_lazy_array(store_path, path)


def _remove_existing_level(store_path: str, path: str) -> None:
    """Remove the array stored at ``path``, so its replacement starts clean."""
    node_dir = Path(store_path).joinpath(*_node_parts(path))
    if node_dir.exists():
        shutil.rmtree(node_dir)


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
    _gate_axis_model(metadata, version)
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
    including mapping the API version ``"0.6"`` to the pre-release string
    stored on disk (:data:`~ngff_zarr._supported_versions.V06_ONDISK_VERSION`). ``metadata_dict`` is mutated in place: its ``omero`` entry
    is popped so it lives only in its hoisted location, matching historical
    behavior.
    """
    if version != "0.4":
        # RFC 2, Zarr 3 - omero goes inside ome namespace
        if version == "0.6":
            version = V06_ONDISK_VERSION.value
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


def _check_root_attributes(attributes: Mapping[str, Any] | None) -> dict:
    """The root attributes a caller writes beside the OME metadata."""
    if not attributes:
        return {}
    owned = sorted(OME_ROOT_KEYS.intersection(attributes))
    if owned:
        raise ValueError(
            f"root_attributes cannot set {owned}: the writer derives the OME "
            "metadata from the multiscales."
        )
    return dict(attributes)


def update_root_attributes(store: StoreLike, attributes: Mapping[str, Any]) -> None:
    """Merge ``attributes`` into the root attributes of an existing OME-Zarr store.

    Keys beside the OME metadata hold application metadata, such as a fact
    only known once the last region of a store created with
    ``metadata_only=True`` has been written. Existing keys are replaced and
    the others kept; the OME keys cannot be set this way. Consolidated
    metadata is refreshed when the store carries it.
    """
    attributes = _check_root_attributes(attributes)
    if not attributes:
        return
    store_path = normalize_store(store)
    for zarr_format in (3, 2):
        if read_group_attributes(store_path, zarr_format=zarr_format) is not None:
            break
    else:
        raise ValueError(f"No OME-Zarr root group at '{store_path}'.")
    consolidated = has_consolidated_metadata(store_path, zarr_format)
    create_zarrista_group(store_path, attributes, zarr_format)
    if consolidated:
        _zarrista_consolidate_metadata(store_path, zarr_format)


def _foreign_root_attrs(store: StoreLike) -> dict:
    """Root attributes of an existing store that the writer does not own.

    Application metadata such as direction cosines or provenance lives beside
    the OME keys and survives an overwrite; the OME keys of every version are
    excluded so a version change never leaves a stale entry behind. An absent
    store or root group has nothing to preserve, and a root that holds an
    array carries no group attributes. Any other failure to read the root
    propagates, so an unreadable store is never cleared.
    """
    for zarr_format in (3, 2):
        try:
            attributes = read_group_attributes(store, zarr_format=zarr_format)
        except json.JSONDecodeError:
            raise
        except ValueError:
            return {}
        if attributes is not None:
            return {
                key: value
                for key, value in attributes.items()
                if key not in OME_ROOT_KEYS
            }
    return {}


def _create_zarr_root(
    store: StoreLike,
    version: str,
    overwrite: bool,
    metadata_dict: dict,
    root_attributes: dict | None = None,
):
    """Create and configure the root Zarr group with proper attributes.

    The group metadata is written through the zarrista compatibility layer
    and the returned handle is a ``zarrista.Group``. An overwrite keeps the
    root attributes the writer does not own; ``root_attributes`` are written
    over them, and the OME keys over both.
    """
    zarr_format = 2 if version == "0.4" else 3
    attributes = _foreign_root_attrs(store) if overwrite else {}
    if root_attributes:
        attributes.update(root_attributes)
    attributes.update(_root_ome_attrs(metadata_dict, version))
    return create_zarrista_group(store, attributes, zarr_format, overwrite=overwrite)


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

    # The shard (outer chunk) grid and the internal chunk shape within each
    # shard; ``_handle_large_array_writing`` optimizes both to the array shape.
    sharding_kwargs = {
        "chunk_shape": internal_chunk_shape,
        "_shard_shape": shards,
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


def _unsupported_codec_message(subject: str) -> str:
    """Explain that *subject* cannot be written, and what to do instead.

    The engine encodes with the ``bytes`` codec plus one bytes-to-bytes
    compressor, so any other codec chain has to be written by a writer that
    implements it. ``metadata_only=True`` is what makes that a two-library
    job rather than a reimplementation: this writer lays down the OME
    metadata and the empty arrays, the other one fills them.
    """
    return (
        f"{subject} not supported by the zarrista write engine, which encodes "
        "with the bytes codec plus gzip, zstd, or blosc compression. To write "
        "another codec chain, create the store's metadata and empty arrays "
        "with to_ome_zarr(..., metadata_only=True), then create and fill each "
        "dataset with a writer that supports those codecs (for example "
        "zarr-python's zarr.create_array)."
    )


def _compression_chain(kwargs: dict) -> list | None:
    """Translate the public compression kwargs into a single ordered chain.

    ``None`` selects the engine default and an empty chain selects no
    compression. Mirroring the legacy engine's kwargs handling, an explicit
    ``compressor=None`` disables compression while ``compressors=None`` is
    indistinguishable from unset (default compression); the unset cases are
    detected with a sentinel. Only the array-to-bytes ("bytes") codec is
    dropped from a supplied chain since the writer always leads with one. The
    consumed keys are removed from ``kwargs``.

    ``filters`` and a ``serializer`` naming anything but the ``bytes`` codec
    name codec chains the engine cannot write, and raise rather than being
    dropped into an array whose metadata then disagrees with the request. A
    ``serializer`` that names ``bytes`` asks for what the writer already
    does, and is accepted as the no-op it is. Anything left over is not an
    array-creation option this writer knows; it warns, since silently
    ignoring it is how a caller comes to believe an inert argument works.
    """
    filters = kwargs.pop("filters", None)
    if filters:
        raise ValueError(_unsupported_codec_message("filters are"))
    serializer = kwargs.pop("serializer", None)
    if serializer is not None and not _is_bytes_codec(serializer):
        raise ValueError(
            _unsupported_codec_message("a serializer other than the bytes codec is")
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
    kwargs.pop("chunks", None)  # Remove chunks from kwargs since it's a positional arg
    if kwargs:
        warnings.warn(
            "to_ome_zarr() ignores the array-creation keyword argument(s) "
            + ", ".join(sorted(kwargs))
            + ". Supported: compressor (zarr format 2), compressors (zarr "
            "format 3), chunks. Ignoring them is deprecated and will become "
            "a TypeError.",
            DeprecationWarning,
            stacklevel=2,
        )
    return compression_chain


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
    compression_chain = _compression_chain(kwargs)

    scale_path = f"{store_path}/{path}"
    sharding_kwargs = (
        {} if shards is None else {"internal_chunk_shape": internal_chunk_shape}
    )
    try:
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
            **sharding_kwargs,
            **kwargs,
        )
    except OSError as e:
        # Writing evaluates the lazy dask graph, so a corrupt source (e.g. a
        # truncated TIFF) can surface here as a cryptic OSError (gh-issue-343).
        raise _array_write_error(path, e) from e


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


def _resolve_scale_chunks(
    arr: dask.array.Array,
    chunks: tuple[int, ...],
    sharding_kwargs: dict,
    internal_chunk_shape: tuple[int, ...] | None,
    shards: tuple[int, ...] | None,
) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, tuple[int, ...] | None
]:
    """Resolve the stored chunk grid of a scale array from the requested one.

    Returns the chunk shape to plan region writes on (the shard shape when
    sharding), the chunks clamped to the axes, the shard shape and the inner
    chunk shape of a shard.
    """

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

    if sharding_kwargs and "_shard_shape" in sharding_kwargs:
        # Sharding: clamp the shard shape to the axis length
        shard_shape = sharding_kwargs["_shard_shape"]
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

        # For region computation, use the optimized shard shape (the actual
        # stored chunk grid); the array is created from these values.
        zarr_chunk_shape = optimized_shard_shape
        chunks = optimized_shard_shape
        shards = optimized_shard_shape
    else:
        # No sharding
        chunks = tuple(
            [
                _find_optimal_chunk_size(c[0], arr.shape[i])
                for i, c in enumerate(arr.chunks)
            ]
        )
        zarr_chunk_shape = chunks

    return zarr_chunk_shape, chunks, shards, internal_chunk_shape


def _create_scale_array(
    store_path: str,
    path: str,
    arr: dask.array.Array,
    chunks: tuple[int, ...],
    sharding_kwargs: dict,
    zarr_format: int,
    dimension_names: tuple[str, ...] | None,
    internal_chunk_shape: tuple[int, ...] | None,
    shards: tuple[int, ...] | None,
    **kwargs,
):
    """Create the empty array of one scale level.

    Chunk, shard and compression settings are resolved the way the region
    writer resolves them, so an array created for a metadata-only store is
    identical to one the writer fills itself.
    """
    _, chunks, shards, internal_chunk_shape = _resolve_scale_chunks(
        arr, chunks, sharding_kwargs, internal_chunk_shape, shards
    )
    compression_chain = _compression_chain(dict(kwargs))
    return _create_dataset(
        f"{store_path}/{path}",
        arr.shape,
        arr.dtype,
        chunks,
        zarr_format,
        dimension_names,
        internal_chunk_shape if shards is not None else None,
        compression_chain,
    )


def _handle_large_array_writing(
    image,
    arr: dask.array.Array,
    path: str,
    dims: tuple[str, ...],
    dim_factors: dict[str, int],
    chunks: tuple[int, ...],
    sharding_kwargs: dict,
    store_path: str,
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
    shrink_factors = []
    for dim in dims:
        if dim in dim_factors:
            shrink_factors.append(dim_factors[dim])
        else:
            shrink_factors.append(1)

    zarr_chunk_shape, chunks, shards, internal_chunk_shape = _resolve_scale_chunks(
        arr, chunks, sharding_kwargs, internal_chunk_shape, shards
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
        arr_region = _prep_for_write(arr_region)
        optimized = dask.array.Array(
            dask.array.optimize(
                arr_region.__dask_graph__(), arr_region.__dask_keys__()
            ),
            arr_region.name,
            arr_region.chunks,
            meta=arr_region,
        )

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


def _is_generated_level(multiscales: NgffMultiscales, index: int) -> bool:
    """Whether level ``index`` still carries the data to_multiscales gave it."""
    keys = multiscales.generated_data_keys
    if keys is None or index >= len(keys):
        return False
    data = multiscales.images[index].data
    return getattr(data, "name", None) == keys[index]


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
    # Re-deriving the next level from the level just written keeps the task
    # graph shallow. Only a level still holding the data to_multiscales
    # generated for it is re-derived; anything else is written as given.
    if (
        multiscales.scale_factors
        and multiscales.method
        and multiscales.chunks
        and _is_generated_level(multiscales, index + 1)
    ):
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []

        image.data = open_lazy_array(store, path)

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
    metadata_only: bool = False,
    start_level: int = 0,
    **kwargs,
) -> None:
    """
    Write an image pixel array and metadata to a Zarr store with the OME-NGFF standard data model.

    :param store: Store or path to directory in file system. If the path ends with .ozx, writes an RFC-9
    compliant zipped OME-Zarr file.
    :type  store: StoreLike

    :param multiscales: NgffMultiscales OME-NGFF image pixel data and metadata. Can be generated with ngff_zarr.to_multiscales.
        Its ``root_attributes`` are written beside the OME metadata at the root of the store.
    :type  multiscales: NgffMultiscales

    :param version: OME-Zarr specification version. For .ozx files, version 0.5 is required.
    :type  version: str, optional

    :param overwrite: If True, delete any pre-existing data in `store` before creating groups.
    :type  overwrite: bool, optional

    :param use_tensorstore: Deprecated. If True, write arrays with the zarrista
        backend's fast direct-write path (tensorstore is no longer used).
    :type  use_tensorstore: bool, optional

    :param chunk_store: No longer supported; chunks and metadata are always
        written to `store`. Passing a value raises ``TypeError``.
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

    :param metadata_only: If True, write the OME-Zarr metadata and create every scale
        array with its shape, dtype, chunks, shards and codecs, but compute and write no
        pixel data. Nothing in the dask graphs is evaluated, so the call returns at once
        whatever the image size. Fill the arrays afterwards with any Zarr writer; a chunk
        grid that follows the regions each writer produces gives every chunk a single
        writer and needs no locking. Not available for .ozx output.
    :type  metadata_only: bool, optional

    :param start_level: Index of the first scale level to write. The levels below it must
        already exist in the store with the shape and dtype the multiscales describes; they
        are read, never rewritten, and the first written level is derived from the last of
        them on disk. Requires ``overwrite=False``. Root attributes the writer does not own
        are kept, the multiscales entry is rewritten in full, and a level at or above
        ``start_level`` that already exists is replaced.
    :type  start_level: int, optional

    :param **kwargs: Array-creation options: `compressor` (zarr format 2),
        `compressors` (zarr format 3), and `chunks`. The engine encodes with the
        `bytes` codec plus gzip, zstd, or blosc compression, so `filters` and a
        `serializer` naming any other codec raise `ValueError` — write those
        chains by creating the store with ``metadata_only=True`` and filling the
        arrays with a writer that implements them. Any other keyword is ignored
        with a `DeprecationWarning` and will become a `TypeError`.
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
        if metadata_only:
            raise ValueError(
                "metadata_only=True is not available for .ozx output: a zipped "
                "OME-Zarr is written in one piece, with its data."
            )
        if start_level:
            raise ValueError(
                "start_level is not available for .ozx output: a zipped OME-Zarr "
                "is written in one piece."
            )
        if version != "0.5":
            raise ValueError(
                "RFC-9 zipped OME-Zarr (.ozx) requires OME-Zarr version 0.5. "
                f"Got version '{version}'. Please set version='0.5'."
            )

        # Default chunks_per_shard to 2 for .ozx files if not specified
        if chunks_per_shard is None:
            chunks_per_shard = 2

        # Write to a temporary directory store first, then zip it into the
        # requested .ozx archive (zarrista can only read zip archives).
        temp_dir = tempfile.mkdtemp()
        try:
            _to_ngff_zarr_impl(
                temp_dir,
                multiscales,
                version=version,
                overwrite=overwrite,
                use_tensorstore=False,
                chunk_store=None,
                progress=progress,
                chunks_per_shard=chunks_per_shard,
                scale_strategy=scale_strategy,
                **kwargs,
            )

            # Write temp store to .ozx file
            write_store_to_zip(temp_dir, store, version=version)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        metadata_only=metadata_only,
        start_level=start_level,
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
    metadata_only: bool = False,
    start_level: int = 0,
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

    if chunk_store is not None:
        raise TypeError(
            "chunk_store is no longer supported: chunks and metadata are "
            "always written to the same local directory store."
        )

    # Setup and validation. All writes — group/attribute metadata and
    # arrays — go through the zarrista compat layer, which requires a local
    # directory path store.
    store_path = str(normalize_store(store))

    _validate_ngff_parameters(version, chunks_per_shard)
    if not 0 <= start_level < len(multiscales.images):
        raise ValueError(
            f"start_level={start_level} is out of range for "
            f"{len(multiscales.images)} scale levels."
        )
    if start_level and overwrite:
        raise ValueError(
            "start_level keeps the levels already in the store, which "
            "overwrite=True would delete. Pass overwrite=False."
        )
    if overwrite:
        _guard_overwrite_of_source_store(multiscales, store_path)
    root_attributes = _check_root_attributes(multiscales.root_attributes)
    metadata, dimension_names, _ = _prepare_metadata(multiscales, version)
    _gate_top_level_transforms(metadata, version)
    if start_level:
        _guard_rewrite_of_read_levels(
            multiscales, store_path, metadata.datasets, start_level
        )
    metadata_dict = asdict(metadata)
    metadata_dict = _pop_metadata_optionals(metadata_dict)
    metadata_dict["@type"] = "ngff:Image"

    # Format parameters
    zarr_format = 2 if version == "0.4" else 3

    # Create Zarr root
    # Appending names only the levels it keeps until the new arrays are in
    # place, so an interrupted call leaves a store that reads as those levels.
    if start_level:
        # Read before the rewrite: at zarr format 3 _create_zarr_root writes a
        # fresh zarr.json without consolidated_metadata.
        was_consolidated = has_consolidated_metadata(store, zarr_format)
        kept = copy.deepcopy(metadata_dict)
        kept["datasets"] = kept["datasets"][:start_level]
        _create_zarr_root(store, version, overwrite, kept, root_attributes)
        if was_consolidated:
            _zarrista_consolidate_metadata(store, zarr_format)
    else:
        _create_zarr_root(store, version, overwrite, metadata_dict, root_attributes)

    if version == "0.4" and kwargs.get("compressors") is not None:
        raise ValueError(
            "The argument `compressors` is not supported for OME-Zarr version 0.4 "
            "(Zarr v2). Use `compressor` instead."
        )

    # Process each scale level
    nscales = len(multiscales.images)
    if progress and not metadata_only:
        progress.add_multiscales_task("[green]Writing scales", nscales)

    next_image = multiscales.images[0]
    dims = next_image.dims
    previous_dim_factors = dict.fromkeys(dims, 1)

    for index in range(nscales):
        if progress and not metadata_only:
            progress.update_multiscales_task_completed(index + 1)

        image = next_image
        arr = image.data
        path = metadata.datasets[index].path
        parent = str(PurePosixPath(path).parent)

        # Calculate dimension factors
        if index > 0 and index < nscales - 1 and multiscales.scale_factors:
            dim_factors = _dim_scale_factors(
                dims, multiscales.scale_factors[index], previous_dim_factors
            )
        else:
            dim_factors = dict.fromkeys(dims, 1)
        previous_dim_factors = dim_factors

        if index < start_level:
            _bind_existing_level(image, store_path, path, index)
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
            continue

        # Create parent groups if needed
        if parent not in (".", "/"):
            if (
                not start_level
                and not overwrite
                and read_group_attributes(store, parent, zarr_format=zarr_format)
                is not None
            ):
                raise ValueError(
                    f"The store already holds scale level {index} at "
                    f"'{parent}'. Pass start_level to append levels after the "
                    "ones it holds, or overwrite=True to replace the store."
                )
            create_zarrista_subgroup(
                store,
                parent,
                {"_ARRAY_DIMENSIONS": list(image.dims)},
                zarr_format,
            )
        if start_level:
            _remove_existing_level(store_path, path)

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

        if metadata_only:
            # The levels keep the shapes to_multiscales gave them, which are
            # the shapes the datasets metadata describes.
            _create_scale_array(
                store_path,
                path,
                arr,
                chunks,
                sharding_kwargs,
                zarr_format,
                dimension_names,
                internal_chunk_shape,
                shards,
                **kwargs,
            )
            if index + 1 < nscales:
                next_image = multiscales.images[index + 1]
            continue

        # Determine write method based on memory requirements
        if memory_usage(image) > config.memory_target:
            _handle_large_array_writing(
                image,
                arr,
                path,
                dims,
                dim_factors,
                chunks,
                sharding_kwargs,
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

            # For small arrays, write in one go, clamping a shard wider than
            # the array as the other two paths do. Only the sharding case goes
            # through the resolver: its other branch re-derives chunks from the
            # leading dask chunk, partial after a slice (issue #488).
            small_chunks, small_shards, small_internal_chunk_shape = (
                chunks,
                shards,
                internal_chunk_shape,
            )
            if sharding_kwargs and "_shard_shape" in sharding_kwargs:
                (
                    _,
                    small_chunks,
                    small_shards,
                    small_internal_chunk_shape,
                ) = _resolve_scale_chunks(
                    arr, chunks, sharding_kwargs, internal_chunk_shape, shards
                )
            region = tuple([slice(arr.shape[i]) for i in range(arr.ndim)])
            _write_array_with_zarrista(
                store_path,
                path,
                arr,
                small_chunks,
                small_shards,
                small_internal_chunk_shape,
                zarr_format,
                dimension_names,
                region,
                full_array_shape=arr.shape,
                create_dataset=True,  # Always create for small arrays
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

    if start_level:
        _create_zarr_root(store, version, False, metadata_dict, root_attributes)

    # Clean up callbacks
    for image in multiscales.images:
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []

    # Consolidate metadata
    _zarrista_consolidate_metadata(store, zarr_format)
