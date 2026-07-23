# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Upgrade an OME-Zarr store from one specification version to another.

`upgrade_ome_zarr` changes the OME-Zarr specification version recorded in a
store. It offers two modes:

**In-place metadata-only rewrite.** When no `output` is given (or `output`
resolves to the same store as `input`) and the source and target share the same
underlying Zarr format -- for example 0.5 -> 0.6, both Zarr v3 -- only the root
group's metadata (`zarr.json`) is rewritten. Every array chunk on disk is left
byte-for-byte untouched. This is the fix for issue #219, where the previous
"read, erase, re-write" approach destroyed the array data it was meant to
preserve.

In-place upgrades that cross the Zarr v2<->v3 boundary in the *upgrade*
direction (OME-Zarr 0.4->0.5 or 0.4->0.6) are also metadata-only: each array's
Zarr v3 ``zarr.json`` is given a ``v2`` chunk-key encoding matching the source
separator so the existing chunk binaries resolve unchanged (see
:func:`_rewrite_v2_group_to_v3`). The reverse, an in-place *downgrade* across
the boundary (0.5/0.6->0.4), cannot preserve chunk keys and is rejected with a
clear ``ValueError``; pass an ``output`` store instead.

**Write-to-new-store conversion.** When an `output` store distinct from `input`
is given, the source is read lazily and re-written to `output` at the requested
version through the standard, tested write pipeline. Every supported transition
(0.4<->0.5<->0.6) works in this mode. Array data streams from the source; the
source store is never erased.

Examples
--------
Upgrade a 0.5 store to 0.6 in place, keeping all array chunks:

    upgrade_ome_zarr("image.ome.zarr", version="0.6")

Write an upgraded copy to a new store:

    upgrade_ome_zarr("image_v04.ome.zarr", "image_v05.ome.zarr", version="0.5")
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import packaging.version
import zarr
import zarr.errors
import zarr.storage

from ._supported_versions import NgffVersion
from ._zarr_types import StoreLike
from .from_ngff_zarr import (
    REMOTE_URL_SCHEMES,
    _is_remote_store,
    from_ome_zarr,
    zarr_version_major,
)
from .to_ngff_zarr import (
    _numcodecs_to_zarr_v3_codec,
    _pop_metadata_optionals,
    _write_root_ome_attrs,
    to_ome_zarr,
)

if TYPE_CHECKING:
    from .multiscales import NgffMultiscales


def _normalize_target_version(version: str | NgffVersion) -> str:
    """Return the API version string (``"0.4"``/``"0.5"``/``"0.6"``).

    Both the API alias ``"0.6"`` and the on-disk development string
    ``"0.6.dev4"`` normalize to ``"0.6"`` so the value can be handed to
    ``Metadata.to_version`` (which only knows the three released versions).
    """
    nv = NgffVersion(version)
    if nv in (NgffVersion.V06, NgffVersion.V06dev4):
        return "0.6"
    return nv.value


def _zarr_format_for_version(version: str) -> int:
    """OME-Zarr < 0.5 lives in Zarr v2; 0.5 and later live in Zarr v3."""
    if packaging.version.parse(version) < packaging.version.parse("0.5"):
        return 2
    return 3


def _store_fspath(store: StoreLike) -> Path | None:
    """Best-effort local filesystem path for ``store``, else ``None``.

    Used only to detect input/output aliasing. Remote and in-memory stores
    return ``None`` and are therefore treated as distinct unless they are the
    very same object.
    """
    if isinstance(store, (str, Path)):
        if _is_remote_store(str(store)):
            return None
        return Path(store).resolve()
    if _is_remote_store(store):
        return None
    for attr in ("root", "path"):
        val = getattr(store, attr, None)
        if isinstance(val, (str, Path)):
            return Path(val).resolve()
    dir_path = getattr(store, "dir_path", None)
    if callable(dir_path):
        try:
            return Path(dir_path()).resolve()
        except Exception:
            return None
    return None


def _stores_are_same(input: StoreLike, output: StoreLike | None) -> bool:
    """Whether ``output`` designates the same store as ``input``.

    ``output is None`` always means "same store" (an in-place request). Two
    string/URL forms are compared directly; local paths are compared by their
    resolved filesystem path so ``foo`` and ``./foo/`` match.
    """
    if output is None or input is output:
        return True
    if isinstance(input, (str, Path)) and isinstance(output, (str, Path)):
        in_s, out_s = str(input), str(output)
        if _is_remote_store(in_s) or _is_remote_store(out_s):
            return in_s.rstrip("/") == out_s.rstrip("/")
    in_path = _store_fspath(input)
    out_path = _store_fspath(output)
    return in_path is not None and in_path == out_path


def _resolve_write_store(store: StoreLike, storage_options: dict | None) -> StoreLike:
    """Wrap a remote URL output in an ``FsspecStore`` when needed.

    ``to_ome_zarr`` has no ``storage_options`` parameter, so a remote output URL
    combined with ``storage_options`` is resolved to a store object here, exactly
    as ``from_ome_zarr`` resolves remote inputs. Local and in-memory stores pass
    through unchanged.
    """
    if (
        isinstance(store, str)
        and storage_options is not None
        and store.startswith(REMOTE_URL_SCHEMES)
        and zarr_version_major >= 3
        and hasattr(zarr.storage, "FsspecStore")
    ):
        return zarr.storage.FsspecStore.from_url(store, storage_options=storage_options)
    return store


def _read_root_attrs(store: StoreLike, storage_options: dict | None) -> dict:
    """Read the root-group attributes of ``store`` without loading arrays.

    Mirrors the auto-detection (and Zarr v2 fallback) that ``from_ome_zarr``
    performs internally: a Zarr v2 (OME-Zarr 0.4) store opened without an
    explicit format on zarr-python 3 yields an empty root, so it is retried as
    Zarr v2. Used for both source-version detection and HCS/container rejection.
    """
    open_store = store
    if (
        isinstance(store, str)
        and storage_options is not None
        and store.startswith(REMOTE_URL_SCHEMES)
        and zarr_version_major >= 3
        and hasattr(zarr.storage, "FsspecStore")
    ):
        open_store = zarr.storage.FsspecStore.from_url(
            store, storage_options=storage_options
        )

    def _attrs(**kwargs) -> dict:
        try:
            return zarr.open_group(open_store, mode="r", **kwargs).attrs.asdict()
        except zarr.errors.GroupNotFoundError:
            # No group in the requested Zarr format -- e.g. a Zarr v2 (OME-Zarr
            # 0.4) store opened without an explicit format on zarr-python 3.
            # Signal "not found in this format" so the caller can retry as Zarr
            # v2; genuine failures (missing, corrupt, or permission-restricted
            # stores) propagate instead of being masked as an empty-attrs
            # version-detection error.
            return {}

    root_attrs = _attrs()
    if not root_attrs and zarr_version_major >= 3:
        root_attrs = _attrs(zarr_format=2)

    return root_attrs


def _read_source_version(store: StoreLike, storage_options: dict | None) -> str:
    """Detect the on-disk OME-Zarr version string of ``store``.

    ``from_ome_zarr`` always normalizes the metadata it returns to v0.6, so the
    original version cannot be recovered from the returned object. It is read
    here straight from the root attributes.
    """
    from .parse_metadata import _detect_version

    return _detect_version(_read_root_attrs(store, storage_options)).value


def _validate_target_version(target_version: str, requested: str | NgffVersion) -> None:
    """Reject a target ``version`` that ``upgrade_ome_zarr`` cannot produce.

    ``NgffVersion`` also admits the pre-0.4 drafts (0.1--0.3) that live in
    ``SUPPORTED_VERSIONS`` for read compatibility, but the metadata
    ``to_version`` chains only convert among 0.4/0.5/0.6. Fail early with an
    actionable message rather than deep in a conversion.
    """
    if target_version not in ("0.4", "0.5", "0.6"):
        raise ValueError(
            f"Unsupported target version {requested!r}. upgrade_ome_zarr() can "
            "upgrade to OME-Zarr 0.4, 0.5, or 0.6."
        )


def _reject_plate_or_container(root_attrs: dict) -> None:
    """Reject an HCS plate root or bioformats2raw container root.

    ``upgrade_ome_zarr`` upgrades a single image. Whole-plate / whole-container
    upgrades are out of scope for this effort: each contained image is upgraded
    by pointing at its own path. This reuses the same detection predicates
    ``from_ome_zarr`` relies on so the two agree on what a plate/container is.
    """
    from .parse_metadata import _is_bioformats2raw, _is_hcs_plate

    if _is_hcs_plate(root_attrs):
        raise ValueError(
            "The input is an HCS (High Content Screening) plate root. "
            "upgrade_ome_zarr() upgrades one image at a time; upgrade each "
            "well/field image by its own path (e.g. 'plate.ome.zarr/A/1/0'). "
            "Whole-plate upgrades are out of scope."
        )
    if _is_bioformats2raw(root_attrs):
        raise ValueError(
            "The input is a bioformats2raw container root. "
            "upgrade_ome_zarr() upgrades one image at a time; upgrade each "
            "contained image by its own path (e.g. 'container.ome.zarr/0'). "
            "Whole-container upgrades are out of scope."
        )


def _upgrade_in_place(
    store: StoreLike,
    multiscales: NgffMultiscales,
    target_version: str,
    target_zarr_format: int,
) -> None:
    """Rewrite only the root-group metadata of ``store`` to ``target_version``.

    Opens the existing root group for read/write (``mode="a"``) so the array
    chunks are never rewritten -- the crux of the issue #219 fix. Calling
    ``to_ome_zarr(..., overwrite=True)`` on the source instead would erase those
    arrays.
    """
    root = zarr.open_group(store, mode="a", zarr_format=target_zarr_format)

    # Convert the already-read metadata to the target version, preserving the
    # existing ``type``/``metadata``/``omero`` fields. ``_prepare_metadata`` is
    # deliberately avoided: it resets ``type``/``metadata`` to ``None`` for any
    # store whose method type is not a recognized ``Methods`` value.
    new_metadata = multiscales.metadata.to_version(target_version)
    metadata_dict = asdict(new_metadata)
    metadata_dict = _pop_metadata_optionals(metadata_dict)
    metadata_dict["@type"] = "ngff:Image"

    _write_root_ome_attrs(root, metadata_dict, target_version)


def _parent_group_paths(array_paths: list[str]) -> list[str]:
    """Return every intermediate group path implied by ``array_paths``.

    For a dataset laid out at ``scale0/image`` the intermediate group
    ``scale0`` must gain a Zarr v3 ``zarr.json`` (and keep its ``.zattrs`` such
    as ``_ARRAY_DIMENSIONS``). Flat arrays (``"0"``) imply no intermediate
    group. Returned sorted so parents precede children.
    """
    groups: set[str] = set()
    for path in array_paths:
        parts = PurePosixPath(path).parts
        for i in range(1, len(parts)):
            groups.add("/".join(parts[:i]))
    return sorted(groups)


def _delete_v2_sidecars(store: StoreLike) -> None:
    """Delete the Zarr v2 metadata sidecars from ``store``, sparing chunks.

    Removes every ``.zgroup``/``.zattrs``/``.zarray``/``.zmetadata`` key so the
    store is cleanly Zarr v3. Chunk data files never begin with a dot, so they
    are never matched. Filesystem-backed stores are pruned by ``unlink``;
    other stores (e.g. ``MemoryStore``) go through the async store API, the
    same pattern used by ``rfc9_zip``.
    """
    sidecar_names = {".zgroup", ".zattrs", ".zarray", ".zmetadata"}

    fspath = _store_fspath(store)
    if fspath is not None and fspath.is_dir():
        for path in fspath.rglob("*"):
            if path.is_file() and path.name in sidecar_names:
                path.unlink()
        return

    # A string/Path that is not an existing directory: nothing to prune.
    if isinstance(store, (str, Path)):
        return

    import asyncio

    async def _prune() -> None:
        keys = [key async for key in store.list()]
        for key in keys:
            if key.rsplit("/", 1)[-1] in sidecar_names:
                await store.delete(key)

    try:
        asyncio.run(_prune())
    except RuntimeError as exc:
        if "running event loop" not in str(exc):
            raise
        raise RuntimeError(
            "In-place cross-format upgrade of an in-memory store is not "
            "supported from within a running event loop (e.g. a Jupyter "
            "notebook). Pass an `output` store to use the write-to-new-store "
            "path instead."
        ) from exc


def _rewrite_v2_group_to_v3(
    store: StoreLike,
    new_metadata,
    target_version: str,
) -> None:
    """Rewrite a Zarr v2 (OME-Zarr 0.4) group to Zarr v3 in place, keeping chunks.

    Each array's chunk binaries are left exactly where they are: the new Zarr v3
    ``zarr.json`` is given a ``v2`` ``chunk_key_encoding`` whose separator equals
    the source array's ``dimension_separator``, so the pre-existing chunk keys
    (e.g. ``0/0/0/0`` or ``0.0.0.0``) resolve unchanged. Only metadata is written
    or removed; no chunk file is ever moved, rewritten, or deleted. This is the
    cross-format analogue of the in-place issue #219 fix.

    An array whose Zarr v2 encoding cannot be faithfully represented in Zarr v3
    (Fortran memory order, filters, or a compressor with no v3 codec equivalent)
    raises ``ValueError`` naming the array, rather than silently corrupting data;
    the caller is directed to write-to-new-store instead.
    """
    import warnings

    dim_names = list(new_metadata.dimension_names)
    array_paths = [dataset.path for dataset in new_metadata.datasets]

    # --- Read phase: capture all Zarr v2 metadata before writing anything. ---
    array_specs: list[dict] = []
    for path in array_paths:
        source = zarr.open_array(store, path=path, mode="r", zarr_format=2)
        meta = source.metadata

        order = getattr(meta, "order", "C")
        if order == "F":
            raise ValueError(
                f"Array '{path}' uses Fortran ('F') memory order, which an "
                "in-place Zarr v2->v3 metadata rewrite cannot preserve without "
                "rewriting chunks. Pass an `output` store to write a new upgraded "
                "copy instead."
            )

        filters = getattr(meta, "filters", None)
        if filters:
            raise ValueError(
                f"Array '{path}' declares Zarr v2 filters ({filters!r}) that have "
                "no faithful Zarr v3 codec equivalent for an in-place rewrite. "
                "Pass an `output` store to write a new upgraded copy instead."
            )

        compressor = getattr(meta, "compressor", None)
        if compressor is None:
            # No compression: emit only the (implicit) bytes codec. Passing
            # ``compressors=None`` avoids create_array's default zstd, which
            # would misread the raw chunk bytes.
            compressors = None
        else:
            v3_codec = _numcodecs_to_zarr_v3_codec(compressor)
            if v3_codec is None:
                raise ValueError(
                    f"Array '{path}' uses compressor {compressor!r}, which has no "
                    "faithful Zarr v3 codec equivalent. Pass an `output` store to "
                    "write a new upgraded copy instead."
                )
            compressors = [v3_codec]

        separator = getattr(meta, "dimension_separator", None) or "."
        array_specs.append(
            {
                "path": path,
                "shape": source.shape,
                "chunks": source.chunks,
                "dtype": source.dtype,
                "fill_value": meta.fill_value,
                "separator": separator,
                "compressors": compressors,
                "attrs": dict(source.attrs),
            }
        )

    group_paths = _parent_group_paths(array_paths)
    group_attrs: dict[str, dict] = {}
    for group_path in group_paths:
        try:
            source_group = zarr.open_group(
                store, path=group_path, mode="r", zarr_format=2
            )
            group_attrs[group_path] = dict(source_group.attrs)
        except Exception:
            group_attrs[group_path] = {}

    # --- Write phase: materialize the Zarr v3 hierarchy. Chunks are untouched. ---
    root = zarr.open_group(store, mode="a", zarr_format=3)

    for group_path in group_paths:
        group = zarr.open_group(store, path=group_path, mode="a", zarr_format=3)
        for key, value in group_attrs[group_path].items():
            group.attrs[key] = value

    for spec in array_specs:
        array = root.create_array(
            name=spec["path"],
            shape=spec["shape"],
            chunks=spec["chunks"],
            dtype=spec["dtype"],
            fill_value=spec["fill_value"],
            compressors=spec["compressors"],
            chunk_key_encoding={
                "name": "v2",
                "configuration": {"separator": spec["separator"]},
            },
            dimension_names=dim_names,
            overwrite=False,
        )
        for key, value in spec["attrs"].items():
            array.attrs[key] = value

    # Root OME attributes, exactly as to_ome_zarr / _upgrade_in_place emit them.
    metadata_dict = asdict(new_metadata)
    metadata_dict = _pop_metadata_optionals(metadata_dict)
    metadata_dict["@type"] = "ngff:Image"
    _write_root_ome_attrs(root, metadata_dict, target_version)

    # Drop the now-obsolete Zarr v2 sidecars; chunk files are left in place.
    _delete_v2_sidecars(store)

    # Consolidate so the upgraded store matches a freshly written Zarr v3 store.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        zarr.consolidate_metadata(store, zarr_format=3)


def upgrade_ome_zarr(
    input: StoreLike,
    output: StoreLike | None = None,
    *,
    version: str | NgffVersion = "0.6",
    validate: bool = False,
    storage_options: dict | None = None,
    overwrite: bool = True,
) -> None:
    """Upgrade an OME-Zarr store to a different specification version.

    :param input: Source store or path to read.
    :type  input: StoreLike

    :param output: Destination store or path. If ``None`` (or the same store as
        ``input``), the upgrade is performed in place: only metadata is rewritten
        and every array chunk on disk is left byte-for-byte untouched (this
        includes the cross-format 0.4->0.5/0.6 case, which rewrites the array and
        group metadata to Zarr v3 while preserving chunks). An in-place downgrade
        across the Zarr v2<->v3 boundary (0.5/0.6->0.4) cannot preserve chunk keys
        and raises ``ValueError``. If a distinct store, an upgraded copy is
        written there via the standard write pipeline.
    :type  output: StoreLike, optional

    :param version: Target OME-Zarr specification version. Accepts a string such
        as ``"0.6"`` or an ``NgffVersion`` member.
    :type  version: str or NgffVersion, optional

    :param validate: If ``True``, validate the source metadata against the NGFF
        schema while reading.
    :type  validate: bool, optional

    :param storage_options: Storage options for remote ``input``/``output`` URLs
        (zarr-python 3+).
    :type  storage_options: dict, optional

    :param overwrite: For the write-to-new-store mode, whether to overwrite any
        pre-existing data at ``output``. Ignored for in-place upgrades, which
        never overwrite array data.
    :type  overwrite: bool, optional

    :return: ``None``. The upgraded metadata is written to ``output`` (or, for an
        in-place upgrade, back to ``input``).
    """
    from .parse_metadata import _detect_version

    target_version = _normalize_target_version(version)
    _validate_target_version(target_version, version)
    target_zarr_format = _zarr_format_for_version(target_version)

    root_attrs = _read_root_attrs(input, storage_options)
    # Whole-plate / whole-container upgrades are out of scope; reject early with
    # guidance toward the per-image path form.
    _reject_plate_or_container(root_attrs)

    source_version = _detect_version(root_attrs).value
    source_zarr_format = _zarr_format_for_version(source_version)
    same_store = _stores_are_same(input, output)

    # No-op: the source spec version already equals the requested target.
    # Normalize both to the API version so an on-disk "0.6.dev4" matches a
    # requested "0.6".
    if _normalize_target_version(source_version) == target_version and same_store:
        # In-place no-op: leave the store bit-for-bit unchanged (no read, no
        # write). When an ``output`` is given for a same-version request we fall
        # through instead, so the user still gets their requested new store.
        return

    # Read the source once, using the detected version so the correct parser and
    # Zarr format are selected. The returned metadata is normalized to v0.6.
    multiscales = from_ome_zarr(
        input,
        validate=validate,
        version=source_version,
        storage_options=storage_options,
    )

    if same_store:
        if source_zarr_format == target_zarr_format:
            # Same underlying Zarr format (0.5<->0.6): rewrite only root metadata.
            _upgrade_in_place(input, multiscales, target_version, target_zarr_format)
        elif source_zarr_format == 2 and target_zarr_format == 3:
            # Cross-format upgrade (0.4 -> 0.5/0.6): rewrite array + group
            # metadata to Zarr v3 while preserving every chunk file on disk.
            new_metadata = multiscales.metadata.to_version(target_version)
            _rewrite_v2_group_to_v3(input, new_metadata, target_version)
        else:
            # Cross-format downgrade (0.5/0.6 -> 0.4) in place: Zarr v3 default
            # chunk keys differ from Zarr v2, so chunks cannot be preserved.
            raise ValueError(
                "In-place downgrade across the Zarr v2<->v3 boundary "
                f"(OME-Zarr {source_version} -> {target_version}) cannot preserve "
                "chunk files: Zarr v3 chunk keys differ from Zarr v2. Pass an "
                "`output` store to write a new downgraded copy safely."
            )
        return

    # Write-to-new-store: reuse the tested read/write pipeline. ``to_ome_zarr``
    # re-derives the target-version metadata and streams the source arrays
    # lazily, so all supported transitions work and the source is never erased.
    output_store = _resolve_write_store(output, storage_options)
    to_ome_zarr(
        output_store,
        multiscales,
        version=target_version,
        overwrite=overwrite,
    )
