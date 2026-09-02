# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path

import packaging.version

from ._remote_reader import (
    RemoteZarrStore,
    open_remote_node,
    remote_read_available,
)
from ._store_types import StoreLike
from ._zarrista_utils import (
    OME_ROOT_KEYS,
    _is_bytes_mapping,
    _is_local_path,
    open_local_node,
    open_ozx_store,
)
from .multiscales import NgffMultiscales
from .rfc9_zip import is_ozx_path, read_ozx_version

# Supported remote URL schemes for storage
REMOTE_URL_SCHEMES = ("s3://", "gs://", "azure://", "http://", "https://")

# fsspec filesystem protocols that imply a remote backend is required. Used to
# recognize a remote store even after a URL has been wrapped in an FsspecStore,
# whose ``str()`` is ``"<FsspecStore(...)>"`` rather than the original URL.
REMOTE_FS_PROTOCOLS = frozenset(
    {"s3", "s3a", "gcs", "gs", "az", "abfs", "abfss", "adl", "http", "https"}
)


def _is_remote_store(store) -> bool:
    """Whether ``store`` refers to remote storage needing an fsspec backend.

    Handles plain URL strings/paths as well as zarr-python 3 ``FsspecStore``
    instances (built here when ``storage_options`` is provided, or passed in
    directly), whose ``str()`` does not start with the URL scheme.
    """
    if str(store).startswith(REMOTE_URL_SCHEMES):
        return True
    # ``FsspecStore.path`` keeps the full URL for http(s) but strips the
    # protocol for s3/gcs/azure, so fall back to the filesystem's protocol.
    path = getattr(store, "path", "")
    if isinstance(path, str) and path.startswith(REMOTE_URL_SCHEMES):
        return True
    fs = getattr(store, "fs", None)
    protocol = getattr(fs, "protocol", ())
    protocols = {protocol} if isinstance(protocol, str) else set(protocol or ())
    return not protocols.isdisjoint(REMOTE_FS_PROTOCOLS)


def _remote_backend_import_error(store, original: ImportError | None) -> ImportError:
    """Build an actionable ImportError pointing at the ``[remote]`` extra."""
    message = (
        f"Reading from the remote store '{store}' requires additional "
        "packages that are not installed. Install them with:\n\n"
        '    pip install "ngff-zarr[remote]"\n\n'
        "This provides the obstore backend for http(s), S3, GCS, and Azure."
    )
    if original is not None:
        message += f" Original error: {original}"
    return ImportError(message)


class _NoValidZarrGroupError(ValueError):
    """No zarr group exists in the store (any format, any backend).

    A ``ValueError`` subclass so existing callers are unaffected; typed so
    callers that treat "no group here" as an empty result (e.g. the upgrade
    module's root-attribute probe) need not match on the message text.
    """


def _group_not_found_error(store) -> ValueError:
    """The shared no-group-here error, identical across read backends."""
    store_path = str(store)
    return _NoValidZarrGroupError(
        f"No valid Zarr group found at '{store_path}'. "
        "This error typically occurs when:\n"
        "  1. The path does not contain a valid Zarr store\n"
        "  2. The Zarr store is empty or corrupted\n"
        "  3. The download was incomplete\n"
        "  4. The store contains a Zarr array at the root instead of a group\n\n"
        "For OME-Zarr files, the root must be a Zarr group "
        "containing multiscale metadata. "
        "Please verify that the input path points to a valid OME-Zarr store."
    )


def _array_at_root_error(store) -> ValueError:
    """The shared array-at-root error, identical across read backends."""
    store_path = str(store)
    return ValueError(
        f"The Zarr store at '{store_path}' contains an array at the root level, "
        "but OME-Zarr requires a group structure with multiscale metadata. "
        "Single Zarr arrays cannot be directly converted to OME-Zarr format."
    )


def _open_local_root(store, version: str | None):
    """Open the root group of a local directory store through the compat layer.

    Mirrors the zarr-python branch's semantics: an explicit *version* pins the
    zarr format (< 0.5 -> format 2, otherwise 3); auto-detection prefers
    format 3 but falls back to the format 2 documents when the ``zarr.json``
    group has empty attributes (the spurious-``zarr.json`` case) or is absent.
    """
    if version:
        zarr_format = (
            2
            if packaging.version.parse(version) < packaging.version.parse("0.5")
            else 3
        )
        node = open_local_node(store, zarr_format=zarr_format)
    else:
        node = open_local_node(store, zarr_format=3)
        if node is None or (not hasattr(node, "shape") and not node.attrs):
            # Only take the format 2 node when one is actually there: a
            # format 3 group whose attributes are empty is still valid, and
            # overwriting it with None would report "group not found".
            fallback = open_local_node(store, zarr_format=2)
            if fallback is not None:
                node = fallback
    if node is None:
        raise _group_not_found_error(store)
    if hasattr(node, "shape"):
        raise _array_at_root_error(store)
    return node


def _open_remote_root(store: RemoteZarrStore, version: str | None):
    """Open the root group of a remote store via zarrista's async API.

    zarrista auto-detects the zarr format (preferring format 3, with the
    spurious-``zarr.json`` v2 attribute fallback handled inside
    :func:`~._remote_reader.open_remote_node`), so an explicit *version*
    needs no format pinning here.
    """
    node = open_remote_node(store)
    if node is None:
        raise _group_not_found_error(store)
    if hasattr(node, "shape"):
        raise _array_at_root_error(store)
    return node


def _open_root_node(store, version: str | None):
    """Open the root node of *store* for reading via the best backend.

    Local directory paths go through the zarrista compat layer (direct
    metadata-document reads), remote URL handles through zarrista's async
    API over obstore, and bytes mappings through the pure-Python store
    reader. Any other store object raises ``TypeError``.
    """
    if isinstance(store, RemoteZarrStore):
        return _open_remote_root(store, version)
    if _is_bytes_mapping(store):
        from ._v2_store_reader import V2Group, V3Group, open_store_node

        try:
            node = open_store_node(store)
        except KeyError as e:
            raise _group_not_found_error(store) from e
        if not isinstance(node, (V2Group, V3Group)):
            raise _array_at_root_error(store)
        return node
    if _is_local_path(store):
        return _open_local_root(store, version)
    raise TypeError(
        "ngff-zarr reads from local directory paths, remote URL strings, "
        f"zip archive paths, and key-to-bytes mappings; got "
        f"{type(store).__name__}. Pass a path instead of a zarr-python "
        "store object."
    )


def _distributed_singlescale_transforms(ome_value: dict, root) -> dict:
    """Transformations of singlescale children that carry none inline.

    The RFC-8 distributed layout stores each level's singlescale document in
    the level's own ``zarr.json``; this collects their
    ``coordinateTransformations``, keyed by dataset path, reading only the
    children the inlined document leaves empty.
    """
    collected: dict[str, list] = {}
    for node in ome_value.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "singlescale":
            continue
        attributes = node.get("attributes") or {}
        if attributes.get("coordinateTransformations"):
            continue
        raw_path = (node.get("path") or {}).get("path")
        if not isinstance(raw_path, str):
            continue
        path = raw_path[2:] if raw_path.startswith("./") else raw_path
        try:
            child = root[path]
        except Exception:
            continue
        child_ome = child.attrs.asdict().get("ome")
        if not isinstance(child_ome, dict):
            continue
        transforms = (child_ome.get("attributes") or {}).get(
            "coordinateTransformations"
        )
        if transforms:
            collected[path] = transforms
    return collected


def from_ome_zarr(
    store: StoreLike,
    validate: bool = False,
    version: str | None = None,
    storage_options: dict | None = None,
) -> NgffMultiscales:
    """
    Read an OME-Zarr NGFF multiscales data structure (NgffMultiscales) from a Zarr store.

    store : StoreLike
        Store or path to directory in file system. Can be a string URL
        (e.g., 's3://bucket/path') for remote storage. For .ozx files,
        provide the path to the .ozx file.

        For HCS (High Content Screening) plates, provide the path to a
        specific well and field within the plate (e.g., 'plate.zarr/A/1/0'
        for well A1, field 0). To load the entire plate structure, use
        from_hcs_zarr() instead.

        For bioformats2raw containers with a single image, the function
        will automatically navigate into the image subgroup. For containers
        with multiple images, provide the path to a specific image
        (e.g., 'container.ome.zarr/0').

    validate : bool
        If True, validate the NGFF metadata against the schema.

    version : string, optional
        OME-Zarr version, if known. For .ozx files, the version will be
        read from the ZIP comment if not provided.

    storage_options : dict, optional
        Storage options to pass to the store if store is a string URL.
        For S3 URLs, this can include authentication credentials and other
        options for the underlying filesystem. fsspec-style option names
        are accepted; when the zarrista/obstore remote engine handles the
        read they are translated to the obstore equivalents, warning on
        (and ignoring) options with no translation.

    Returns
    -------

    multiscales: multiscale ngff image with dask-chunked arrays for data

    """
    from .parse_metadata import (
        _detect_version,
        _extract_method_metadata,
        _get_bioformats2raw_series,
        _is_bioformats2raw,
        _is_hcs_plate,
        _parse_hcs_path,
    )

    # Parse potential HCS sub-paths (e.g., 'plate.zarr/A/1/0')
    original_store = store
    subpath = None
    if isinstance(store, (str, Path)):
        store_str = str(store)
        # Only parse for local file paths (not URLs)
        if not store_str.startswith(REMOTE_URL_SCHEMES):
            store, subpath = _parse_hcs_path(store_str)

    # RFC-9: Handle .ozx (zipped OME-Zarr) files
    if isinstance(store, (str, Path)) and is_ozx_path(store):
        # Read version from .ozx comment if not provided
        if version is None:
            version = read_ozx_version(store)
            if version is None:
                version = "0.5"  # Default to 0.5 for .ozx files

        # Read the archive through the zarrista-backed zip store handle
        # (zarr-python ZipStore only when zarrista is unavailable).
        store = open_ozx_store(store)

    # Remote URL strings read through zarrista's async API over obstore
    # (storage_options translated to obstore configuration).
    if isinstance(store, str) and store.startswith(REMOTE_URL_SCHEMES):
        if not remote_read_available():
            raise _remote_backend_import_error(store, None)
        store = RemoteZarrStore(store, storage_options=storage_options)

    # Open the root node with helpful error messages, dispatching on the
    # store type: local paths -> compat layer, bytes mappings -> store
    # reader, remote handles -> zarrista/obstore.
    root = _open_root_node(store, version)

    # Check root-level attributes first to see if this is an HCS plate
    root_attrs_initial = root.attrs.asdict()
    is_hcs_plate_root = _is_hcs_plate(root_attrs_initial)

    # If this is an HCS plate and no subpath was provided, error with guidance
    if is_hcs_plate_root and not subpath:
        # Try to provide helpful error message with available wells
        try:
            from .hcs import from_hcs_zarr

            # Attempt to load plate metadata to provide well information
            plate = from_hcs_zarr(original_store, validate=False)

            # Build helpful error message with well examples
            well_examples = []
            if plate.metadata.wells:
                # Normalize store path to forward slashes for display (Windows-friendly)
                display_store = str(original_store).replace("\\", "/")
                # Show up to 3 well examples
                for well in plate.metadata.wells[:3]:
                    well_path = well.path
                    # Add field 0 as example
                    well_examples.append(f"'{display_store}/{well_path}/0'")

            examples_str = (
                ", ".join(well_examples) if well_examples else "'plate.zarr/A/1/0'"
            )

            raise ValueError(
                f"The input appears to be an HCS (High Content Screening) plate structure "
                f"with {len(plate.metadata.wells)} wells. "
                f"To convert a specific well/image, provide the full path including well and field:\n"
                f"  Examples: {examples_str}\n"
                f"For programmatic access to the full plate, use from_hcs_zarr() instead of from_ome_zarr()."
            )
        except ValueError:
            # Re-raise ValueError (from our error message above)
            raise
        except Exception:
            # Fallback to generic error if we can't load plate metadata
            raise ValueError(
                "The input appears to be an HCS (High Content Screening) plate structure, "
                "which contains multiple wells and images. To convert a specific well/image, "
                "provide the full path including well and field (e.g., 'plate.zarr/A/1/0' for well A1, field 0). "
                "For programmatic access to the full plate, use from_hcs_zarr() instead of from_ome_zarr()."
            ) from None

    # Check if this is a bioformats2raw container layout
    if _is_bioformats2raw(root_attrs_initial) and not subpath:
        image_paths = _get_bioformats2raw_series(root)

        if not image_paths:
            raise ValueError(
                "The input is a bioformats2raw container but no image subgroups were found. "
                "The container may be empty or malformed."
            )
        if len(image_paths) == 1:
            # Single image: navigate into it silently
            subpath = image_paths[0]
        else:
            # Multiple images: require user to specify which one
            display_store = str(original_store).replace("\\", "/")
            examples = [f"'{display_store}/{p}'" for p in image_paths[:5]]
            examples_str = ", ".join(examples)
            more = f" (and {len(image_paths) - 5} more)" if len(image_paths) > 5 else ""
            raise ValueError(
                f"The input is a bioformats2raw container with {len(image_paths)} images. "
                f"To load a specific image, provide the full path including the image index:\n"
                f"  Available images: {examples_str}{more}\n"
                f"For example: ngff-zarr -i {display_store}/{image_paths[0]}"
            )

    # Navigate to sub-path if provided (for HCS well/image access,
    # or bioformats2raw image navigation)
    if subpath:
        try:
            root = root[subpath]
        except KeyError:
            raise ValueError(
                f"Sub-path '{subpath}' not found in store. "
                f"Ensure the path is correct (e.g., 'A/1/0' for well A1, field 0)."
            )
    root_attrs = root.attrs.asdict()

    if not version:
        version = _detect_version(root_attrs).value

    # Check if this is an HCS plate structure
    if _is_hcs_plate(root_attrs):
        raise ValueError(
            "The input appears to be an HCS (High Content Screening) plate structure, "
            "which contains multiple wells and images. Use from_hcs_zarr() instead of from_ngff_zarr() "
            "to load plate data. For CLI usage, you may need to specify a specific well/image path "
            "within the plate (e.g., 'plate.zarr/A/1/0' for well A1, field 0)."
        )

    if version == "0.9.dev3":
        # RFC-8: the root ``ome`` value is a typed node document, not a
        # multiscales wrapper. A multiscale node converts to the 0.9.dev1
        # entry shape and reads through the 0.9.dev1 machinery; any other
        # node type is a collection-model document (and without this branch
        # a 0.9.dev3 store would fall through to the v0.4 branch and fail on
        # a missing ``multiscales`` key).
        ome_value = root_attrs.get("ome")
        node_type = ome_value.get("type") if isinstance(ome_value, dict) else None
        if node_type != "multiscale":
            raise ValueError(
                f"The input is an OME-Zarr 0.9.dev3 store whose root is a "
                f"{node_type!r} node. 0.9.dev3 stores the RFC-8 node model in "
                "place of the multiscales metadata; use from_collection_zarr() "
                "to read it. from_ome_zarr() reads multiscale image stores."
            )
        from .rfc8.multiscale import multiscales_entry_from_ome
        from .v09.zarr_metadata import Metadata

        if validate:
            from .rfc8.io import _validate_document

            declared = ome_value.get("version")
            declared = declared if isinstance(declared, str) else None
            _validate_document(ome_value, declared)

        singlescale_transforms = _distributed_singlescale_transforms(ome_value, root)
        entry, omero = multiscales_entry_from_ome(ome_value, singlescale_transforms)
        converted_attrs = {"ome": {"version": "0.9.dev1", "multiscales": [entry]}}
        if omero is not None:
            converted_attrs["ome"]["omero"] = omero
        metadata_obj, images = Metadata._from_zarr_attrs(
            converted_attrs, store, validate=False, subpath=subpath
        )
        if validate:
            from .structural_validation import validate_structural

            validate_structural(metadata_obj, version="0.9.dev3")
        method, method_type, method_metadata = _extract_method_metadata(entry)

    elif version == "0.9.dev1":
        from .v09.zarr_metadata import Metadata

        # No 0.9.dev1 JSON Schema is published; `validate` is forwarded so
        # validate=True reports that rather than validating nothing.
        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=validate, subpath=subpath
        )
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs["ome"]["multiscales"][0]
        )

    elif version.startswith("0.6"):
        from .v06.zarr_metadata import Metadata

        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=validate, subpath=subpath
        )
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs["ome"]["multiscales"][0]
        )

    elif version == "0.5":
        from .v05.zarr_metadata import Metadata

        # Validate v0.5 structure before accessing
        if "ome" not in root_attrs:
            raise ValueError(
                f"Expected OME-Zarr v{version} format with 'ome' key in root attributes, "
                f"but 'ome' key is missing. The store may not be a valid OME-Zarr or may be v0.4 format. "
                f"Available keys: {list(root_attrs.keys())}"
            )
        if "multiscales" not in root_attrs["ome"]:
            available_keys = list(root_attrs["ome"].keys())
            error_msg = (
                f"Expected OME-Zarr v{version} format with 'multiscales' under 'ome' key, "
                f"but 'multiscales' key is missing. Available keys under 'ome': {available_keys}\n\n"
                "Possible causes:\n"
                "  1. The file may be corrupted or incomplete\n"
                "  2. The file may not be a valid OME-Zarr image (it could be an HCS plate root that "
                "requires navigating to a specific well/field)\n"
                "  3. The file may use a custom or unsupported metadata structure\n\n"
            )

            # Provide additional generic guidance when 'multiscales' is missing under 'ome'
            error_msg += (
                "The 'ome' key is present but doesn't contain the required 'multiscales' metadata. "
                "Please verify that this is a valid OME-Zarr file."
            )

            raise ValueError(error_msg)

        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=validate, subpath=subpath
        )
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs["ome"]["multiscales"][0]
        )

    else:
        from .v04.zarr_metadata import Metadata

        # v0.4 metadata validation is handled by Metadata._from_zarr_attrs
        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=validate, subpath=subpath
        )
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs["multiscales"][0]
        )

    # Normalize to the richest model the store can express. A 0.9 development
    # series store stays on the 0.9 model; downgrading it to 0.6 would discard
    # its RFC-3 axis model.
    metadata_obj = metadata_obj.to_version(
        version if version in ("0.9.dev1", "0.9.dev3") else "0.6"
    )
    metadata_obj.type = method_type
    metadata_obj.metadata = method_metadata

    root_attributes = {
        key: value for key, value in root_attrs.items() if key not in OME_ROOT_KEYS
    }
    return NgffMultiscales(
        images, metadata_obj, method=method, root_attributes=root_attributes or None
    )


#: Backwards-compatible alias for :func:`from_ome_zarr`.
from_ngff_zarr = from_ome_zarr
