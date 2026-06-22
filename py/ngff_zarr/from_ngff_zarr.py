# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path

import packaging.version
import zarr
import zarr.errors
import zarr.storage

from ._zarr_types import StoreLike
from .multiscales import NgffMultiscales
from .rfc9_zip import is_ozx_path, read_ozx_version

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

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


def _remote_backend_import_error(store, original: ImportError) -> ImportError:
    """Build an actionable ImportError pointing at the ``[remote]`` extra."""
    return ImportError(
        f"Reading from the remote store '{store}' requires additional "
        "packages that are not installed. Install them with:\n\n"
        '    pip install "ngff-zarr[remote]"\n\n'
        "This provides the fsspec backends for http(s), S3, GCS, and Azure. "
        f"Original error: {original}"
    )


# -- blosc codec backward-compatibility -----------------------------------

# Deprecated Blosc codec configuration keys that should be stripped on read.
_DEPRECATED_BLOSC_KEYS: frozenset = frozenset({"nthreads", "blocksize"})


def _apply_blosc_codec_compat() -> None:
    """Monkey-patch BloscCodec.from_dict to strip deprecated nthreads/blocksize keys.

    Older zarr v3 implementations wrote ``nthreads`` and ``blocksize`` into the
    Blosc codec metadata.  These keys are no longer recognised and cause
    ``BloscCodec.from_dict`` to raise.  This patch filters them out (both at
    the top level of the codec dict and inside a nested ``"configuration"``
    mapping) before passing control to the original method.
    """
    try:
        from zarr.codecs.blosc import BloscCodec

        _orig = BloscCodec.from_dict

        @classmethod
        def _patched(cls, data):
            if isinstance(data, dict):
                if any(k in _DEPRECATED_BLOSC_KEYS for k in data):
                    data = {
                        k: v for k, v in data.items() if k not in _DEPRECATED_BLOSC_KEYS
                    }
                if "configuration" in data and isinstance(data["configuration"], dict):
                    cfg = data["configuration"]
                    if any(k in _DEPRECATED_BLOSC_KEYS for k in cfg):
                        data = {
                            **data,
                            "configuration": {
                                k: v
                                for k, v in cfg.items()
                                if k not in _DEPRECATED_BLOSC_KEYS
                            },
                        }
            return _orig.__func__(cls, data)

        BloscCodec.from_dict = _patched
    except ImportError:
        pass


_apply_blosc_codec_compat()


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
        options for the underlying filesystem.

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

        # For zarr v3, create ZipStore directly with the path
        store = zarr.storage.ZipStore(str(store), mode="r")

    # Handle string URLs with storage options (zarr-python 3+ only)
    if isinstance(store, str) and storage_options is not None:
        if store.startswith(REMOTE_URL_SCHEMES):
            if zarr_version_major >= 3 and hasattr(zarr.storage, "FsspecStore"):
                try:
                    store = zarr.storage.FsspecStore.from_url(
                        store, storage_options=storage_options
                    )
                except ImportError as e:
                    # Building the FsspecStore imports the fsspec backend
                    # eagerly, so a missing backend surfaces here rather than
                    # in _open_group_with_helpful_errors below.
                    raise _remote_backend_import_error(store, e) from e
            else:
                raise RuntimeError(
                    "storage_options parameter requires zarr-python 3+ with FsspecStore support. "
                    f"Current zarr version: {zarr.__version__}"
                )

    format_kwargs = {}
    if version and zarr_version_major >= 3:
        format_kwargs = (
            {"zarr_format": 2}
            if packaging.version.parse(version) < packaging.version.parse("0.5")
            else {"zarr_format": 3}
        )

    def _open_group_with_helpful_errors(**open_kwargs):
        try:
            return zarr.open_group(store, mode="r", **open_kwargs)
        except zarr.errors.GroupNotFoundError as e:
            store_path = str(store)
            raise ValueError(
                f"No valid Zarr group found at '{store_path}'. "
                "This error typically occurs when:\n"
                "  1. The path does not contain a valid Zarr store\n"
                "  2. The Zarr store is empty or corrupted\n"
                "  3. The download was incomplete\n"
                "  4. The store contains a Zarr array at the root instead of a group\n\n"
                "For OME-Zarr files, the root must be a Zarr group "
                "containing multiscale metadata. "
                "Please verify that the input path points to a valid OME-Zarr store."
            ) from e
        except zarr.errors.ContainsArrayError as e:
            store_path = str(store)
            raise ValueError(
                f"The Zarr store at '{store_path}' contains an array at the root level, "
                "but OME-Zarr requires a group structure with multiscale metadata. "
                "Single Zarr arrays cannot be directly converted to OME-Zarr format."
            ) from e
        except ImportError as e:
            # Reading a remote store (e.g. https://, s3://) goes through an
            # fsspec filesystem backend such as aiohttp, requests, s3fs, gcsfs,
            # or adlfs. When the relevant backend is missing, fsspec raises an
            # ImportError. Rewrite it into an actionable message; re-raise any
            # unrelated import failure unchanged. _is_remote_store also matches
            # an FsspecStore, whose str() is not the original URL.
            if _is_remote_store(store):
                raise _remote_backend_import_error(store, e) from e
            raise

    # Open the Zarr store as a group with helpful error messages
    root = _open_group_with_helpful_errors(**format_kwargs)
    # When auto-detecting version (no explicit version provided) on zarr-python 3,
    # zarr may prefer a spurious zarr.json (format 3) over .zgroup (format 2),
    # which can yield an empty-root group. Retry the v2 open through the same
    # helper so fallback errors get the same helpful wrapping.
    if version is None and zarr_version_major >= 3 and not root.attrs:
        root = _open_group_with_helpful_errors(zarr_format=2)
    # resulting in empty root attributes. Fall back to zarr_format=2 in that case.
    if not version and zarr_version_major >= 3:
        root_attrs_check = root.attrs.asdict()
        if not root_attrs_check:
            root = zarr.open_group(store, mode="r", zarr_format=2)

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

    if version == "0.6":
        from .v06.zarr_metadata import Metadata
        # TODO: Restore validation for v0.6
        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=False)
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs['ome']['multiscales'][0])

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

    metadata_obj = metadata_obj.to_version("0.6")
    metadata_obj.type = method_type
    metadata_obj.metadata = method_metadata

    return NgffMultiscales(images, metadata_obj, method=method)


#: Backwards-compatible alias for :func:`from_ome_zarr`.
from_ngff_zarr = from_ome_zarr
