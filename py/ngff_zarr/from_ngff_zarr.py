# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import Optional
import packaging.version

import zarr
import zarr.storage

from .to_multiscales import Multiscales
from ._zarr_types import StoreLike

from .rfc9_zip import is_ozx_path, read_ozx_version

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

# Supported remote URL schemes for storage
REMOTE_URL_SCHEMES = ("s3://", "gs://", "azure://", "http://", "https://")


def from_ngff_zarr(
    store: StoreLike,
    validate: bool = False,
    version: Optional[str] = None,
    storage_options: Optional[dict] = None,
) -> Multiscales:
    """
    Read an OME-Zarr NGFF Multiscales data structure from a Zarr store.

    store : StoreLike
        Store or path to directory in file system. Can be a string URL
        (e.g., 's3://bucket/path') for remote storage. For .ozx files,
        provide the path to the .ozx file.

        For HCS (High Content Screening) plates, provide the path to a
        specific well and field within the plate (e.g., 'plate.zarr/A/1/0'
        for well A1, field 0). To load the entire plate structure, use
        from_hcs_zarr() instead.

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
        _extract_method_metadata,
        _detect_version,
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
                store = zarr.storage.FsspecStore.from_url(
                    store, storage_options=storage_options
                )
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
    root = zarr.open_group(store, mode="r", **format_kwargs)

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
                f"For programmatic access to the full plate, use from_hcs_zarr() instead of from_ngff_zarr()."
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
                "For programmatic access to the full plate, use from_hcs_zarr() instead of from_ngff_zarr()."
            ) from None

    # Navigate to sub-path if provided (for HCS well/image access)
    # We get metadata from the subpath but pass the subpath to _from_zarr_attrs
    # so it can correctly construct data paths
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

    if version == "0.5":
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

    metadata_obj.type = method_type
    metadata_obj.metadata = method_metadata

    return Multiscales(images, metadata_obj, method=method)
