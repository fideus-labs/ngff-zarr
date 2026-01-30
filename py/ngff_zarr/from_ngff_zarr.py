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
    from .parse_metadata import _extract_method_metadata, _detect_version

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
        if store.startswith(("s3://", "gs://", "azure://", "http://", "https://")):
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
    root_attrs = root.attrs.asdict()

    if not version:
        version = _detect_version(root_attrs).value

    if version == "0.5":
        from .v05.zarr_metadata import Metadata

        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=validate
        )
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs["ome"]["multiscales"][0]
        )

    else:
        from .v04.zarr_metadata import Metadata

        metadata_obj, images = Metadata._from_zarr_attrs(
            root_attrs, store, validate=validate
        )
        method, method_type, method_metadata = _extract_method_metadata(
            root_attrs["multiscales"][0]
        )

    metadata_obj.type = method_type
    metadata_obj.metadata = method_metadata

    return Multiscales(images, metadata_obj, method=method)
