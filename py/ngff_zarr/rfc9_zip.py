# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
RFC-9: Zipped OME-Zarr (.ozx) support

This module provides utilities for reading and writing OME-Zarr hierarchies
in ZIP archives according to RFC-9 specification.
"""

import json
import zipfile
from pathlib import Path
from typing import Union, Optional

import zarr
import zarr.storage


def is_ozx_path(path: Union[str, Path]) -> bool:
    """
    Check if a path refers to a .ozx file.

    Parameters
    ----------
    path : str or Path
        Path to check

    Returns
    -------
    bool
        True if path ends with .ozx extension
    """
    return str(path).endswith(".ozx")


def write_store_to_zip(
    source_store: Union[zarr.storage.StoreLike, str, Path],
    zip_path: Union[str, Path],
    version: str = "0.5",
    compression: int = zipfile.ZIP_STORED,
) -> None:
    """
    Write a zarr store to a ZIP archive following RFC-9 specification.

    According to RFC-9:
    - Root-level zarr.json must be the first entry
    - Other zarr.json files should follow in breadth-first order
    - ZIP-level compression should be disabled (ZIP_STORED)
    - ZIP64 format should be used
    - A comment with OME-Zarr version should be added

    This function can be used to convert any existing zarr store (including HCS plates)
    to .ozx format after it has been written.

    Parameters
    ----------
    source_store : zarr.storage.StoreLike, str, or Path
        Source zarr store to write from. Can be a store object (LocalStore, DirectoryStore)
        or a path string to a directory containing zarr data.
    zip_path : str or Path
        Path to output .ozx file
    version : str, optional
        OME-Zarr version string (e.g., "0.5")
    compression : int, optional
        ZIP compression method (default: ZIP_STORED for no compression)

    Examples
    --------
    Convert an existing HCS plate to .ozx:

    >>> from ngff_zarr.rfc9_zip import write_store_to_zip
    >>> # After writing plate with to_hcs_zarr or HCSPlateWriter
    >>> write_store_to_zip("my_plate.ome.zarr", "my_plate.ozx", version="0.5")
    """
    import asyncio
    from zarr.core.buffer import default_buffer_prototype

    zip_path = Path(zip_path)

    # Get the buffer prototype for zarr v3 stores
    proto = default_buffer_prototype()

    # Handle string/Path inputs by converting to appropriate store or using filesystem directly
    if isinstance(source_store, (str, Path)):
        # For path strings, enumerate files directly from filesystem
        root_dir = Path(source_store)
        all_files = []
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                # Get relative path from root
                rel_path = file_path.relative_to(root_dir)
                # Convert to forward slashes for ZIP
                all_files.append(str(rel_path).replace("\\", "/"))
    elif hasattr(zarr.storage, "LocalStore") and isinstance(
        source_store, zarr.storage.LocalStore
    ):
        # Get the root directory from LocalStore
        root_dir = Path(source_store.root)
        all_files = []
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                # Get relative path from root
                rel_path = file_path.relative_to(root_dir)
                # Convert to forward slashes for ZIP
                all_files.append(str(rel_path).replace("\\", "/"))
    elif hasattr(zarr.storage, "DirectoryStore") and isinstance(
        source_store, zarr.storage.DirectoryStore
    ):
        # For DirectoryStore (zarr v2), use dir_path()
        root_dir = Path(source_store.dir_path())
        all_files = []
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(root_dir)
                all_files.append(str(rel_path).replace("\\", "/"))
    else:
        # For other store types, use async list()
        async def get_all_files():
            items = []
            async for item in source_store.list():
                items.append(item)
            return items

        all_files = asyncio.run(get_all_files())

    if not all_files:
        raise ValueError(f"No files found in source store of type {type(source_store)}")

    # Get zarr.json files - root first, then in breadth-first order
    zarr_jsons = [f for f in all_files if f.endswith("zarr.json")]
    if "zarr.json" in zarr_jsons:
        zarr_jsons.remove("zarr.json")
        zarr_jsons.insert(0, "zarr.json")

    # Other files
    other_files = [f for f in all_files if not f.endswith("zarr.json")]

    # Order: zarr.json files first, then everything else sorted
    ordered_files = zarr_jsons + sorted(other_files)

    # Read file data based on store type
    if isinstance(source_store, (str, Path)):
        # Read files directly from filesystem for path strings
        root_dir = Path(source_store)
        file_data = {}
        for file_path in ordered_files:
            full_path = root_dir / file_path
            if full_path.exists():
                file_data[file_path] = full_path.read_bytes()
            else:
                raise ValueError(f"Could not read data for {file_path} from {root_dir}")
    elif hasattr(zarr.storage, "LocalStore") and isinstance(
        source_store, zarr.storage.LocalStore
    ):
        # Read files directly from filesystem
        root_dir = Path(source_store.root)
        file_data = {}
        for file_path in ordered_files:
            full_path = root_dir / file_path
            if full_path.exists():
                file_data[file_path] = full_path.read_bytes()
            else:
                raise ValueError(f"Could not read data for {file_path} from {root_dir}")
    elif hasattr(zarr.storage, "DirectoryStore") and isinstance(
        source_store, zarr.storage.DirectoryStore
    ):
        # Read files directly from filesystem
        root_dir = Path(source_store.dir_path())
        file_data = {}
        for file_path in ordered_files:
            full_path = root_dir / file_path
            if full_path.exists():
                file_data[file_path] = full_path.read_bytes()
            else:
                raise ValueError(f"Could not read data for {file_path} from {root_dir}")
    else:
        # For other store types, use async get()
        # Helper async function to get file data
        async def get_file_data(file_path: str):
            """Get data from store using zarr v3 async API"""
            try:
                result = await source_store.get(file_path, proto)
                if result:
                    return result.to_bytes()
                return None
            except (KeyError, FileNotFoundError):
                # File not found in store - this is an error condition
                return None

        # Gather all file data in a single event loop (more efficient than creating one per file)
        async def get_all_file_data(file_paths):
            results = await asyncio.gather(*(get_file_data(fp) for fp in file_paths))
            return dict(zip(file_paths, results))

        file_data = asyncio.run(get_all_file_data(ordered_files))

    # Create ZIP archive with ZIP64 support
    with zipfile.ZipFile(
        zip_path, mode="w", compression=compression, allowZip64=True
    ) as zf:
        # Write files in order
        for file_path in ordered_files:
            data = file_data[file_path]
            if data is None:
                raise ValueError(
                    f"Could not read data for {file_path} from source store"
                )

            # Write to ZIP
            zf.writestr(file_path, data)

        # Add OME-Zarr version comment as per RFC-9
        comment_dict = {"ome": {"version": version}}
        comment_json = json.dumps(comment_dict)
        zf.comment = comment_json.encode("utf-8")


def read_ozx_version(zip_path: Union[str, Path]) -> Optional[str]:
    """
    Read the OME-Zarr version from a .ozx file's ZIP comment.

    Parameters
    ----------
    zip_path : str or Path
        Path to the .ozx file

    Returns
    -------
    str or None
        OME-Zarr version string if found, None otherwise
    """
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            if zf.comment:
                try:
                    comment_str = zf.comment.decode("utf-8")
                    comment_dict = json.loads(comment_str)
                    if "ome" in comment_dict and "version" in comment_dict["ome"]:
                        return comment_dict["ome"]["version"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # ZIP comment is not valid JSON or not UTF-8 encoded
                    pass
    except (zipfile.BadZipFile, FileNotFoundError):
        # File is not a valid ZIP or doesn't exist
        pass
    return None
