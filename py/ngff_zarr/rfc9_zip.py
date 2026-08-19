# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
RFC-9: Zipped OME-Zarr (.ozx) support

This module provides utilities for reading and writing OME-Zarr hierarchies
in ZIP archives according to RFC-9 specification.
"""

import json
import posixpath
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path


class ZipReadStore(Mapping):
    """Read-only name-to-bytes view of an OME-Zarr zip archive (.ozx/.zip).

    The ``Mapping[str, bytes]`` surface feeds the pure-Python store reader
    for metadata parsing and group navigation, while the retained archive
    *path* lets the compat layer route pixel reads through zarrista's native
    zip store — required for the sharded arrays ``.ozx`` archives hold by
    default, which the mapping reader cannot decode. A *prefix* narrows the
    view to a sub-hierarchy (HCS well/field access): keys are resolved
    relative to it and iteration yields only the keys below it.
    """

    def __init__(self, path: str | Path, prefix: str = ""):
        self.path = Path(path)
        self.prefix = "/".join(
            part for part in str(prefix).split("/") if part not in ("", ".")
        )
        self._zipfile = zipfile.ZipFile(self.path, mode="r")
        self._names = frozenset(
            name for name in self._zipfile.namelist() if not name.endswith("/")
        )

    def with_prefix(self, prefix: str) -> "ZipReadStore":
        """A view of the same archive narrowed to *prefix* (joined to any
        existing prefix)."""
        return ZipReadStore(self.path, posixpath.join(self.prefix, str(prefix)))

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def __getitem__(self, key: str) -> bytes:
        name = self._full_key(key)
        if name not in self._names:
            raise KeyError(key)
        return self._zipfile.read(name)

    def __iter__(self) -> Iterator[str]:
        if not self.prefix:
            yield from self._names
            return
        start = f"{self.prefix}/"
        for name in self._names:
            if name.startswith(start):
                yield name[len(start) :]

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __str__(self) -> str:
        # Error messages built from the store display the archive path.
        return str(self.path if not self.prefix else self.path / self.prefix)


def is_ozx_path(path: str | Path) -> bool:
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


def _get_store_root_path(source_store) -> Path | None:
    """Return the filesystem root Path for a store, or None for non-filesystem stores.

    Delegates to the compat layer's resolver, which handles plain
    ``str``/``Path``/``os.PathLike`` targets and zarrista ``FilesystemStore``
    handles.
    """
    # Imported lazily: _zarrista_utils imports is_ozx_path from this module.
    from ._zarrista_utils import resolve_store_path

    return resolve_store_path(source_store)


def _enumerate_fs_files(root_dir: Path) -> list[str]:
    """Enumerate all files under root_dir as forward-slash-separated relative paths."""
    all_files = []
    for file_path in root_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(root_dir)
            # Convert to forward slashes for ZIP
            all_files.append(str(rel_path).replace("\\", "/"))
    return all_files


def _order_files_rfc9(all_files: list[str]) -> list[str]:
    """Order files per RFC-9: root zarr.json first, other zarr.json next, then rest sorted."""
    # Collect and deterministically order all zarr.json files
    has_root = "zarr.json" in all_files
    non_root_zarr_jsons = [
        f for f in all_files if f.endswith("zarr.json") and f != "zarr.json"
    ]
    # Sort non-root zarr.json by path depth (shallower first) then lexicographically
    non_root_zarr_jsons_sorted = sorted(
        non_root_zarr_jsons, key=lambda p: (p.count("/"), p)
    )
    if has_root:
        zarr_jsons = ["zarr.json"] + non_root_zarr_jsons_sorted
    else:
        zarr_jsons = non_root_zarr_jsons_sorted

    other_files = [f for f in all_files if not f.endswith("zarr.json")]
    return zarr_jsons + sorted(other_files)


def _write_rfc9_comment(zf: zipfile.ZipFile, version: str) -> None:
    """Write the RFC-9 ZIP comment with version and jsonFirst flag."""
    comment_dict = {
        "ome": {
            "version": version,
            "zipFile": {"centralDirectory": {"jsonFirst": True}},
        }
    }
    zf.comment = json.dumps(comment_dict).encode("utf-8")


def write_store_to_zip(
    source_store,
    zip_path: str | Path,
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

    Files are streamed one at a time from filesystem-backed stores so that
    peak memory stays proportional to the largest single file (typically one
    chunk or shard) rather than the entire dataset.

    This function can be used to convert any existing zarr store (including HCS plates)
    to .ozx format after it has been written.

    Parameters
    ----------
    source_store : str, Path, os.PathLike, Mapping, or zarrista FilesystemStore
        Source zarr store to write from. Can be a path (str, Path, or
        os.PathLike) to a directory containing zarr data, a zarrista
        FilesystemStore handle, or an in-memory key-to-bytes mapping.
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
    zip_path = Path(zip_path)

    # Determine if the source is filesystem-backed
    root_dir = _get_store_root_path(source_store)

    if root_dir is not None:
        # Filesystem-backed store: enumerate files from disk
        all_files = _enumerate_fs_files(root_dir)
    elif isinstance(source_store, Mapping):
        # In-memory key-to-bytes mapping: keys are the file paths.
        all_files = list(source_store)
    else:
        raise TypeError(
            "write_store_to_zip requires a local directory path, a zarrista "
            f"FilesystemStore, or a key-to-bytes mapping; got "
            f"{type(source_store).__name__}."
        )

    if not all_files:
        raise ValueError(f"No files found in source store of type {type(source_store)}")

    ordered_files = _order_files_rfc9(all_files)

    # Prepare the destination. On Windows, opening a directory path in write
    # mode raises PermissionError (Errno 13) rather than IsADirectoryError, so
    # an existing directory at the destination must be removed explicitly. This
    # commonly happens when a plate was first written as an unzipped store to
    # the same path before being converted to .ozx (see GH #241). A pre-existing
    # regular file is fine -- zipfile mode="w" overwrites it.
    if zip_path.exists() and zip_path.is_dir():
        # Refuse to delete the source store itself.
        if root_dir is not None:
            try:
                same_path = zip_path.resolve() == root_dir.resolve()
            except OSError:
                same_path = False
            if same_path:
                raise ValueError(
                    f"Destination path '{zip_path}' is the same as the source "
                    "store. Choose a different destination for the .ozx archive."
                )
        import shutil

        shutil.rmtree(zip_path)

    # Create ZIP archive with ZIP64 support
    with zipfile.ZipFile(
        zip_path, mode="w", compression=compression, allowZip64=True
    ) as zf:
        if root_dir is not None:
            # Filesystem-backed: stream one file at a time to keep memory
            # proportional to the largest single file, not the whole dataset.
            for file_path in ordered_files:
                full_path = root_dir / file_path
                if not full_path.exists():
                    raise ValueError(
                        f"Could not read data for {file_path} from {root_dir}"
                    )
                zf.write(full_path, arcname=file_path)
        else:
            # In-memory mapping: data is already in memory.
            for file_path in ordered_files:
                data = source_store.get(file_path)
                if data is None:
                    raise ValueError(
                        f"Could not read data for {file_path} from source store"
                    )
                zf.writestr(file_path, bytes(data))

        _write_rfc9_comment(zf, version)


def read_ozx_version(zip_path: str | Path) -> str | None:
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


def read_ozx_json_first(zip_path: str | Path) -> bool:
    """
    Read the jsonFirst flag from a .ozx file's ZIP comment.

    The jsonFirst flag indicates that zarr.json files are ordered
    breadth-first in the central directory and precede other content.

    Parameters
    ----------
    zip_path : str or Path
        Path to the .ozx file

    Returns
    -------
    bool
        True if jsonFirst is set to true in the ZIP comment, False otherwise
    """
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            if zf.comment:
                try:
                    comment_str = zf.comment.decode("utf-8")
                    comment_dict = json.loads(comment_str)
                    if "ome" in comment_dict:
                        ome = comment_dict["ome"]
                        if "zipFile" in ome:
                            zip_file = ome["zipFile"]
                            if "centralDirectory" in zip_file:
                                central_dir = zip_file["centralDirectory"]
                                return central_dir.get("jsonFirst", False)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # ZIP comment is not valid JSON or not UTF-8 encoded
                    pass
    except (zipfile.BadZipFile, FileNotFoundError):
        # File is not a valid ZIP or doesn't exist
        pass
    return False
