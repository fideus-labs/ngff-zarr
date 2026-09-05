# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Dereferencing of RFC-8 typed paths to the documents they locate.

A resolved document is the raw mapping the path's type designates: a Zarr
group's attribute dict for ``"zarr"`` (the group's ``zarr.json`` attributes,
with a Zarr format 2 fallback), or a standalone JSON file's root object for
``"json"``. Interpreting what the document holds -- an RFC-8 node, a
multiscales image, something else -- is the caller's concern
(:meth:`ngff_zarr.rfc8.NgffCollection.load`).

Per the RFC's security considerations, nothing here restricts where a path
may point: a relative path can climb out of the collection's directory and a
URL can name any host. Callers resolving untrusted collection documents
should check :func:`resolve_location` results before fetching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import url2pathname, urlopen

from .._remote_reader import remote_read_available
from .model import PATH_TYPE_JSON, PATH_TYPE_ZARR, OmePath

_HTTP_SCHEMES = ("http://", "https://")


def _is_http(location: str) -> bool:
    return location.startswith(_HTTP_SCHEMES)


def resolve_location(path: OmePath, *, base: str | None = None) -> str:
    """Resolve a path's location string against the document base.

    ``base`` is the directory-like location of the document that carries the
    path: the Zarr group directory or URL for a document stored in group
    attributes, the parent directory or URL of a standalone JSON file. A
    relative path (IETF RFC 1808) is resolved against it; ``file://`` paths
    (IETF RFC 8089) and ``http(s)://`` URLs stand alone.
    """
    location = path.path
    if _is_http(location):
        return location
    if location.startswith("file://"):
        parsed = urlparse(location)
        # RFC 8089 permits a Windows drive in the authority position
        # (``file://C:/...``); rejoin it with the path before decoding.
        return url2pathname(f"{parsed.netloc}{parsed.path}")
    if base is None:
        if not Path(location).is_absolute():
            raise ValueError(
                f"The relative path {location!r} cannot be resolved: the "
                "document has no resolution base (it was loaded from an "
                "in-memory mapping)."
            )
        return str(Path(location))
    if _is_http(base):
        return urljoin(base.rstrip("/") + "/", location)
    return str((Path(base) / location).resolve())


def document_base(path: OmePath, location: str) -> str:
    """The base future relative paths in the resolved document count from.

    A ``"zarr"`` path's document lives in the group's own attributes, so the
    group location is the base; a ``"json"`` path's document is the file, so
    its parent is.
    """
    if path.type == PATH_TYPE_ZARR:
        return location
    if _is_http(location):
        return urljoin(location, ".").rstrip("/")
    return str(Path(location).parent)


def _read_zarr_document(location: str) -> dict[str, Any] | None:
    if _is_http(location):
        from .._remote_reader import RemoteZarrStore, open_remote_node

        if not remote_read_available():
            from ..from_ngff_zarr import _remote_backend_import_error

            raise _remote_backend_import_error(location, None)
        node = open_remote_node(RemoteZarrStore(location))
        return None if node is None else node.attrs.asdict()

    from .._zarrista_utils import read_group_attributes

    attributes = read_group_attributes(location, zarr_format=3)
    if not attributes:
        # A bare or spuriously empty ``zarr.json``: take the format 2
        # documents when they are actually there.
        fallback = read_group_attributes(location, zarr_format=2)
        if fallback is not None:
            return fallback
    return attributes


def _read_json_document(location: str) -> Any:
    if _is_http(location):
        with urlopen(location) as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(location).read_text(encoding="utf-8"))


def resolve_ome_document(
    path: OmePath, *, base: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Load the document an RFC-8 path locates.

    Returns ``(resolved location, document)``, where the document is the Zarr
    group's attribute mapping or the JSON file's root object. A prefixed
    extension path type is opaque to this package and reported as such.
    """
    if path.type not in (PATH_TYPE_ZARR, PATH_TYPE_JSON):
        raise ValueError(
            f"Path type {path.type!r} is not resolvable by ngff-zarr; only "
            f"'zarr' and 'json' paths are. A prefixed extension type is "
            f"opaque to implementations that do not recognize it."
        )
    location = resolve_location(path, base=base)
    if path.type == PATH_TYPE_ZARR:
        document = _read_zarr_document(location)
        if document is None:
            raise ValueError(
                f"No Zarr group with attributes found at '{location}' "
                f"(resolved from path '{path.path}')."
            )
        return location, document
    document = _read_json_document(location)
    if not isinstance(document, dict):
        raise ValueError(
            f"The JSON document at '{location}' (resolved from path "
            f"'{path.path}') is not an object."
        )
    return location, document
