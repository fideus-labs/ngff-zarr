# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Reading and writing RFC-8 node documents (OME-Zarr 0.9.dev3).

The Zarr entry points handle documents stored in a group's ``attributes.ome``;
the JSON entry points handle the RFC's standalone storage medium, a ``*.json``
file with a root ``ome`` key. Writing is metadata-only: a collection document
references image stores, it does not contain their arrays, so the referenced
stores are written separately (:func:`ngff_zarr.to_ome_zarr`) at their own
versions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from importlib_resources import files as file_resources

from .._store_types import StoreLike
from .._supported_versions import NgffVersion
from .._zarrista_utils import (
    OME_ROOT_KEYS,
    consolidate_metadata,
    create_zarrista_group,
    has_consolidated_metadata,
    normalize_store,
)
from .collection import NgffCollection
from .model import Node
from .resolve import _is_http, _read_json_document
from .serialize import node_from_ome_dict, node_to_ome_dict
from .validation import validate_collection


def load_rfc8_node_schema() -> dict:
    """Load the RFC-8 node JSON schema bundled with ngff-zarr.

    RFC-8 has no normative upstream schema yet (the RFC is in draft and
    ``ome/ngff-spec`` carries none), so unlike the vendored ``spec/0.x``
    trees this schema is authored in this repository from the RFC text,
    like the RFC-4 orientation schema beside it.
    """
    schema = (
        file_resources("ngff_zarr")
        .joinpath(Path("spec") / Path("rfc") / Path("8") / "node.schema.json")
        .read_text()
    )
    return json.loads(schema)


def _validate_document(ome: Mapping[str, Any], version: str | None) -> None:
    """Run the structural rules, then the bundled schema, on a raw document.

    The vendored ``spec/0.9`` tree cannot check a 0.9.dev3 document (its
    ``_version`` schema enumerates only ``0.9.dev1``, and the tree stays
    verbatim upstream), so :func:`ngff_zarr.validate` is never involved
    here; the repo-authored RFC-8 schema is used instead.
    """
    validate_collection(ome, version=version)
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise ImportError(
            "jsonschema is required to validate NGFF metadata - install the ngff-zarr[validate] extra"
        )
    validator = Draft202012Validator(load_rfc8_node_schema())
    validator.validate(dict(ome))


def _node_document_or_raise(attrs: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    """The ``ome`` node document in ``attrs``, or a pointed error."""
    ome = attrs.get("ome")
    is_image = "multiscales" in attrs or (
        isinstance(ome, Mapping) and "multiscales" in ome
    )
    if is_image:
        raise ValueError(
            f"The input at {where} is a multiscales image store, not an "
            "RFC-8 node document. Read it with from_ome_zarr()."
        )
    if not isinstance(ome, Mapping) or "type" not in ome:
        raise ValueError(
            f"The input at {where} carries no RFC-8 node document: expected "
            "an 'ome' object with a 'type' key (OME-Zarr 0.9.dev3)."
        )
    return ome


def from_collection_zarr(
    store: StoreLike,
    validate: bool = False,
    storage_options: dict | None = None,
) -> NgffCollection:
    """Read an RFC-8 node document from a Zarr store's root group.

    Parameters
    ----------
    store:
        Local directory path, remote URL string, or key-to-bytes mapping of
        a Zarr store whose root group attributes carry the document under
        ``ome``. The root may be any node type; a collection is typical.
    validate:
        Run the RFC-8 structural rules and the bundled node schema against
        the raw document before parsing.
    storage_options:
        Passed to the remote backend for URL stores, as in
        :func:`ngff_zarr.from_ome_zarr`.

    Returns
    -------
    NgffCollection
        The parsed document with its resolution base set to the store, so
        relative paths in the document dereference from it.
    """
    from .._remote_reader import RemoteZarrStore, remote_read_available
    from ..from_ngff_zarr import (
        REMOTE_URL_SCHEMES,
        _open_root_node,
        _remote_backend_import_error,
    )

    base: str | None = None
    if isinstance(store, str) and store.startswith(REMOTE_URL_SCHEMES):
        if not remote_read_available():
            raise _remote_backend_import_error(store, None)
        base = store.rstrip("/")
        store = RemoteZarrStore(store, storage_options=storage_options)
    elif isinstance(store, RemoteZarrStore):
        base = store.url
    elif isinstance(store, (str, os.PathLike)):
        base = str(Path(store).resolve())

    root = _open_root_node(store, None)
    attrs = root.attrs.asdict()
    ome = _node_document_or_raise(attrs, f"'{store}'")
    declared = ome.get("version") if isinstance(ome.get("version"), str) else None
    if validate:
        _validate_document(ome, declared)
    node, declared = node_from_ome_dict(ome)
    root_attributes = {
        key: value for key, value in attrs.items() if key not in OME_ROOT_KEYS
    }
    return NgffCollection(
        root=node,
        base=base,
        version=declared or NgffVersion.V09dev3.value,
        root_attributes=root_attributes or None,
    )


def from_collection_json(
    source: str | os.PathLike | Mapping[str, Any],
    validate: bool = False,
) -> NgffCollection:
    """Read an RFC-8 node document from standalone JSON.

    ``source`` is the path or ``http(s)`` URL of a JSON file whose root
    object carries the document under ``ome``, or such a parsed object
    itself. For an in-memory mapping the collection has no resolution base,
    so its relative paths cannot be dereferenced.
    """
    base: str | None = None
    if isinstance(source, Mapping):
        document: Mapping[str, Any] = source
        where = "the given mapping"
    else:
        location = str(source)
        if _is_http(location):
            base = urljoin(location, ".").rstrip("/")
        else:
            location = str(Path(location).resolve())
            base = str(Path(location).parent)
        document = _read_json_document(location)
        where = f"'{location}'"
        if not isinstance(document, Mapping):
            raise ValueError(f"The JSON document at {where} is not an object.")
    ome = _node_document_or_raise(document, where)
    declared = ome.get("version") if isinstance(ome.get("version"), str) else None
    if validate:
        _validate_document(ome, declared)
    node, declared = node_from_ome_dict(ome)
    return NgffCollection(
        root=node,
        base=base,
        version=declared or NgffVersion.V09dev3.value,
    )


def _collection_root(collection: NgffCollection | Node) -> Node:
    if isinstance(collection, NgffCollection):
        return collection.root
    if isinstance(collection, Node):
        return collection
    raise TypeError(
        f"Expected an NgffCollection or a Node; got {type(collection).__name__}."
    )


def _check_collection_version(version: str) -> str:
    if str(version) != NgffVersion.V09dev3.value:
        raise ValueError(
            f"Unsupported version for an RFC-8 node document: {version!r}. "
            "Only '0.9.dev3' stores the node model."
        )
    return NgffVersion.V09dev3.value


def to_collection_zarr(
    store: StoreLike,
    collection: NgffCollection | Node,
    version: str = "0.9.dev3",
    overwrite: bool = True,
    validate: bool = True,
    root_attributes: Mapping[str, Any] | None = None,
) -> None:
    """Write an RFC-8 node document as a Zarr store's root group.

    Writes metadata only: a Zarr format 3 group whose ``attributes.ome`` is
    the serialized document. The stores the document references are written
    separately. ``root_attributes`` (or, when omitted, the collection's own
    ``root_attributes``) are written beside the OME key.

    ``overwrite`` replaces the root group's metadata document, keeping its
    non-OME attributes; unlike :func:`ngff_zarr.to_ome_zarr` it never removes
    the tree beneath the root, which commonly holds the very stores the
    document references. Without ``overwrite`` the document is merged into an
    existing root's attributes. Consolidated root metadata is refreshed when
    the store already carries it.
    """
    from ..to_ngff_zarr import _check_root_attributes, _foreign_root_attrs

    version = _check_collection_version(version)
    root = _collection_root(collection)
    if validate:
        validate_collection(root, version=version)
    if root_attributes is None and isinstance(collection, NgffCollection):
        root_attributes = collection.root_attributes
    store_path = normalize_store(store)
    consolidated = has_consolidated_metadata(store_path, 3)
    attributes: dict[str, Any] = {}
    if overwrite:
        # Replace the root metadata document, but never the tree beneath it:
        # a collection root commonly holds the very stores its document
        # references, so the recursive removal ``to_ome_zarr`` performs on
        # overwrite would destroy them. Non-OME root attributes survive.
        attributes.update(_foreign_root_attrs(store_path))
        store_path.mkdir(parents=True, exist_ok=True)
        (store_path / "zarr.json").unlink(missing_ok=True)
    attributes.update(_check_root_attributes(root_attributes))
    attributes["ome"] = node_to_ome_dict(root, version=version)
    create_zarrista_group(store_path, attributes, 3)
    if consolidated:
        consolidate_metadata(store_path, zarr_format=3)


def to_collection_json(
    collection: NgffCollection | Node,
    path: str | os.PathLike | None = None,
    version: str = "0.9.dev3",
    validate: bool = True,
) -> dict[str, Any]:
    """Serialize an RFC-8 node document to standalone JSON.

    Returns the ``{"ome": ...}`` document; when ``path`` is given it is also
    written there as UTF-8 JSON.
    """
    version = _check_collection_version(version)
    root = _collection_root(collection)
    if validate:
        validate_collection(root, version=version)
    document = {"ome": node_to_ome_dict(root, version=version)}
    if path is not None:
        Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document
