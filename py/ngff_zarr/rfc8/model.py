# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The RFC-8 node model: typed paths, references, nodes and collections.

RFC-8 ("Collections and Extensibility",
https://ngff.openmicroscopy.org/rfc/8/index.html) replaces the ``multiscales``
wrapper with a typed node document stored under the ``ome`` key of a Zarr
group's attributes or of a standalone JSON file. This module holds the
in-memory model of that document; serialization lives in
:mod:`ngff_zarr.rfc8.serialize` and the structural rules in
:mod:`ngff_zarr.rfc8.validation`.

The model is deliberately open-world. RFC-8 declares node types, attribute
keys and path types as extension points, so a node of a type this package has
no dedicated class for (a prefixed extension type, or ``"multiscale"`` /
``"singlescale"`` until issue #714's later stages land) is carried as a
generic :class:`Node` and round-trips unchanged, and ``attributes`` is opaque
JSON whose prefixed extension keys (e.g. ``mobie:grid``) are preserved
verbatim, never interpreted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: The RFC-8 id production, shared by node ids and reference ids.
RFC8_ID_PATTERN = re.compile(r"[a-zA-Z0-9\-_.]+\Z")

#: Path type for a node stored in a Zarr group: implementations append
#: ``zarr.json`` to the path to reach the metadata document.
PATH_TYPE_ZARR = "zarr"

#: Path type for a node stored in a standalone JSON file: the path is the
#: metadata document itself.
PATH_TYPE_JSON = "json"


def is_prefixed_identifier(value: str) -> bool:
    """Whether ``value`` is a prefixed extension identifier (``prefix:name``).

    Unprefixed identifiers are reserved for the core specification; custom
    extensions introduce prefixed ones, where the prefix names the user or
    organization maintaining the extension.
    """
    prefix, separator, name = value.partition(":")
    return bool(separator) and bool(prefix) and bool(name)


@dataclass(frozen=True)
class OmePath:
    """RFC-8 ``Path``: a typed locator for out-of-document metadata.

    ``type`` is :data:`PATH_TYPE_ZARR`, :data:`PATH_TYPE_JSON`, or a prefixed
    extension type (e.g. ``"myorg:zip"``). ``path`` is a relative path
    (IETF RFC 1808, resolved against the document that holds it), an absolute
    ``file://`` path (IETF RFC 8089), or an ``http(s)://`` URL
    (IETF RFC 1738).

    Named ``OmePath`` rather than the RFC's bare ``Path``: half this package
    imports :class:`pathlib.Path` unqualified.
    """

    type: str
    path: str


@dataclass(frozen=True)
class Reference:
    """RFC-8 ``Reference`` to a node or coordinate system by ``id``.

    ``path`` locates the document that declares the referenced ``id``; it is
    required for external references and omitted for in-document ones.
    """

    id: str
    path: OmePath | None = None


@dataclass
class Node:
    """RFC-8 ``Node``: the base of every 0.9.dev3 metadata document.

    ``type`` and ``name`` are required by the RFC; ``id`` (unique within the
    JSON document), ``attributes``, and the container fields ``nodes`` /
    ``path`` are optional. Exactly one of ``nodes`` and ``path`` must be
    present on the node types that define them; that is a validation rule
    (:func:`ngff_zarr.rfc8.validate_collection`), not a constructor error, so
    an invalid document can be parsed, inspected and repaired.

    ``extra`` captures unrecognized top-level node keys on read and is
    serialized back on write: unlike the ``extra`` of the multiscales
    metadata models, RFC-8 extensions must survive a round trip. No node
    carries a ``version`` field; the serializer stamps the root's.
    """

    type: str
    name: str
    id: str | None = field(default=None, kw_only=True)
    attributes: dict[str, Any] | None = field(default=None, kw_only=True)
    nodes: list[Node] | None = field(default=None, kw_only=True)
    path: OmePath | None = field(default=None, kw_only=True)
    extra: dict[str, Any] = field(default_factory=dict, kw_only=True, repr=False)


@dataclass
class Collection(Node):
    """A ``type == "collection"`` node: an arbitrary group of nodes."""

    type: str = field(default="collection", init=False)


def is_collection_node(node: Node) -> bool:
    """Whether ``node`` is a collection, whichever class carries it."""
    return node.type == "collection"
