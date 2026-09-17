# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The in-memory handle for an RFC-8 node document and its neighborhood."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Union

from .._lru_cache import LRUCache
from .._supported_versions import NgffVersion
from .model import Node, OmePath, Reference, is_collection_node
from .resolve import document_base, resolve_location, resolve_ome_document
from .serialize import node_from_ome_dict

#: What :meth:`NgffCollection.load` can hand back: a nested collection, a
#: multiscales image read through :func:`ngff_zarr.from_ome_zarr`, or the
#: parsed node itself when ngff-zarr has no richer model for its type yet.
LoadedNode = Union["NgffCollection", "NgffMultiscales", Node]  # noqa: F821


@dataclass
class NgffCollection:
    """An RFC-8 node document plus the context needed to resolve it.

    ``root`` is typically a :class:`~ngff_zarr.rfc8.model.Collection` but may
    be any node: RFC-8 lets every node type head a document. ``base`` anchors
    relative path resolution: the Zarr group directory or URL the document's
    ``zarr.json`` lives in, or a standalone JSON file's parent. ``None``
    (a document built in memory) leaves relative paths unresolvable.

    ``root_attributes`` carries the non-OME root attribute keys of the group
    the document was read from, mirroring
    :attr:`~ngff_zarr.multiscales.NgffMultiscales.root_attributes`.
    """

    root: Node
    base: str | None = None
    version: str = NgffVersion.V09dev3.value
    root_attributes: dict[str, Any] | None = None
    _cache: LRUCache = field(
        default_factory=lambda: LRUCache(max_size=32),
        init=False,
        repr=False,
        compare=False,
    )

    def walk(self) -> Iterator[Node]:
        """Yield every node in the document, depth-first, root first."""

        def _walk(node: Node) -> Iterator[Node]:
            yield node
            for child in node.nodes or ():
                yield from _walk(child)

        return _walk(self.root)

    def node(self, id_or_name: str) -> Node:
        """The document's node with the given ``id``, else with the ``name``.

        Ids are unique within a document, so they are looked up first; names
        are only unique among siblings, and the first match in depth-first
        order wins.
        """
        for node in self.walk():
            if node.id == id_or_name:
                return node
        for node in self.walk():
            if node.name == id_or_name:
                return node
        raise KeyError(f"No node with id or name {id_or_name!r} in this document.")

    def load(self, target: Node | Reference | str) -> LoadedNode:
        """Materialize what ``target`` designates, dereferencing its path.

        ``target`` may be a node of this document (or its ``id`` / ``name``
        as a string), or a :class:`~ngff_zarr.rfc8.model.Reference`. What
        comes back depends on what the dereferenced document holds:

        - an RFC-8 collection document nests as another
          :class:`NgffCollection`, its base moved to the referenced location;
        - a multiscales image store (any OME-Zarr version up to 0.9.dev1)
          reads through :func:`ngff_zarr.from_ome_zarr` into
          :class:`~ngff_zarr.multiscales.NgffMultiscales`;
        - any other RFC-8 node document returns its parsed root
          :class:`~ngff_zarr.rfc8.model.Node`.

        An inline node (no ``path``) stays in this document: a collection
        nests sharing this base, and any other node is returned as parsed.
        Dereferenced documents are cached per resolved location.
        """
        if isinstance(target, str):
            target = self.node(target)
        if isinstance(target, Reference):
            if target.path is None:
                return self.load(self.node(target.id))
            loaded = self._load_document(target.path)
            if isinstance(loaded, NgffCollection):
                for node in loaded.walk():
                    if node.id == target.id:
                        return loaded.load(node)
                raise KeyError(
                    f"Reference id {target.id!r} is not declared by the "
                    f"document at {target.path.path!r}"
                )
            if isinstance(loaded, Node) and loaded.id != target.id:
                raise KeyError(
                    f"Reference id {target.id!r} is not declared by the "
                    f"document at {target.path.path!r}"
                )
            # A multiscales image store predates RFC-8 and declares no ids.
            return loaded
        if target.path is not None:
            return self._load_document(target.path)
        if is_collection_node(target):
            return NgffCollection(root=target, base=self.base, version=self.version)
        return target

    def _load_document(self, path: OmePath) -> LoadedNode:
        location = resolve_location(path, base=self.base)
        cached = self._cache.get((path.type, location))
        if cached is not None:
            return cached
        location, document = resolve_ome_document(path, base=self.base)
        loaded = self._dispatch_document(path, location, document)
        self._cache.set((path.type, location), loaded)
        return loaded

    def _dispatch_document(
        self, path: OmePath, location: str, document: Mapping[str, Any]
    ) -> LoadedNode:
        ome = document.get("ome")
        node_shaped = (
            isinstance(ome, Mapping) and "type" in ome and "multiscales" not in ome
        )
        if node_shaped:
            root, declared = node_from_ome_dict(ome)
            if is_collection_node(root):
                return NgffCollection(
                    root=root,
                    base=document_base(path, location),
                    version=declared or self.version,
                )
            return root
        has_multiscales = "multiscales" in document or (
            isinstance(ome, Mapping) and "multiscales" in ome
        )
        if has_multiscales:
            if path.type != "zarr":
                raise ValueError(
                    f"The document at '{location}' holds multiscales image "
                    f"metadata, which only a 'zarr' path can designate; got "
                    f"path type {path.type!r}."
                )
            from ..from_ngff_zarr import from_ome_zarr

            return from_ome_zarr(location)
        raise ValueError(
            f"The document at '{location}' is neither an RFC-8 node "
            f"document nor a multiscales image."
        )
