# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Serialization between the RFC-8 node model and its ``ome`` document."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .model import Collection, Node, OmePath

#: The node keys this package models; everything else round-trips through
#: ``Node.extra``. ``version`` is the root-only key handled by the (de)serializer.
_NODE_FIELDS = frozenset({"type", "id", "name", "attributes", "nodes", "path"})


def node_to_ome_dict(node: Node, *, version: str | None = None) -> dict[str, Any]:
    """Serialize a node tree to the dict stored as the ``ome`` value.

    ``version`` stamps the root node (the RFC-8 root-only key) and is never
    passed down to children. ``attributes`` is deep-copied verbatim, unknown
    and prefixed keys included, and so are the ``extra`` keys, emitted after
    the modeled ones in their captured order. ``None``-valued optional fields
    are omitted rather than written as JSON ``null``.
    """
    out: dict[str, Any] = {}
    if version is not None:
        out["version"] = str(version)
    out["type"] = node.type
    if node.id is not None:
        out["id"] = node.id
    out["name"] = node.name
    if node.attributes is not None:
        out["attributes"] = copy.deepcopy(node.attributes)
    if node.nodes is not None:
        out["nodes"] = [node_to_ome_dict(child) for child in node.nodes]
    if node.path is not None:
        out["path"] = {"type": node.path.type, "path": node.path.path}
    for key, value in node.extra.items():
        out[key] = copy.deepcopy(value)
    return out


def _require_str(value: Any, key: str, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"An RFC-8 node must declare a non-empty string {key!r}; "
            f"got {value!r} at {location}."
        )
    return value


def _node_from_dict(data: Mapping[str, Any], location: str) -> Node:
    """Parse one node dict, recursing into ``nodes``.

    The parser guarantees the model invariants (well-typed core keys) and is
    otherwise open-world: unknown keys land in ``extra`` and unknown node
    types produce a generic :class:`Node`. Rule violations a document can
    express without breaking the model, such as carrying both ``nodes`` and
    ``path``, parse fine and are reported by
    :func:`ngff_zarr.rfc8.validate_collection`.
    """
    node_type = _require_str(data.get("type"), "type", location)
    name = _require_str(data.get("name"), "name", location)

    node_id = data.get("id")
    if node_id is not None and not isinstance(node_id, str):
        raise ValueError(
            f"An RFC-8 node 'id' must be a string; got {node_id!r} at {location}."
        )

    attributes = data.get("attributes")
    if attributes is not None:
        if not isinstance(attributes, Mapping):
            raise ValueError(
                f"An RFC-8 node 'attributes' must be an object; "
                f"got {attributes!r} at {location}."
            )
        attributes = copy.deepcopy(dict(attributes))

    raw_nodes = data.get("nodes")
    nodes: list[Node] | None = None
    if raw_nodes is not None:
        if not isinstance(raw_nodes, list):
            raise ValueError(
                f"An RFC-8 node 'nodes' must be an array of nodes; "
                f"got {raw_nodes!r} at {location}."
            )
        nodes = []
        for index, child in enumerate(raw_nodes):
            child_location = f"{location}.nodes[{index}]"
            if not isinstance(child, Mapping):
                raise ValueError(
                    f"An RFC-8 node 'nodes' entry must be an object; "
                    f"got {child!r} at {child_location}."
                )
            nodes.append(_node_from_dict(child, child_location))

    raw_path = data.get("path")
    path: OmePath | None = None
    if raw_path is not None:
        if not isinstance(raw_path, Mapping):
            raise ValueError(
                f"An RFC-8 node 'path' must be an object with 'type' and "
                f"'path'; got {raw_path!r} at {location}."
            )
        path = OmePath(
            type=_require_str(raw_path.get("type"), "path.type", location),
            path=_require_str(raw_path.get("path"), "path.path", location),
        )

    extra = {
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key not in _NODE_FIELDS
    }

    if node_type == "collection":
        node = Collection(
            name,
            id=node_id,
            attributes=attributes,
            nodes=nodes,
            path=path,
            extra=extra,
        )
    else:
        node = Node(
            type=node_type,
            name=name,
            id=node_id,
            attributes=attributes,
            nodes=nodes,
            path=path,
            extra=extra,
        )
    return node


def node_from_ome_dict(data: Mapping[str, Any]) -> tuple[Node, str | None]:
    """Parse an ``ome`` value into ``(root node, declared version)``.

    The root's ``version`` key is returned separately rather than modeled on
    the node: RFC-8 puts it on the root only, and the serializer stamps it
    back. A non-string ``version`` is reported rather than carried.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"An RFC-8 'ome' value must be an object; got {data!r}.")
    version = data.get("version")
    if version is not None and not isinstance(version, str):
        raise ValueError(f"An RFC-8 'version' must be a string; got {version!r}.")
    root_data = {key: value for key, value in data.items() if key != "version"}
    return _node_from_dict(root_data, "ome"), version
