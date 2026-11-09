# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 "Collections and Extensibility" support (OME-Zarr 0.9.dev3)."""

from .collection import NgffCollection
from .coordinates import (
    coordinate_systems,
    coordinate_transformations,
    set_coordinate_systems,
    set_coordinate_transformations,
)
from .io import (
    from_collection_json,
    from_collection_zarr,
    to_collection_json,
    to_collection_zarr,
)
from .model import (
    PATH_TYPE_JSON,
    PATH_TYPE_ZARR,
    RFC8_ID_PATTERN,
    Collection,
    Node,
    OmePath,
    Reference,
    is_collection_node,
    is_prefixed_identifier,
)
from .resolve import resolve_location, resolve_ome_document
from .serialize import node_from_ome_dict, node_to_ome_dict
from .validation import validate_collection

__all__ = [
    "PATH_TYPE_JSON",
    "PATH_TYPE_ZARR",
    "RFC8_ID_PATTERN",
    "Collection",
    "NgffCollection",
    "Node",
    "OmePath",
    "Reference",
    "from_collection_json",
    "from_collection_zarr",
    "is_collection_node",
    "is_prefixed_identifier",
    "coordinate_systems",
    "coordinate_transformations",
    "node_from_ome_dict",
    "node_to_ome_dict",
    "set_coordinate_systems",
    "set_coordinate_transformations",
    "resolve_location",
    "resolve_ome_document",
    "to_collection_json",
    "to_collection_zarr",
    "validate_collection",
]
