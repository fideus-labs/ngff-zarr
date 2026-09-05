# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-8 "Collections and Extensibility" support (OME-Zarr 0.9.dev3)."""

from .collection import NgffCollection
from .io import (
    from_collection_json,
    from_collection_zarr,
    load_rfc8_node_schema,
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
from .validation import (
    validate_collection,
    validate_node_id_format,
    validate_node_id_unique,
    validate_node_name_required,
    validate_node_name_unique,
    validate_node_nodes_xor_path,
    validate_node_type_required,
    validate_path_type_known,
)

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
    "load_rfc8_node_schema",
    "node_from_ome_dict",
    "node_to_ome_dict",
    "resolve_location",
    "resolve_ome_document",
    "to_collection_json",
    "to_collection_zarr",
    "validate_collection",
    "validate_node_id_format",
    "validate_node_id_unique",
    "validate_node_name_required",
    "validate_node_name_unique",
    "validate_node_nodes_xor_path",
    "validate_node_type_required",
    "validate_path_type_known",
]
