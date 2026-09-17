# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Structural validation of RFC-8 node documents.

The seven node/collection rules extend the canonical rule table in
:mod:`ngff_zarr.structural_validation` (whose :class:`SpecRule` enum declares
their identifiers) and are dispatched by :func:`validate_collection`, a
sibling of ``validate_plate`` / ``validate_well`` for the RFC-8 document
kind. The rules walk the raw ``ome`` dict rather than the parsed model, so a
document the parser refuses (for instance one whose node lacks a ``type``)
can still be diagnosed with a stable rule identifier.

Shape errors beyond the rules, such as ``nodes`` not being an array, are the
JSON-schema pass's concern; the walker skips what it cannot traverse.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ..structural_validation import (
    SpecRule,
    ValidateOptions,
    ValidationError,
    ValidationLevel,
    is_rfc8_node_model,
)
from .model import RFC8_ID_PATTERN, Node, is_prefixed_identifier
from .serialize import node_to_ome_dict

#: The node types whose schema declares the ``nodes`` / ``path`` pair, and for
#: which exactly one of the two must be present. Unknown and extension node
#: types are exempt (open-world). Issue #714's multiscale stage adds
#: ``"multiscale"`` here rather than introducing a new rule.
XOR_NODE_TYPES = frozenset({"collection"})

#: The unprefixed path types the core specification defines.
CORE_PATH_TYPES = frozenset({"zarr", "json"})


def _iter_nodes(
    node: Mapping[str, Any], location: str
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield ``(location, node_dict)`` over the tree, depth-first, root first."""
    yield location, node
    children = node.get("nodes")
    if isinstance(children, list):
        for index, child in enumerate(children):
            if isinstance(child, Mapping):
                yield from _iter_nodes(child, f"{location}.nodes[{index}]")


def _iter_coordinate_systems(
    document: Mapping[str, Any],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield ``(location, system)`` for each declared coordinate system."""
    for location, node in _iter_nodes(document, "ome"):
        attributes = node.get("attributes")
        if not isinstance(attributes, Mapping):
            continue
        systems = attributes.get("coordinateSystems")
        if not isinstance(systems, list):
            continue
        for index, system in enumerate(systems):
            if isinstance(system, Mapping):
                yield (
                    f"{location}.attributes.coordinateSystems[{index}]",
                    system,
                )


def _iter_transform_reference_sites(
    document: Mapping[str, Any],
) -> Iterator[tuple[str, Any]]:
    """Yield ``(location, value)`` for each transformation input/output.

    Walks the top-level entries of every node's
    ``attributes.coordinateTransformations``; the transforms wrapped inside a
    sequence or bijection omit their references and are not walked.
    """
    for location, node in _iter_nodes(document, "ome"):
        attributes = node.get("attributes")
        if not isinstance(attributes, Mapping):
            continue
        transformations = attributes.get("coordinateTransformations")
        if not isinstance(transformations, list):
            continue
        for index, transformation in enumerate(transformations):
            if not isinstance(transformation, Mapping):
                continue
            base = f"{location}.attributes.coordinateTransformations[{index}]"
            for field in ("input", "output"):
                yield f"{base}.{field}", transformation.get(field)


def _iter_path_sites(
    document: Mapping[str, Any],
) -> Iterator[tuple[str, Any]]:
    """Yield ``(location, value)`` for every typed path in the document.

    Covers each node's own ``path`` and the ``path`` of every reference
    site, so an extension path type is judged the same wherever it appears.
    """
    for location, node in _iter_nodes(document, "ome"):
        yield f"{location}.path", node.get("path")
    for location, reference in _iter_transform_reference_sites(document):
        if isinstance(reference, Mapping):
            yield f"{location}.path", reference.get("path")


def _iter_id_sites(
    document: Mapping[str, Any],
) -> Iterator[tuple[str, Any, bool]]:
    """Yield ``(location, value, declared)`` for every id in the document.

    Declared sites are node ids and coordinate-system ids, which share the
    document-wide uniqueness namespace; reference sites reuse a declared id
    and are checked for format only.
    """
    for location, node in _iter_nodes(document, "ome"):
        if node.get("id") is not None:
            yield f"{location}.id", node.get("id"), True
    for location, system in _iter_coordinate_systems(document):
        if system.get("id") is not None:
            yield f"{location}.id", system.get("id"), True
    for location, reference in _iter_transform_reference_sites(document):
        if isinstance(reference, Mapping) and reference.get("id") is not None:
            yield f"{location}.id", reference.get("id"), False


def _declared_ids(document: Mapping[str, Any]) -> set[str]:
    """The node and coordinate-system ids declared in the document."""
    return {
        value
        for _, value, declared in _iter_id_sites(document)
        if declared and isinstance(value, str)
    }


def _require_nonempty_str(
    document: Mapping[str, Any], key: str, rule: SpecRule
) -> None:
    for location, node in _iter_nodes(document, "ome"):
        value = node.get(key)
        if not isinstance(value, str) or not value:
            raise ValidationError(
                rule,
                f"Node must declare a non-empty string {key!r}; got {value!r}.",
                location,
            )


def validate_node_type_required(document: Mapping[str, Any]) -> None:
    """Validate that every node declares a non-empty string ``type``.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.NODE_TYPE_REQUIRED` for the first node, in
        depth-first order, whose ``type`` is missing, empty, or not a string;
        location e.g. ``ome.nodes[1]``.
    """
    _require_nonempty_str(document, "type", SpecRule.NODE_TYPE_REQUIRED)


def validate_node_name_required(document: Mapping[str, Any]) -> None:
    """Validate that every node declares a non-empty string ``name``.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.NODE_NAME_REQUIRED` for the first node, in
        depth-first order, whose ``name`` is missing, empty, or not a string;
        location e.g. ``ome.nodes[1]``.
    """
    _require_nonempty_str(document, "name", SpecRule.NODE_NAME_REQUIRED)


def validate_node_name_unique(document: Mapping[str, Any]) -> None:
    """Validate that sibling nodes carry distinct names.

    RFC-8 requires names to be unique within the enclosing collection; two
    nodes in different collections may share one.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.NODE_NAME_UNIQUE` for the first child node, in
        depth-first order, whose ``name`` repeats an earlier sibling's;
        location e.g. ``ome.nodes[2]``.
    """
    for location, node in _iter_nodes(document, "ome"):
        children = node.get("nodes")
        if not isinstance(children, list):
            continue
        seen: set[str] = set()
        for index, child in enumerate(children):
            if not isinstance(child, Mapping):
                continue
            name = child.get("name")
            if not isinstance(name, str):
                continue
            if name in seen:
                raise ValidationError(
                    SpecRule.NODE_NAME_UNIQUE,
                    f"Node name {name!r} is repeated; node names must be "
                    f"unique within the enclosing collection.",
                    f"{location}.nodes[{index}]",
                )
            seen.add(name)


def validate_node_id_format(document: Mapping[str, Any]) -> None:
    """Validate that every declared node id matches ``[a-zA-Z0-9-_.]+``.

    The id production is shared across the document: node ids,
    coordinate-system ids, and the reference ids of transformation inputs
    and outputs are all checked here.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.NODE_ID_FORMAT` for the first declared or
        referenced ``id``, in depth-first order, that is not a match;
        location e.g. ``ome.nodes[0].id``.
    """
    for location, value, _ in _iter_id_sites(document):
        if not isinstance(value, str) or not RFC8_ID_PATTERN.match(value):
            raise ValidationError(
                SpecRule.NODE_ID_FORMAT,
                f"Id {value!r} does not match [a-zA-Z0-9-_.]+.",
                location,
            )


def validate_node_id_unique(document: Mapping[str, Any]) -> None:
    """Validate that declared ids are unique within the JSON document.

    Node ids and coordinate-system ids share the namespace; the reference
    ids of transformation inputs and outputs reuse a declared id and are
    exempt.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.NODE_ID_UNIQUE` for the first declared ``id``,
        node ids first then coordinate-system ids, that repeats an earlier
        one; location e.g. ``ome.nodes[3].id``.
    """
    seen: set[str] = set()
    for location, value, declared in _iter_id_sites(document):
        if not declared or not isinstance(value, str):
            continue
        if value in seen:
            raise ValidationError(
                SpecRule.NODE_ID_UNIQUE,
                f"Id {value!r} is already used in this document; ids must "
                f"be unique within the JSON document.",
                location,
            )
        seen.add(value)


def validate_node_nodes_xor_path(document: Mapping[str, Any]) -> None:
    """Validate that a collection carries exactly one of ``nodes`` / ``path``.

    Applies to the node types in :data:`XOR_NODE_TYPES`; unknown and
    extension node types are exempt.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.NODE_NODES_XOR_PATH` for the first applicable
        node, in depth-first order, declaring both or neither; location e.g.
        ``ome.nodes[1]``.
    """
    for location, node in _iter_nodes(document, "ome"):
        if node.get("type") not in XOR_NODE_TYPES:
            continue
        has_nodes = node.get("nodes") is not None
        has_path = node.get("path") is not None
        if has_nodes and has_path:
            raise ValidationError(
                SpecRule.NODE_NODES_XOR_PATH,
                "Node declares both 'nodes' and 'path'; exactly one must be present.",
                location,
            )
        if not has_nodes and not has_path:
            raise ValidationError(
                SpecRule.NODE_NODES_XOR_PATH,
                "Node declares neither 'nodes' nor 'path'; exactly one must "
                "be present.",
                location,
            )


def validate_path_type_known(document: Mapping[str, Any]) -> None:
    """Validate that every path ``type`` is ``zarr``, ``json``, or prefixed.

    Unprefixed identifiers are reserved for the core specification, which
    defines exactly ``zarr`` and ``json``; a prefixed identifier
    (``prefix:name``) belongs to the extension that registered the prefix
    and passes as opaque.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.PATH_TYPE_KNOWN` for the first node, in
        depth-first order, whose path ``type`` is missing, empty, not a
        string, or an unprefixed identifier outside the core set; location
        e.g. ``ome.nodes[1].path.type``.
    """
    for location, path in _iter_path_sites(document):
        if not isinstance(path, Mapping):
            continue
        value = path.get("type")
        if not isinstance(value, str) or not value:
            raise ValidationError(
                SpecRule.PATH_TYPE_KNOWN,
                f"Path must declare a non-empty string 'type'; got {value!r}.",
                f"{location}.type",
            )
        if value in CORE_PATH_TYPES or is_prefixed_identifier(value):
            continue
        raise ValidationError(
            SpecRule.PATH_TYPE_KNOWN,
            f"Path type {value!r} is not 'zarr', 'json', or a prefixed extension type.",
            f"{location}.type",
        )


def validate_coordinate_system_id_required(
    document: Mapping[str, Any],
) -> None:
    """Validate that every declared coordinate system carries an ``id``.

    RFC-8 references coordinate systems by id, so a system without one is
    unreachable; the ``name`` becomes an optional label. Malformed ids are
    reported by :func:`validate_node_id_format` instead.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.COORDINATE_SYSTEM_ID_REQUIRED` for the first
        coordinate system, in depth-first order, without an ``id``; location
        e.g. ``ome.nodes[0].attributes.coordinateSystems[1]``.
    """
    for location, system in _iter_coordinate_systems(document):
        if system.get("id") is None:
            raise ValidationError(
                SpecRule.COORDINATE_SYSTEM_ID_REQUIRED,
                "Coordinate system declares no 'id'; RFC-8 references "
                "coordinate systems by id.",
                location,
            )


def validate_reference_id_required(document: Mapping[str, Any]) -> None:
    """Validate that transformation inputs and outputs are id references.

    RFC-8 replaces the name-based RFC-5 identifiers: each top-level
    transformation's ``input`` and ``output`` must be an object carrying an
    ``id``.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.REFERENCE_ID_REQUIRED` for the first
        transformation ``input``/``output``, in depth-first order, that is
        absent, not an object, or without an ``id``; location e.g.
        ``ome.nodes[0].attributes.coordinateTransformations[0].input``.
    """
    for location, reference in _iter_transform_reference_sites(document):
        if not isinstance(reference, Mapping) or reference.get("id") is None:
            raise ValidationError(
                SpecRule.REFERENCE_ID_REQUIRED,
                "Transformation reference must be an object with an 'id'.",
                location,
            )


def validate_reference_path_required(document: Mapping[str, Any]) -> None:
    """Validate that unresolved references carry a ``path``.

    A reference whose ``id`` is declared in this document (as a node id or a
    coordinate-system id) is internal; any other reference is external, and
    RFC-8 requires external references to locate their document with a
    ``path``. What the path points at is not resolved here.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.REFERENCE_PATH_REQUIRED` for the first
        transformation reference, in depth-first order, whose ``id``
        resolves to nothing in-document and that carries no ``path``;
        location e.g.
        ``ome.nodes[0].attributes.coordinateTransformations[0].output``.
    """
    declared = _declared_ids(document)
    for location, reference in _iter_transform_reference_sites(document):
        if not isinstance(reference, Mapping):
            continue
        value = reference.get("id")
        if not isinstance(value, str) or value in declared:
            continue
        if reference.get("path") is None:
            raise ValidationError(
                SpecRule.REFERENCE_PATH_REQUIRED,
                f"Reference id {value!r} is not declared in this document, "
                f"so the reference must carry a 'path'.",
                location,
            )


#: The RFC-8 rules in canonical evaluation order (the SpecRule declaration
#: order), dispatched fail-fast by :func:`validate_collection`.
_COLLECTION_RULES = (
    validate_node_type_required,
    validate_node_name_required,
    validate_node_name_unique,
    validate_node_id_format,
    validate_node_id_unique,
    validate_node_nodes_xor_path,
    validate_path_type_known,
    validate_coordinate_system_id_required,
    validate_reference_id_required,
    validate_reference_path_required,
)


def validate_collection(
    root: Mapping[str, Any] | Node,
    version: object | None = None,
    options: ValidateOptions | None = None,
) -> None:
    """Run the RFC-8 node/collection rules in canonical order, fail-fast.

    ``root`` is either the raw ``ome`` dict or a parsed
    :class:`~ngff_zarr.rfc8.model.Node` tree (serialized before checking, so
    both forms are judged by the same walker). Names and ids are checked per
    document: nodes a ``path`` points at live in their own documents and are
    validated when those are read.

    ``version`` gates the rules the RFC-4 way: they run when it is ``None``
    (a document with no declared version gets the strict reading) or when
    :func:`~ngff_zarr.structural_validation.is_rfc8_node_model` says the
    version stores the node model, and are inert when an earlier version is
    declared.

    Raises
    ------
    ValidationError
        From the first violated rule, with its :class:`SpecRule` and a
        dotted-segment location.
    """
    options = options or ValidateOptions()
    if options.level == ValidationLevel.SCHEMA_ONLY:
        return
    if version is not None and not is_rfc8_node_model(version):
        return
    document = node_to_ome_dict(root) if isinstance(root, Node) else root
    for rule in _COLLECTION_RULES:
        rule(document)
