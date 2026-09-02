# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The scene metadata: spatial relationships between images.

RFC-5 introduced scene metadata for collections of images that share a
coordinate space, stored as ``ome.scene`` on a Zarr group of the 0.6 family
(0.6 and 0.9.dev1), with name-based coordinate-system identifiers. RFC-8
embeds the same information as a ``scene`` attribute of a collection node
(OME-Zarr 0.9.dev3), with id-based references. Both shapes parse into one
in-memory :class:`Scene`, so tools written against it read either store.

The scene's transformations relate coordinate systems that mostly live in
other documents (the collection's member images); identifiers are therefore
preserved verbatim, unresolved names and ids included, and dereferencing is
left to the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._store_types import StoreLike
from ..v06.zarr_metadata import Metadata as _MetadataV06
from ..v09.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Transform,
    _filter_axis_dict,
)
from .coordinates import COORDINATE_SYSTEMS_KEY, _without_none
from .model import Node

#: The RFC-8 attribute key (and the RFC-5 ``ome`` key) this module owns.
SCENE_KEY = "scene"


@dataclass
class Scene:
    """Scene metadata: transformations between referenced coordinate systems.

    ``coordinateTransformations`` is required by both the RFC-5 and RFC-8
    shapes; ``coordinateSystems`` declares the systems the collection itself
    owns (the first one is the recommended common reference).
    """

    coordinateTransformations: list[Transform]
    coordinateSystems: list[CoordinateSystem] | None = None


def _identifier_from(raw: Any) -> CoordinateSystemIdentifier | None:
    """An identifier carrying the raw reference fields verbatim.

    The multiscales parser drops a ``name`` that resolves to no declared
    system; a scene's references usually point into member documents, so
    every field survives here.
    """
    if not isinstance(raw, Mapping):
        return None
    return CoordinateSystemIdentifier(
        path=raw.get("path"),
        name=raw.get("name"),
        id=raw.get("id") if isinstance(raw.get("id"), str) else None,
    )


def _parse_scene(raw: Mapping[str, Any], version: str | None) -> Scene:
    systems = None
    if raw.get(COORDINATE_SYSTEMS_KEY) is not None:
        systems = [
            CoordinateSystem(
                name=entry.get("name"),
                axes=[Axis(**_filter_axis_dict(axis)) for axis in entry["axes"]],
                id=entry.get("id"),
            )
            for entry in raw[COORDINATE_SYSTEMS_KEY]
        ]
    raw_transforms = raw.get("coordinateTransformations") or []
    transforms = _MetadataV06._parse_transforms(raw_transforms, systems or [], version)
    # Restore the references verbatim: the wrapped parser resolves names
    # against the declared systems and drops the rest.
    for parsed, entry in zip(transforms, raw_transforms):
        if isinstance(entry, Mapping):
            parsed.input = _identifier_from(entry.get("input"))
            parsed.output = _identifier_from(entry.get("output"))
    return Scene(coordinateTransformations=transforms, coordinateSystems=systems)


def _serialize_scene(value: Scene) -> dict[str, Any]:
    from dataclasses import asdict

    serialized: dict[str, Any] = {}
    if value.coordinateSystems is not None:
        systems = []
        for system in value.coordinateSystems:
            entry: dict[str, Any] = {}
            if system.id is not None:
                entry["id"] = system.id
            if system.name is not None:
                entry["name"] = system.name
            entry["axes"] = [_without_none(asdict(axis)) for axis in system.axes]
            systems.append(entry)
        serialized[COORDINATE_SYSTEMS_KEY] = systems
    serialized["coordinateTransformations"] = [
        _without_none(transform.to_dict())
        for transform in value.coordinateTransformations
    ]
    return serialized


def scene(node: Node, version: str | None = "0.9.dev3") -> Scene | None:
    """The parsed ``scene`` attribute of ``node``, or ``None``."""
    raw = (node.attributes or {}).get(SCENE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"A 'scene' attribute must be an object; got {raw!r}.")
    return _parse_scene(raw, version)


def set_scene(node: Node, value: Scene | None) -> None:
    """Write ``value`` into ``node.attributes``, replacing the key."""
    if node.attributes is None:
        node.attributes = {}
    if value is None:
        node.attributes.pop(SCENE_KEY, None)
        return
    node.attributes[SCENE_KEY] = _serialize_scene(value)


def read_scene(store: StoreLike, version: str | None = None) -> Scene | None:
    """Read the scene metadata of a store's root group, or ``None``.

    Handles both shapes: the RFC-5 ``ome.scene`` of the 0.6 family and the
    ``scene`` attribute of an RFC-8 node document (OME-Zarr 0.9.dev3).
    """
    from ..from_ngff_zarr import _open_root_node

    root = _open_root_node(store, version)
    root_attrs = root.attrs.asdict()
    ome = root_attrs.get("ome")
    if not isinstance(ome, Mapping):
        return None
    declared = ome.get("version") if isinstance(ome.get("version"), str) else None
    raw = ome.get(SCENE_KEY)
    if raw is None and isinstance(ome.get("attributes"), Mapping):
        # An RFC-8 node document carries the scene in its attributes.
        raw = ome["attributes"].get(SCENE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"A 'scene' entry must be an object; got {raw!r}.")
    return _parse_scene(raw, declared or version)


def write_scene(store: StoreLike, value: Scene, version: str = "0.6") -> None:
    """Write ``value`` as the scene metadata of a store's root group.

    For the 0.6 family the scene lands at ``ome.scene`` beside any existing
    OME metadata, tagging a fresh group with the version's on-disk string.
    For ``"0.9.dev3"`` the root must already be an RFC-8 node document (a
    collection written with :func:`ngff_zarr.to_collection_zarr`), and the
    scene lands in its ``attributes``. Only the root metadata document is
    rewritten; nothing beneath the root is touched.
    """
    import copy

    from .._supported_versions import (
        V06_ONDISK_VERSION,
        NgffVersion,
        is_v06_version,
    )
    from .._zarrista_utils import (
        consolidate_metadata,
        create_zarrista_group,
        has_consolidated_metadata,
        normalize_store,
        read_group_attributes,
    )

    version = NgffVersion(version).value
    if version == "0.6":
        ondisk = V06_ONDISK_VERSION.value
    elif is_v06_version(version) or version in ("0.9.dev1", "0.9.dev3"):
        ondisk = version
    else:
        raise ValueError(
            f"Unsupported version for scene metadata: {version!r}. Scene "
            "metadata is defined for OME-Zarr 0.6 and the 0.9 development "
            "series."
        )

    store_path = normalize_store(store)
    existing = read_group_attributes(store_path, zarr_format=3) or {}
    ome = copy.deepcopy(existing.get("ome")) if existing.get("ome") else {}
    if not isinstance(ome, Mapping):
        raise ValueError(f"The existing 'ome' entry is not an object: {ome!r}.")
    ome = dict(ome)
    serialized = _serialize_scene(value)
    if version == NgffVersion.V09dev3.value:
        if ome.get("type") is None:
            raise ValueError(
                "A 0.9.dev3 scene lives in the attributes of an RFC-8 node "
                "document, but the store's root carries none. Write the "
                "collection first with to_collection_zarr()."
            )
        attributes = dict(ome.get("attributes") or {})
        attributes[SCENE_KEY] = serialized
        ome["attributes"] = attributes
    else:
        ome.setdefault("version", ondisk)
        ome[SCENE_KEY] = serialized
    consolidated = has_consolidated_metadata(store_path, 3)
    create_zarrista_group(store_path, {"ome": ome}, 3)
    if consolidated:
        consolidate_metadata(store_path, zarr_format=3)
