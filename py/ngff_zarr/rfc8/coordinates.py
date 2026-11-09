# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Typed views over the RFC-8 coordinate attribute keys of a node.

RFC-8 defines ``coordinateSystems`` and ``coordinateTransformations`` as
attribute keys: a multiscale node carries its systems, a singlescale node its
transformations, and a collection may carry either for its members. The
accessors here parse those keys into the RFC-5 model classes (a coordinate
system's ``id`` and an identifier's ``id`` were added for RFC-8) and write
them back, leaving every other attribute key untouched.

Cross-document references (an ``id`` paired with a typed ``path``) are kept
as parsed on the identifier; dereferencing them is the concern of
:meth:`ngff_zarr.rfc8.NgffCollection.load` and later RFC-8 stages.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

from ..to_ngff_zarr import _remove_none_values
from ..v06.zarr_metadata import Metadata as _MetadataV06
from ..v09.zarr_metadata import (
    Axis,
    CoordinateSystem,
    Transform,
    _filter_axis_dict,
)
from .model import Node


def coordinate_systems(node: Node) -> list[CoordinateSystem]:
    """The coordinate systems declared in ``node.attributes``.

    RFC-8 makes a system's ``name`` optional (the required ``id`` carries the
    identity), so ``name`` may come back ``None``.
    """
    raw = (node.attributes or {}).get("coordinateSystems", [])
    systems = []
    for entry in raw:
        axes = [Axis(**_filter_axis_dict(axis)) for axis in entry["axes"]]
        systems.append(
            CoordinateSystem(name=entry.get("name"), axes=axes, id=entry.get("id"))
        )
    return systems


def set_coordinate_systems(node: Node, systems: Sequence[CoordinateSystem]) -> None:
    """Write ``systems`` into ``node.attributes``, replacing the key."""
    serialized = [
        _remove_none_values(
            {
                "id": system.id,
                "name": system.name,
                "axes": [asdict(axis) for axis in system.axes],
            }
        )
        for system in systems
    ]
    if node.attributes is None:
        node.attributes = {}
    node.attributes["coordinateSystems"] = serialized


def coordinate_transformations(
    node: Node,
    systems: Sequence[CoordinateSystem] | None = None,
    version: str | None = "0.9.dev3",
) -> list[Transform]:
    """The coordinate transformations declared in ``node.attributes``.

    ``systems`` are the coordinate systems the transformations' references
    resolve against; they default to the ones declared on ``node`` itself.
    Identifiers keep the RFC-8 ``id`` alongside any RFC-5 ``name``/``path``.
    """
    raw = (node.attributes or {}).get("coordinateTransformations", [])
    if not raw:
        return []
    resolved = list(systems) if systems is not None else coordinate_systems(node)
    return _MetadataV06._parse_transforms(raw, resolved, version)


def set_coordinate_transformations(
    node: Node, transformations: Sequence[Transform]
) -> None:
    """Write ``transformations`` into ``node.attributes``, replacing the key."""
    serialized = [
        _remove_none_values(transform.to_dict()) for transform in transformations
    ]
    if node.attributes is None:
        node.attributes = {}
    node.attributes["coordinateTransformations"] = serialized
