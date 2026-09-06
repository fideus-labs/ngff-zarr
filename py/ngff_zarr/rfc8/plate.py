# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Typed views over the RFC-8 HCS attributes: plate, well and acquisition.

RFC-8 reworks the high-content-screening metadata as collection attributes:
a plate is a collection with a ``plate`` attribute declaring acquisitions,
columns and rows as id-carrying entries; a well is a collection with a
``well`` attribute referencing one column and one row by id; and an
``acquisition`` attribute (a reference to one of the plate's acquisitions)
sits on individual multiscale nodes (the "wide" layout) or on
sub-collections grouping one acquisition's images (the "tall" layout). The
numeric ids and name-keyed paths of the 0.4/0.5 HCS spec become string-id
references, consistent with the rest of RFC-8.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model import Node, OmePath, Reference

#: The RFC-8 attribute keys this module owns.
PLATE_KEY = "plate"
WELL_KEY = "well"
ACQUISITION_KEY = "acquisition"


@dataclass(frozen=True)
class PlateEntry:
    """One acquisition, column or row of a plate: an id plus a label.

    The RFC's Acquisition, Column and Row interfaces share this shape; ids
    match ``[a-zA-Z0-9-_.]+`` and are unique within the JSON document.
    """

    id: str
    name: str | None = None


@dataclass
class PlateAttribute:
    """The ``plate`` attribute of a plate-level collection."""

    columns: list[PlateEntry]
    rows: list[PlateEntry]
    acquisitions: list[PlateEntry] | None = None


@dataclass
class WellAttribute:
    """The ``well`` attribute of a well-level collection.

    ``column`` and ``row`` reference entries declared in the ``plate``
    attribute of the enclosing plate collection.
    """

    column: Reference
    row: Reference


def _entry_from(value: Any) -> PlateEntry:
    if not isinstance(value, Mapping):
        raise ValueError(f"A plate entry must be an object; got {value!r}.")
    entry_id = value.get("id")
    if not isinstance(entry_id, str):
        raise ValueError(f"A plate entry must declare a string 'id'; got {value!r}.")
    name = value.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"A plate entry 'name' must be a string; got {name!r}.")
    return PlateEntry(id=entry_id, name=name)


def _entry_to_dict(entry: PlateEntry) -> dict[str, Any]:
    serialized: dict[str, Any] = {"id": entry.id}
    if entry.name is not None:
        serialized["name"] = entry.name
    return serialized


def _reference_from(value: Any) -> Reference:
    if not isinstance(value, Mapping):
        raise ValueError(f"A well reference must be an object; got {value!r}.")
    reference_id = value.get("id")
    if not isinstance(reference_id, str):
        raise ValueError(f"A well reference must declare a string 'id'; got {value!r}.")
    path = value.get("path")
    if path is not None and (
        not isinstance(path, Mapping)
        or not isinstance(path.get("type"), str)
        or not isinstance(path.get("path"), str)
    ):
        raise ValueError(
            f"A reference 'path' must be an object with string 'type' and "
            f"'path'; got {path!r}."
        )
    return Reference(
        id=reference_id,
        path=OmePath(path["type"], path["path"]) if path is not None else None,
    )


def _reference_to_dict(reference: Reference) -> dict[str, Any]:
    serialized: dict[str, Any] = {"id": reference.id}
    if reference.path is not None:
        serialized["path"] = {
            "type": reference.path.type,
            "path": reference.path.path,
        }
    return serialized


def plate(node: Node) -> PlateAttribute | None:
    """The parsed ``plate`` attribute of ``node``, or ``None``."""
    raw = (node.attributes or {}).get(PLATE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"A 'plate' attribute must be an object; got {raw!r}.")
    for field in ("columns", "rows", "acquisitions"):
        value = raw.get(field)
        if value is not None and not isinstance(value, list):
            raise ValueError(f"A plate {field!r} must be an array; got {value!r}.")
    acquisitions = None
    if raw.get("acquisitions") is not None:
        acquisitions = [_entry_from(entry) for entry in raw["acquisitions"]]
    return PlateAttribute(
        columns=[_entry_from(entry) for entry in raw.get("columns") or []],
        rows=[_entry_from(entry) for entry in raw.get("rows") or []],
        acquisitions=acquisitions,
    )


def set_plate(node: Node, value: PlateAttribute | None) -> None:
    """Write ``value`` into ``node.attributes``, replacing the key."""
    if node.attributes is None:
        node.attributes = {}
    if value is None:
        node.attributes.pop(PLATE_KEY, None)
        return
    serialized: dict[str, Any] = {}
    if value.acquisitions is not None:
        serialized["acquisitions"] = [
            _entry_to_dict(entry) for entry in value.acquisitions
        ]
    serialized["columns"] = [_entry_to_dict(entry) for entry in value.columns]
    serialized["rows"] = [_entry_to_dict(entry) for entry in value.rows]
    node.attributes[PLATE_KEY] = serialized


def well(node: Node) -> WellAttribute | None:
    """The parsed ``well`` attribute of ``node``, or ``None``."""
    raw = (node.attributes or {}).get(WELL_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"A 'well' attribute must be an object; got {raw!r}.")
    return WellAttribute(
        column=_reference_from(raw.get("column")),
        row=_reference_from(raw.get("row")),
    )


def set_well(node: Node, value: WellAttribute | None) -> None:
    """Write ``value`` into ``node.attributes``, replacing the key."""
    if node.attributes is None:
        node.attributes = {}
    if value is None:
        node.attributes.pop(WELL_KEY, None)
        return
    node.attributes[WELL_KEY] = {
        "column": _reference_to_dict(value.column),
        "row": _reference_to_dict(value.row),
    }


def acquisition(node: Node) -> Reference | None:
    """The parsed ``acquisition`` attribute of ``node``, or ``None``."""
    raw = (node.attributes or {}).get(ACQUISITION_KEY)
    if raw is None:
        return None
    return _reference_from(raw)


def set_acquisition(node: Node, value: Reference | None) -> None:
    """Write ``value`` into ``node.attributes``, replacing the key."""
    if node.attributes is None:
        node.attributes = {}
    if value is None:
        node.attributes.pop(ACQUISITION_KEY, None)
        return
    node.attributes[ACQUISITION_KEY] = _reference_to_dict(value)


def plate_collection_from_hcs(hcs_plate: Any) -> Node:
    """Build an RFC-8 plate collection from an :class:`~ngff_zarr.HCSPlate`.

    Produces the "wide" layout of the RFC: one child collection per well,
    whose multiscale nodes reference the image groups by their paths within
    the plate store and carry an ``acquisition`` reference when the source
    declares one. RFC-8 ids share one namespace per document and 0.4/0.5
    row, column and acquisition labels do not (a column ``"1"`` and an
    acquisition ``1`` are routine), so the ids are prefixed by kind --
    ``row.A``, ``col.1``, ``acq.0`` -- and the labels stay on ``name``.
    Written beside the existing plate metadata (with
    :func:`ngff_zarr.to_collection_zarr` on the plate root), the collection
    is the RFC-8 view of the same store.
    """
    metadata = hcs_plate.metadata
    acquisitions = None
    if metadata.acquisitions:
        acquisitions = [
            PlateEntry(id=f"acq.{entry.id}", name=entry.name or str(entry.id))
            for entry in metadata.acquisitions
        ]
    plate_attribute = PlateAttribute(
        columns=[
            PlateEntry(id=f"col.{column.name}", name=column.name)
            for column in metadata.columns
        ],
        rows=[PlateEntry(id=f"row.{row.name}", name=row.name) for row in metadata.rows],
        acquisitions=acquisitions,
    )

    from .model import Collection

    wells: list[Node] = []
    for well_meta in metadata.wells or []:
        row_name, column_name = well_meta.path.split("/")
        well_node = Collection(f"well {well_meta.path}", nodes=[])
        set_well(
            well_node,
            WellAttribute(
                column=Reference(f"col.{column_name}"),
                row=Reference(f"row.{row_name}"),
            ),
        )
        hcs_well = hcs_plate.get_well(row_name, column_name)
        for image in (hcs_well.metadata.images if hcs_well else []) or []:
            image_node = Node(
                type="multiscale",
                name=f"{well_meta.path}/{image.path}",
                path=OmePath("zarr", f"./{well_meta.path}/{image.path}"),
            )
            if image.acquisition is not None:
                set_acquisition(image_node, Reference(f"acq.{image.acquisition}"))
            well_node.nodes.append(image_node)
        wells.append(well_node)

    root = Collection(metadata.name or "plate", nodes=wells)
    set_plate(root, plate_attribute)
    return root
