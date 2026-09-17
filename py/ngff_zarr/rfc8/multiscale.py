# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Conversion between multiscales entries and RFC-8 multiscale node documents.

At OME-Zarr 0.9.dev3 an image store's root ``ome`` value is a ``multiscale``
node: the datasets become ``singlescale`` child nodes referenced by typed
paths, the coordinate systems move into the node's ``attributes`` and gain
required ids, and the per-level transformations reference the singlescale's
own id and a coordinate-system id in place of the RFC-5 path/name
identifiers. The functions here convert between that document and the
``multiscales`` entry shape of 0.6/0.9.dev1, so the readers, writers and
upgraders keep one in-memory model (:class:`ngff_zarr.v09.zarr_metadata.Metadata`).

Both directions are dict-to-dict: the writer hands over the serialized entry
(``asdict`` + ``None``-stripping already applied), the reader hands back an
entry the 0.9.dev1 machinery parses. Entry keys with no RFC-8 home of their
own (``@type``, the method ``type``/``metadata``) ride in the node's
``attributes`` verbatim, so they round-trip.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

from .model import RFC8_ID_PATTERN

_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9\-_.]")


def _claim_id(candidate: str, used: set[str]) -> str:
    """A unique document id derived from ``candidate``.

    Characters outside the RFC-8 id production are replaced with ``-`` (a
    nested dataset path like ``scale0/image`` becomes ``scale0-image``), an
    empty result falls back to ``n``, and a taken id gains a numeric suffix.
    """
    cleaned = _INVALID_ID_CHARS.sub("-", candidate) or "n"
    claimed = cleaned
    suffix = 2
    while claimed in used:
        claimed = f"{cleaned}-{suffix}"
        suffix += 1
    used.add(claimed)
    return claimed


def _reference_to_ids(identifier: Any, name_to_id: dict[str | None, str]) -> Any:
    """Rewrite a name-based identifier dict to an RFC-8 id reference."""
    if not isinstance(identifier, dict):
        return identifier
    name = identifier.get("name")
    if identifier.get("id") is not None:
        reference = {"id": identifier["id"]}
    elif name in name_to_id:
        reference = {"id": name_to_id[name]}
    else:
        return identifier
    if identifier.get("path") is not None:
        reference["path"] = identifier["path"]
    return reference


def multiscale_ome_dict(entry: dict, version: str) -> dict[str, Any]:
    """Build the 0.9.dev3 ``ome`` value for a serialized multiscales entry.

    ``entry`` is consumed as the writer hands it over: ``omero`` is hoisted
    into the node's attributes (the RFC-8 root has no ``ome.omero`` slot),
    every coordinate system gains an id (its own, else one derived from its
    name), each dataset becomes a ``singlescale`` child node whose
    transformations reference its id and a coordinate-system id, and the
    remaining entry keys ride in the attributes verbatim.
    """
    entry = dict(entry)
    omero = entry.pop("omero", None)
    name = entry.pop("name", None) or "image"
    systems = [dict(system) for system in entry.pop("coordinateSystems", [])]
    datasets = entry.pop("datasets", [])
    top_level = entry.pop("coordinateTransformations", None)

    used_ids: set[str] = {
        system["id"] for system in systems if system.get("id") is not None
    }
    name_to_id: dict[str | None, str] = {}
    for system in systems:
        if system.get("id") is None:
            system["id"] = _claim_id(system.get("name") or "system", used_ids)
        name_to_id.setdefault(system.get("name"), system["id"])

    nodes = []
    for dataset in datasets:
        path = dataset["path"]
        singlescale_id = _claim_id(path, used_ids)
        transformations = []
        for transform in dataset.get("coordinateTransformations", []):
            transform = dict(transform)
            transform["input"] = {"id": singlescale_id}
            transform["output"] = _reference_to_ids(transform.get("output"), name_to_id)
            transformations.append(transform)
        nodes.append(
            {
                "type": "singlescale",
                "id": singlescale_id,
                "name": path,
                "path": {"type": "zarr", "path": f"./{path}"},
                "attributes": {"coordinateTransformations": transformations},
            }
        )

    attributes: dict[str, Any] = {"coordinateSystems": systems}
    if top_level:
        attributes["coordinateTransformations"] = [
            {
                **dict(transform),
                "input": _reference_to_ids(transform.get("input"), name_to_id),
                "output": _reference_to_ids(transform.get("output"), name_to_id),
            }
            for transform in top_level
        ]
    if omero is not None:
        attributes["omero"] = omero
    attributes.update(entry)

    ome: dict[str, Any] = {"version": version, "type": "multiscale"}
    if RFC8_ID_PATTERN.match(name) and name not in used_ids:
        ome["id"] = name
    ome["name"] = name
    ome["nodes"] = nodes
    ome["attributes"] = attributes
    return ome


def _dataset_path_from(node: dict) -> str:
    """The dataset path a singlescale node's zarr path designates."""
    path = node.get("path", {}).get("path", "")
    normalized = posixpath.normpath(path)
    return normalized[2:] if normalized.startswith("./") else normalized


def _reference_to_names(
    identifier: Any, id_to_name: dict[str, str], dataset_path: str | None
) -> Any:
    """Rewrite an id reference to the name/path shape the v0.6 reader keys on.

    The RFC-8 ``id`` is kept alongside, so the parsed identifiers carry it.
    """
    if not isinstance(identifier, dict):
        return identifier
    rewritten = dict(identifier)
    value = rewritten.get("id")
    if isinstance(value, str) and value in id_to_name:
        rewritten.setdefault("name", id_to_name[value])
    elif dataset_path is not None and rewritten.get("path") is None:
        # The singlescale's own id: the v0.6 shape records the dataset path.
        rewritten["path"] = dataset_path
    return rewritten


def multiscales_entry_from_ome(
    ome: dict, singlescale_transforms: dict[str, list] | None = None
) -> tuple[dict[str, Any], Any]:
    """Rebuild the 0.9.dev1-shaped multiscales entry of a multiscale node.

    Returns ``(entry, omero)``. Coordinate systems keep their ids and gain a
    name when RFC-8 omitted it (the in-memory model resolves systems by
    name), and each singlescale child becomes a dataset whose
    transformations carry both the RFC-8 ids and the name/path fields the
    0.9.dev1 reader keys on. ``singlescale_transforms`` supplies the
    transformations of children that carry none inline (the RFC-8
    distributed layout, where each level's own ``zarr.json`` holds them),
    keyed by dataset path.
    """
    attributes = dict(ome.get("attributes") or {})
    systems = [dict(system) for system in attributes.pop("coordinateSystems", [])]
    id_to_name: dict[str, str] = {}
    for system in systems:
        system_id = system.get("id")
        if not system.get("name") and system_id is not None:
            system["name"] = system_id
        if system_id is not None:
            id_to_name[system_id] = system["name"]

    datasets = []
    for node in ome.get("nodes") or []:
        path = _dataset_path_from(node)
        node_attributes = node.get("attributes") or {}
        transformations = node_attributes.get("coordinateTransformations")
        if not transformations and singlescale_transforms:
            transformations = singlescale_transforms.get(path)
        rewritten = [
            {
                **dict(transform),
                "input": _reference_to_names(transform.get("input"), id_to_name, path),
                "output": _reference_to_names(
                    transform.get("output"), id_to_name, None
                ),
            }
            for transform in transformations or []
        ]
        dataset: dict[str, Any] = {"path": path}
        if rewritten:
            dataset["coordinateTransformations"] = rewritten
        datasets.append(dataset)

    entry: dict[str, Any] = {
        "name": ome.get("name", "image"),
        "coordinateSystems": systems,
        "datasets": datasets,
    }
    top_level = attributes.pop("coordinateTransformations", None)
    if top_level:
        entry["coordinateTransformations"] = [
            {
                **dict(transform),
                "input": _reference_to_names(transform.get("input"), id_to_name, None),
                "output": _reference_to_names(
                    transform.get("output"), id_to_name, None
                ),
            }
            for transform in top_level
        ]
    omero = attributes.pop("omero", None)
    entry.update(attributes)
    return entry, omero
