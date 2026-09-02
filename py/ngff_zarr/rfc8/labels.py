# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Typed views over the RFC-8 ``labels`` attribute of a multiscale node.

A multiscale node with a ``labels`` attribute is a label map:
``labelAttributes`` describes individual label values (required
``labelValue``, optional RGBA ``color``) and ``source`` references the
annotated multiscale images. Because ``source`` is an array of references,
a label map can annotate several images, an image can carry several label
maps, and the two can live anywhere relative to each other.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .model import Node, OmePath, Reference

#: The RFC-8 attribute key this module owns.
LABELS_KEY = "labels"


@dataclass
class LabelAttributes:
    """Attributes of one label value: the RFC-8 ``Label Attributes`` object.

    ``color`` is the RGBA color as four integers between 0 and 255. ``extra``
    carries the additional keys the RFC permits (prefixed extension keys
    included) and is serialized back verbatim.
    """

    labelValue: float
    color: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Labels:
    """The RFC-8 ``labels`` attribute of a multiscale node.

    Both fields are optional and an empty object is meaningful: it denotes
    the node a label map without saying more.
    """

    labelAttributes: list[LabelAttributes] | None = None
    source: list[Reference] | None = None


def is_label_map(node: Node) -> bool:
    """Whether ``node`` declares the ``labels`` attribute."""
    return isinstance(node.attributes, Mapping) and LABELS_KEY in node.attributes


def _reference_from(value: Any) -> Reference:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"A labels source entry must be a reference object; got {value!r}."
        )
    path = value.get("path")
    return Reference(
        id=value.get("id"),
        path=OmePath(path["type"], path["path"]) if isinstance(path, Mapping) else None,
    )


def labels(node: Node) -> Labels | None:
    """The parsed ``labels`` attribute of ``node``, or ``None``.

    An empty ``labels`` object parses to a :class:`Labels` with both fields
    ``None``; use :func:`is_label_map` to tell it apart from an absent
    attribute.
    """
    raw = (node.attributes or {}).get(LABELS_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"A 'labels' attribute must be an object; got {raw!r}.")
    label_attributes = None
    if raw.get("labelAttributes") is not None:
        label_attributes = []
        for entry in raw["labelAttributes"]:
            entry = dict(entry)
            label_attributes.append(
                LabelAttributes(
                    labelValue=entry.pop("labelValue", None),
                    color=entry.pop("color", None),
                    extra=copy.deepcopy(entry),
                )
            )
    source = None
    if raw.get("source") is not None:
        source = [_reference_from(entry) for entry in raw["source"]]
    return Labels(labelAttributes=label_attributes, source=source)


def set_labels(node: Node, value: Labels | None) -> None:
    """Write ``value`` into ``node.attributes``, replacing the key.

    ``Labels()`` writes the empty object that denotes a label map;
    ``None`` removes the attribute.
    """
    if node.attributes is None:
        node.attributes = {}
    if value is None:
        node.attributes.pop(LABELS_KEY, None)
        return
    serialized: dict[str, Any] = {}
    if value.labelAttributes is not None:
        serialized["labelAttributes"] = [
            {
                "labelValue": entry.labelValue,
                **({"color": list(entry.color)} if entry.color is not None else {}),
                **copy.deepcopy(entry.extra),
            }
            for entry in value.labelAttributes
        ]
    if value.source is not None:
        serialized["source"] = [
            {
                "id": reference.id,
                **(
                    {
                        "path": {
                            "type": reference.path.type,
                            "path": reference.path.path,
                        }
                    }
                    if reference.path is not None
                    else {}
                ),
            }
            for reference in value.source
        ]
    node.attributes[LABELS_KEY] = serialized
