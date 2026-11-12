# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The rules of the OME-Zarr specification that ngff-zarr checks beyond the
official JSON Schemas.

``rules.json`` lists each rule once: its code, the versions whose text states
it, whether it is a MUST or a SHOULD, where the specification states it, and a
minimal document that breaks it. The TypeScript port reads the same file. The
checks run on a document the JSON Schema accepted, so they rely on the shapes
the schema enforces.
"""

import json
from collections.abc import Iterator

import numpy as np
from importlib_resources import files

RULES = {
    rule["code"]: rule
    for rule in json.loads(files("ngff_zarr").joinpath("rules.json").read_text())
}

#: How far a stored rotation may be from orthonormal, per matrix entry.
ROTATION_TOLERANCE = 1e-5

#: A broken rule: its code and what breaks it.
Finding = tuple[str, str]


def rule_problems(attributes: dict, version: str) -> list[Finding]:
    """The rules that the OME-Zarr Metadata in ``attributes`` breaks at
    ``version``."""
    ome = attributes.get("ome", attributes)
    return [
        (code, message)
        for code, message in _transformation_findings(ome)
        if version in RULES[code]["versions"]
    ]


def _systems(document: dict) -> dict[str, dict]:
    return {system["name"]: system for system in document.get("coordinateSystems", [])}


def _axis_count(reference, systems: dict[str, dict]) -> int | None:
    """The axis count of the coordinate system a reference names in the same
    document, or None when it points elsewhere."""
    if isinstance(reference, dict) and not reference.get("path"):
        system = systems.get(reference.get("name"))
        if system is not None:
            return len(system["axes"])
    return None


def _transformation_findings(ome: dict) -> Iterator[Finding]:
    """Check each coordinate transformation of a document against the axis
    counts of the coordinate systems it maps between."""
    scene = ome.get("scene")
    if isinstance(scene, dict):
        systems = _systems(scene)
        for index, transform in enumerate(scene.get("coordinateTransformations", [])):
            yield from _dimension_findings(
                f"scene.coordinateTransformations[{index}]",
                transform,
                _axis_count(transform.get("input"), systems),
                _axis_count(transform.get("output"), systems),
            )
    for m, entry in enumerate(ome.get("multiscales", [])):
        systems = _systems(entry)
        for index, transform in enumerate(entry.get("coordinateTransformations", [])):
            yield from _dimension_findings(
                f"multiscales[{m}].coordinateTransformations[{index}]",
                transform,
                _axis_count(transform.get("input"), systems),
                _axis_count(transform.get("output"), systems),
            )
        for d, dataset in enumerate(entry.get("datasets", [])):
            for index, transform in enumerate(
                dataset.get("coordinateTransformations", [])
            ):
                # A dataset's array has as many dimensions as the system it
                # maps to has axes.
                count = _axis_count(transform.get("output"), systems)
                yield from _dimension_findings(
                    f"multiscales[{m}].datasets[{d}].coordinateTransformations[{index}]",
                    transform,
                    count,
                    count,
                )


def _axes_after(step: dict, count: int | None) -> int | None:
    """The axis count after a step of a sequence, when it follows from the step."""
    kind = step.get("type")
    if kind in ("identity", "scale", "translation", "rotation", "mapAxis"):
        return count
    if kind == "affine" and isinstance(step.get("affine"), list):
        return len(step["affine"])
    return None


def _orthonormal(matrix: list) -> bool:
    rotation = np.asarray(matrix, dtype=float)
    return bool(
        np.allclose(
            rotation @ rotation.T, np.eye(len(rotation)), atol=ROTATION_TOLERANCE
        )
        and abs(np.linalg.det(rotation) - 1) <= ROTATION_TOLERANCE
    )


def _dimension_findings(
    where: str, transform: dict, n_in: int | None, n_out: int | None
) -> Iterator[Finding]:
    """Check a transformation's parameters against ``n_in`` input and ``n_out``
    output axes, either of which may be unknown."""
    kind = transform.get("type")
    known = n_in is not None and n_out is not None
    if kind == "identity" and known and n_in != n_out:
        yield "NGFF-CT-015", f"{where}: an identity maps {n_in} axes to {n_out}"
    elif kind in ("scale", "translation") and isinstance(transform.get(kind), list):
        for count in sorted({n_in, n_out} - {None}):
            if len(transform[kind]) != count:
                given = len(transform[kind])
                message = f"{where}: {kind} gives {given} values for {count} axes"
                yield "NGFF-CT-015", message
    elif kind == "rotation":
        if known and n_in != n_out:
            yield "NGFF-CT-026", f"{where}: a rotation maps {n_in} axes to {n_out}"
        matrix = transform.get("rotation")
        count = n_in if n_in is not None else n_out
        if isinstance(matrix, list):
            if count is None:
                count = len(matrix)
            if len(matrix) != count or any(len(row) != count for row in matrix):
                message = f"{where}: a rotation of {count} axes needs a {count} x {count} matrix"
                yield "NGFF-CT-030", message
            elif not _orthonormal(matrix):
                message = f"{where}: the rotation matrix is not orthonormal with determinant one"
                yield "NGFF-CT-027", message
    elif kind == "affine" and isinstance(transform.get("affine"), list):
        matrix = transform["affine"]
        needs = []
        if n_out is not None and len(matrix) != n_out:
            needs.append(f"{n_out} rows for {n_out} output axes")
        if n_in is not None and any(len(row) != n_in + 1 for row in matrix):
            needs.append(f"rows of {n_in + 1} values for {n_in} input axes")
        if needs:
            yield (
                "NGFF-CT-025",
                f"{where}: the affine matrix needs " + " and ".join(needs),
            )
    elif kind == "sequence":
        count = n_in
        for index, step in enumerate(transform.get("transformations", [])):
            after = _axes_after(step, count)
            yield from _dimension_findings(
                f"{where}.transformations[{index}]", step, count, after
            )
            count = after
        if count is not None and n_out is not None and count != n_out:
            yield (
                "NGFF-CT-015",
                f"{where}: the sequence ends with {count} axes, its output has {n_out}",
            )
    elif kind == "byDimension":
        # Each item's transformation maps as many axes as the item lists.
        for index, item in enumerate(transform.get("transformations", [])):
            yield from _dimension_findings(
                f"{where}.transformations[{index}].transformation",
                item["transformation"],
                len(item["inputAxes"]),
                len(item["outputAxes"]),
            )
    elif kind == "bijection":
        if known and n_in != n_out:
            yield "NGFF-CT-040", f"{where}: a bijection maps {n_in} axes to {n_out}"
        for direction, counts in (
            ("forward", (n_in, n_out)),
            ("inverse", (n_out, n_in)),
        ):
            if isinstance(transform.get(direction), dict):
                yield from _dimension_findings(
                    f"{where}.{direction}", transform[direction], *counts
                )
