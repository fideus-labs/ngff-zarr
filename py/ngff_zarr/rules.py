# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The rules of the OME-Zarr specification that ngff-zarr checks beyond the
official JSON Schemas.

``rules.json`` lists each rule once: its code, the versions whose text states
it, where the specification states it, and documents that break it, which the
tests of both ports check. The TypeScript port reads the same file. The checks
run on a document the JSON Schema accepted, so they rely on the shapes the
schema enforces.
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


def _records(value: object) -> list[dict]:
    """The JSON objects in a list, and nothing for any other value: the schema
    leaves parts of a document open, which may hold anything."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _length(value: object) -> int | None:
    """The length of a list, or None for any other value."""
    return len(value) if isinstance(value, list) else None


def _systems(document: dict) -> dict[str, dict]:
    return {
        system["name"]: system
        for system in _records(document.get("coordinateSystems"))
        if isinstance(system.get("name"), str)
    }


def _axis_count(reference, systems: dict[str, dict]) -> int | None:
    """The axis count of the coordinate system a reference names in the same
    document, or None when it points elsewhere."""
    if isinstance(reference, dict) and not reference.get("path"):
        system = systems.get(reference.get("name"))
        if system is not None:
            return _length(system.get("axes"))
    return None


def _transformation_findings(ome: dict) -> Iterator[Finding]:
    """Check each coordinate transformation of a document against the axis
    counts of the coordinate systems it maps between."""
    # Transformations sit beside the systems they name: at the root of a
    # standalone transformation, in a scene and in each multiscales entry.
    documents = [("", ome)]
    if isinstance(ome.get("scene"), dict):
        documents.append(("scene.", ome["scene"]))
    documents += [
        (f"multiscales[{m}].", entry)
        for m, entry in enumerate(_records(ome.get("multiscales")))
    ]
    for prefix, document in documents:
        systems = _systems(document)
        transforms = _records(document.get("coordinateTransformations"))
        for i, transform in enumerate(transforms):
            yield from _dimension_findings(
                f"{prefix}coordinateTransformations[{i}]",
                transform,
                _axis_count(transform.get("input"), systems),
                _axis_count(transform.get("output"), systems),
            )
        for d, dataset in enumerate(_records(document.get("datasets"))):
            transforms = _records(dataset.get("coordinateTransformations"))
            for i, transform in enumerate(transforms):
                # A dataset's array has as many dimensions as the system it
                # maps to has axes.
                count = _axis_count(transform.get("output"), systems)
                yield from _dimension_findings(
                    f"{prefix}datasets[{d}].coordinateTransformations[{i}]",
                    transform,
                    count,
                    count,
                )


#: The transformation types with as many output axes as input axes.
_KEEPS_COUNT = ("identity", "scale", "translation", "rotation", "mapAxis", "bijection")


def _keeps_count(step: dict) -> bool:
    """Whether a step of a sequence has as many output axes as input axes."""
    if step.get("type") == "sequence":
        return all(
            _keeps_count(inner) for inner in _records(step.get("transformations"))
        )
    return step.get("type") in _KEEPS_COUNT


def _axes_after(step: dict, count: int | None) -> int | None:
    """The axis count after a step of a sequence, when it follows from the step."""
    kind = step.get("type")
    if kind in _KEEPS_COUNT:
        return count
    if kind == "affine" and isinstance(step.get("affine"), list):
        return len(step["affine"])
    if kind == "projectAxis" and count is not None:
        dropped = _length(step.get("droppedInputs", []))
        created = _length(step.get("createdOutputs", []))
        if dropped is not None and created is not None:
            return count - dropped + created
    if kind == "sequence":
        for inner in _records(step.get("transformations")):
            count = _axes_after(inner, count)
        return count
    if kind == "byDimension":
        # Every output axis belongs to exactly one item.
        items = _records(step.get("transformations"))
        outputs = [_length(item.get("outputAxes")) for item in items]
        if None not in outputs:
            return sum(outputs)
    return None


def _orthonormal(matrix: list) -> bool:
    try:
        rotation = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        np.allclose(
            rotation @ rotation.T,
            np.eye(len(rotation)),
            rtol=0,
            atol=ROTATION_TOLERANCE,
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
            if len(matrix) != count or any(
                not isinstance(row, list) or len(row) != count for row in matrix
            ):
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
        if n_in is not None and any(
            not isinstance(row, list) or len(row) != n_in + 1 for row in matrix
        ):
            needs.append(f"rows of {n_in + 1} values for {n_in} input axes")
        if needs:
            yield (
                "NGFF-CT-025",
                f"{where}: the affine matrix needs " + " and ".join(needs),
            )
    elif kind == "sequence":
        steps = _records(transform.get("transformations"))
        count = n_in
        if count is None and all(_keeps_count(step) for step in steps):
            # Steps that keep their axis count give the input the output's.
            count = n_out
        for index, step in enumerate(steps):
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
        for index, item in enumerate(_records(transform.get("transformations"))):
            if isinstance(item.get("transformation"), dict):
                yield from _dimension_findings(
                    f"{where}.transformations[{index}].transformation",
                    item["transformation"],
                    _length(item.get("inputAxes")),
                    _length(item.get("outputAxes")),
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
