# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC 4 conformance output for the ``ngff-zarr`` CLI.

``ngff-zarr conformance <input>`` prints one JSON verdict per input -- the shape
a tool-agnostic conformance driver feeds against its manifest to check that an
implementation follows RFC 4:

    {"input": ..., "format": "ome-zarr", "rfc4_valid": bool,
     "axes": {name: value}, "violations": [code, ...], "warnings": [code, ...]}

It reads the *raw* axes, so malformed metadata is still classifiable -- the
high-level reader would reject it before it could be reported. Findings are
reported with the conformance contract's code names: RFC 4 defines no error codes,
and the package's own :class:`~ngff_zarr.structural_validation.SpecRule`
identifiers are a different vocabulary, so the classification here is deliberately
expressed in the contract's terms while applying the same RFC 4 rules as
:func:`~ngff_zarr.rfc4_validation.validate_rfc4_orientation`.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .parse_metadata import _raw_axes
from .rfc4_validation import _ANATOMICAL_AXIS_OF

# Metadata files that may hold the multiscales, most specific first.
_METADATA_NAMES = ("zarr.json", ".zattrs")


class _UnreadableInput(Exception):
    """The input is not a readable OME-Zarr carrying an axes list.

    Holds the conformance ``code`` to surface as the verdict violation, so an
    unreadable or non-OME-Zarr input is reported as invalid rather than silently
    treated as an empty (and therefore "valid") axes list.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _axes_from_metadata(meta: Any) -> list[dict[str, Any]]:
    """Pull the axes out of a v2 ``.zattrs`` or v3 ``zarr.json`` blob.

    The axes are located by :func:`~ngff_zarr.parse_metadata._raw_axes`, so a
    v0.6 document, whose axes live in the intrinsic coordinate system rather
    than in a flat ``axes`` key, yields its axes instead of being classified
    as not-OME-Zarr.
    """
    if not isinstance(meta, dict):
        raise _UnreadableInput("input-not-ome-zarr", "metadata is not a JSON object")
    attributes = meta.get("attributes", meta)
    ome = attributes.get("ome", attributes) if isinstance(attributes, dict) else {}
    try:
        entry = ome["multiscales"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise _UnreadableInput(
            "input-not-ome-zarr", "metadata has no multiscales[0]"
        ) from exc
    axes = _raw_axes(entry)
    if not axes:
        raise _UnreadableInput(
            "input-not-ome-zarr",
            "multiscales[0] carries no axes, in a flat 'axes' list or in "
            "'coordinateSystems'",
        )
    return axes


def _load_json(raw: bytes | str, source: str) -> Any:
    """Parse metadata JSON, classifying a decode failure as an unreadable input."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _UnreadableInput(
            "input-unreadable", f"{source} is not valid JSON"
        ) from exc


def _read_axes(path: str) -> list[dict[str, Any]]:
    """Read the raw axis dicts from an OME-Zarr directory or ``.zip``.

    Raises :class:`_UnreadableInput` when no readable OME-Zarr metadata carrying
    an axes list is found, so the caller can emit an invalid verdict instead of
    an empty axes list.
    """
    source = Path(path)
    if source.suffix == ".zip":
        try:
            archive = zipfile.ZipFile(source)
        except (FileNotFoundError, OSError, zipfile.BadZipFile) as exc:
            raise _UnreadableInput(
                "input-unreadable", f"{path} is not a readable zip archive"
            ) from exc
        with archive:
            for name in sorted(archive.namelist(), key=lambda n: n.count("/")):
                if name.endswith(_METADATA_NAMES):
                    return _axes_from_metadata(_load_json(archive.read(name), name))
        raise _UnreadableInput(
            "input-not-ome-zarr", f"{path} has no zarr.json or .zattrs"
        )
    for name in _METADATA_NAMES:
        candidate = source / name
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as metadata_file:
                return _axes_from_metadata(_load_json(metadata_file.read(), name))
    raise _UnreadableInput("input-not-ome-zarr", f"{path} has no zarr.json or .zattrs")


def _classify(axes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Classify RFC 4 findings using the conformance contract's code vocabulary.

    RFC 4 defines no error codes, so the names below belong to the conformance
    contract rather than to the specification. The rules applied are the RFC's:

    - ``orientation`` MUST only be used on spatial axes;
    - an ``orientation`` object MUST carry a ``type`` (only ``"anatomical"`` is
      defined) and a ``value`` drawn from the controlled vocabulary;
    - a set of axes MUST only carry one direction of each antonym pair.

    Orientation is optional per spatial axis, and an absent orientation is
    equivalent to an explicit ``null`` -- neither is a violation -- but a writer
    SHOULD omit the field rather than serialize ``null``, which is reported as a
    warning. A code may repeat, once per offending axis.
    """
    violations: list[str] = []
    warnings: list[str] = []
    seen_anatomical_axis: dict[str, str] = {}

    for axis in axes:
        if not isinstance(axis, dict):
            continue
        orientation = axis.get("orientation")

        # An explicit null is equivalent to an absent field (undefined), but
        # RFC 4 says writers SHOULD omit it instead.
        if "orientation" in axis and orientation is None:
            warnings.append("null-orientation")
            continue
        if not isinstance(orientation, dict) or not orientation:
            continue

        if axis.get("type") != "space":
            violations.append("orientation-on-non-space")
            continue
        if orientation.get("type") != "anatomical":
            violations.append("bad-type")
            continue
        value = orientation.get("value")
        if value is None:
            violations.append("missing-value")
            continue
        if value not in _ANATOMICAL_AXIS_OF:
            violations.append("bad-value")
            continue

        axis_key = _ANATOMICAL_AXIS_OF[value]
        if axis_key in seen_anatomical_axis:
            violations.append("duplicate-anatomical-axis")
        else:
            seen_anatomical_axis[axis_key] = str(axis.get("name"))

    return violations, warnings


def conformance_report(path: str) -> dict[str, Any]:
    """Build the canonical conformance verdict for one OME-Zarr input.

    An unreadable or non-OME-Zarr input yields an invalid verdict carrying the
    read-failure code, never ``rfc4_valid: true`` with empty axes.
    """
    try:
        axes = _read_axes(path)
    except _UnreadableInput as exc:
        return {
            "input": path,
            "format": "ome-zarr",
            "rfc4_valid": False,
            "axes": {},
            "violations": [exc.code],
            "warnings": [],
        }

    # Built defensively from the raw axes: a malformed axis (missing name,
    # non-dict orientation) is skipped in this display map rather than crashing
    # the verdict; _classify below still reports any rule it violates.
    orientations = {
        axis["name"]: axis["orientation"]["value"]
        for axis in axes
        if isinstance(axis, dict)
        and isinstance(axis.get("name"), str)
        and isinstance(axis.get("orientation"), dict)
        and axis["orientation"].get("value")
    }

    violations, warnings = _classify(axes)

    return {
        "input": path,
        "format": "ome-zarr",
        "rfc4_valid": not violations,
        "axes": orientations,
        "violations": violations,
        "warnings": warnings,
    }
