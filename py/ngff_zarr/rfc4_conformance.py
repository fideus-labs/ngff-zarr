# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC 4 conformance output for the ``ngff-zarr`` CLI.

``ngff-zarr conformance <input>`` prints one JSON verdict per input -- the shape
a tool-agnostic conformance driver feeds against its manifest to check that an
implementation follows RFC 4:

    {"input": ..., "format": "ome-zarr", "rfc4_valid": bool,
     "axes": {name: value}, "violations": [code, ...], "warnings": [code, ...]}

It reuses :func:`~ngff_zarr.rfc4_validation.validate_rfc4_orientation` (no rule is
reimplemented) and reads the *raw* axes, so malformed metadata is still
classifiable -- the high-level reader would reject it before it could be reported.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .rfc4_validation import validate_rfc4_orientation

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
    """Pull ``multiscales[0].axes`` out of a v2 ``.zattrs`` or v3 ``zarr.json`` blob."""
    if not isinstance(meta, dict):
        raise _UnreadableInput("input-not-ome-zarr", "metadata is not a JSON object")
    attributes = meta.get("attributes", meta)
    ome = attributes.get("ome", attributes) if isinstance(attributes, dict) else {}
    try:
        axes = ome["multiscales"][0]["axes"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _UnreadableInput(
            "input-not-ome-zarr", "metadata has no multiscales[0].axes"
        ) from exc
    if not isinstance(axes, list):
        raise _UnreadableInput(
            "input-not-ome-zarr", "multiscales[0].axes is not a list"
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


def _violation_code(exc: ValueError) -> str:
    """Map a ``validate_rfc4_orientation`` ValueError onto its SpecRule code.

    Uses the same stable message markers the structural validator relies on."""
    message = str(exc)
    if "non-space axes" in message:
        return "axis-orientation-on-non-space"
    if "same anatomical axis" in message:
        return "axis-orientation-unique-axis"
    if "same type" in message:
        return "axis-orientation-consistent-type"
    if "all spatial axes" in message:
        return "axis-orientation-completeness"
    return "orientation-invalid"


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
    # non-dict orientation) is skipped here rather than crashing the verdict --
    # validate_rfc4_orientation below still reports it as a violation.
    orientations = {
        axis["name"]: axis["orientation"]["value"]
        for axis in axes
        if isinstance(axis, dict)
        and isinstance(axis.get("name"), str)
        and isinstance(axis.get("orientation"), dict)
        and axis["orientation"].get("value")
    }

    violations: list[str] = []
    try:
        validate_rfc4_orientation(axes)
    except ValueError as exc:
        violations.append(_violation_code(exc))
    except Exception:
        violations.append("orientation-schema-invalid")

    return {
        "input": path,
        "format": "ome-zarr",
        "rfc4_valid": not violations,
        "axes": orientations,
        "violations": violations,
        "warnings": [],
    }
