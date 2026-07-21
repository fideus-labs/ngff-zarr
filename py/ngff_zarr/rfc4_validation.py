# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC 4 validation for anatomical orientation in OME-NGFF.

This module provides validation for RFC 4 anatomical orientation metadata
against the JSON schema.
"""

import json
from pathlib import Path
from typing import Any

from importlib_resources import files as file_resources


def load_rfc4_orientation_schema() -> dict:
    """Load the RFC 4 orientation JSON schema."""
    schema = (
        file_resources("ngff_zarr")
        .joinpath(Path("spec") / Path("rfc") / Path("4") / "orientation.schema.json")
        .read_text()
    )
    return json.loads(schema)


# Each RFC 4 value maps to a stable key for the anatomical axis it lies on, so the
# two antonyms of a pair (e.g. "left-to-right" / "right-to-left") collapse to one
# key. Used to enforce "one direction per anatomical axis" (mutual exclusion).
_ANATOMICAL_AXIS_OF: dict[str, str] = {
    "left-to-right": "left-right",
    "right-to-left": "left-right",
    "anterior-to-posterior": "anterior-posterior",
    "posterior-to-anterior": "anterior-posterior",
    "inferior-to-superior": "inferior-superior",
    "superior-to-inferior": "inferior-superior",
    "dorsal-to-ventral": "dorsal-ventral",
    "ventral-to-dorsal": "dorsal-ventral",
    "dorsal-to-palmar": "dorsal-palmar",
    "palmar-to-dorsal": "dorsal-palmar",
    "dorsal-to-plantar": "dorsal-plantar",
    "plantar-to-dorsal": "dorsal-plantar",
    "rostral-to-caudal": "rostral-caudal",
    "caudal-to-rostral": "rostral-caudal",
    "cranial-to-caudal": "cranial-caudal",
    "caudal-to-cranial": "cranial-caudal",
    "proximal-to-distal": "proximal-distal",
    "distal-to-proximal": "proximal-distal",
    "superficial-to-deep": "superficial-deep",
    "deep-to-superficial": "superficial-deep",
    "apical-to-basal": "apical-basal",
    "basal-to-apical": "apical-basal",
    "apex-to-base": "apex-base",
    "base-to-apex": "apex-base",
}


def validate_rfc4_orientation(axes: list[dict[str, Any]]) -> None:
    """
    Validate RFC 4 anatomical orientation metadata against the JSON schema.

    Parameters
    ----------
    axes : List[Dict[str, Any]]
        List of axis metadata dictionaries to validate

    Raises
    ------
    ImportError
        If jsonschema is not available
    jsonschema.ValidationError
        If the orientation metadata is invalid
    ValueError
        If orientation is inconsistently or illegally defined. The messages carry
        stable marker substrings -- ``"same type"`` (mixed ``type``),
        ``"all spatial axes"`` (all-or-none completeness), ``"non-space axes"``
        (orientation on a non-spatial axis) and ``"same anatomical axis"`` (two
        axes on one antonym pair) -- that
        :func:`ngff_zarr.structural_validation.validate_axis_orientation` reads
        to map each failure onto its :class:`SpecRule`. These markers are a
        load-bearing contract pinned by a message-stability test
        (``test_rfc4_orientation_messages_carry_mapping_markers``) so the
        mapping cannot silently break if the wording is later edited.
    """
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except ImportError:
        raise ImportError(
            "jsonschema is required to validate RFC 4 orientation metadata - "
            "install the ngff-zarr[validate] extra"
        )

    # Check if any spatial axes have orientation defined
    has_orientation = False
    orientation_type = None
    spatial_axes_with_orientation = []
    spatial_axes_without_orientation = []
    spatial_orientation_values: list[tuple[str, str]] = []

    # Valid anatomical orientation values (from the schema)
    valid_orientation_values = {
        "left-to-right",
        "right-to-left",
        "anterior-to-posterior",
        "posterior-to-anterior",
        "inferior-to-superior",
        "superior-to-inferior",
        "dorsal-to-ventral",
        "ventral-to-dorsal",
        "dorsal-to-palmar",
        "palmar-to-dorsal",
        "dorsal-to-plantar",
        "plantar-to-dorsal",
        "rostral-to-caudal",
        "caudal-to-rostral",
        "cranial-to-caudal",
        "caudal-to-cranial",
        "proximal-to-distal",
        "distal-to-proximal",
        "superficial-to-deep",
        "deep-to-superficial",
        "apical-to-basal",
        "basal-to-apical",
        "apex-to-base",
        "base-to-apex",
    }

    for axis in axes:
        if isinstance(axis, dict) and axis.get("type") == "space":
            if "orientation" in axis:
                has_orientation = True
                spatial_axes_with_orientation.append(axis["name"])

                # Check that all orientations have the same type
                if orientation_type is None:
                    orientation_type = axis["orientation"].get("type")
                elif axis["orientation"].get("type") != orientation_type:
                    raise ValueError(
                        f"All spatial axis orientations must have the same type. "
                        f"Found types: {orientation_type} and {axis['orientation'].get('type')}"
                    )

                # Check that the orientation value is valid
                orientation_value = axis["orientation"].get("value")
                if orientation_value not in valid_orientation_values:
                    from jsonschema import ValidationError

                    raise ValidationError(
                        f"Invalid orientation value '{orientation_value}' for axis '{axis['name']}'. "
                        f"Valid values are: {sorted(valid_orientation_values)}"
                    )
                spatial_orientation_values.append((axis["name"], orientation_value))
            else:
                spatial_axes_without_orientation.append(axis["name"])

    # RFC 4 requirement: if orientation is defined for one spatial axis, it must
    # be defined for all spatial axes. Checked before the rules below to keep the
    # canonical SpecRule precedence (completeness is declared before the non-space
    # and unique-axis rules).
    if has_orientation and spatial_axes_without_orientation:
        raise ValueError(
            f"RFC 4 requires that if orientation is defined for one spatial axis, "
            f"it must be defined for all spatial axes. "
            f"Axes with orientation: {spatial_axes_with_orientation}, "
            f"axes without orientation: {spatial_axes_without_orientation}"
        )

    # RFC 4 requirement: orientation is only allowed on spatial axes. (Checked
    # over all axes, independently of has_orientation, so a stray orientation on
    # a time/channel axis is caught even when no spatial axis carries one.)
    non_space_with_orientation = [
        axis["name"]
        for axis in axes
        if isinstance(axis, dict)
        and axis.get("type") != "space"
        and "orientation" in axis
    ]
    if non_space_with_orientation:
        raise ValueError(
            f"RFC 4 orientation is only allowed on spatial axes; found orientation "
            f"on non-space axes: {non_space_with_orientation}"
        )

    # RFC 4 requirement: at most one direction per anatomical axis -- the two
    # antonyms of a pair are mutually exclusive across the spatial axes. A list of
    # (name, value) entries (not a name-keyed map) keeps every entry, so a
    # repeated axis name cannot mask a collision.
    anatomical_axis_seen: dict[str, str] = {}
    for name, value in spatial_orientation_values:
        axis_key = _ANATOMICAL_AXIS_OF[value]
        if axis_key in anatomical_axis_seen:
            raise ValueError(
                f"Axes '{anatomical_axis_seen[axis_key]}' and '{name}' describe the "
                f"same anatomical axis; RFC 4 allows only one direction per axis."
            )
        anatomical_axis_seen[axis_key] = name

    # If no orientation metadata found, nothing to validate
    if not has_orientation:
        return

    # Load the schema and validate the overall structure
    schema = load_rfc4_orientation_schema()
    registry = Registry().with_resource(
        "https://w3id.org/ome/ngff", resource=Resource.from_contents(schema)
    )
    validator = Draft202012Validator(schema, registry=registry)

    # Create a structure that matches the schema format
    axes_structure = {"axes": axes}

    # Validate against the schema
    validator.validate(axes_structure)


def has_rfc4_orientation_metadata(axes: list[dict[str, Any]]) -> bool:
    """
    Check if the axes contain RFC 4 anatomical orientation metadata.

    Parameters
    ----------
    axes : List[Dict[str, Any]]
        List of axis metadata dictionaries

    Returns
    -------
    bool
        True if any spatial axis has orientation metadata
    """
    for axis in axes:
        if (
            isinstance(axis, dict)
            and axis.get("type") == "space"
            and "orientation" in axis
        ):
            # Orientation value has to be non-empty for it to count as valid orientation metadata
            if axis["orientation"]:
                return True
    return False
