# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC 4 validation for anatomical orientation in OME-NGFF.

This module provides validation for RFC 4 anatomical orientation metadata
against the JSON schema.
"""

from typing import Dict, List, Any
from pathlib import Path
import json

from importlib_resources import files as file_resources


def load_rfc4_orientation_schema() -> Dict:
    """Load the RFC 4 orientation JSON schema."""
    schema = (
        file_resources("ngff_zarr")
        .joinpath(Path("spec") / Path("rfc") / Path("4") / "orientation.schema.json")
        .read_text()
    )
    return json.loads(schema)


def validate_rfc4_orientation(axes: List[Dict[str, Any]]) -> None:
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
        If orientation is inconsistently defined across spatial axes
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
            else:
                spatial_axes_without_orientation.append(axis["name"])

    # RFC 4 requirement: if orientation is defined for one spatial axis,
    # it must be defined for all spatial axes
    if has_orientation and spatial_axes_without_orientation:
        raise ValueError(
            f"RFC 4 requires that if orientation is defined for one spatial axis, "
            f"it must be defined for all spatial axes. "
            f"Axes with orientation: {spatial_axes_with_orientation}, "
            f"axes without orientation: {spatial_axes_without_orientation}"
        )

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


def has_rfc4_orientation_metadata(axes: List[Dict[str, Any]]) -> bool:
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
            return True
    return False
