# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC 4 implementation for anatomical orientation in OME-NGFF.

This module implements RFC 4 which adds anatomical orientation support
to OME-NGFF axes, based on the LinkML schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AnatomicalOrientationValues(str, Enum):
    """
    Anatomical orientation refers to the specific arrangement and directional
    alignment of anatomical structures within an imaging dataset. It is crucial
    for ensuring accurate alignment and comparison of images to anatomical atlases,
    facilitating consistent analysis and interpretation of biological data.
    """

    # Describes the directional orientation from the left side to the right lateral side of an anatomical structure or body
    left_to_right = "left-to-right"
    # Describes the directional orientation from the right side to the left lateral side of an anatomical structure or body
    right_to_left = "right-to-left"
    # Describes the directional orientation from the front (anterior) to the back (posterior) of an anatomical structure or body
    anterior_to_posterior = "anterior-to-posterior"
    # Describes the directional orientation from the back (posterior) to the front (anterior) of an anatomical structure or body
    posterior_to_anterior = "posterior-to-anterior"
    # Describes the directional orientation from below (inferior) to above (superior) in an anatomical structure or body
    inferior_to_superior = "inferior-to-superior"
    # Describes the directional orientation from above (superior) to below (inferior) in an anatomical structure or body
    superior_to_inferior = "superior-to-inferior"
    # Describes the directional orientation from the top/upper (dorsal) to the belly/lower (ventral) in an anatomical structure or body
    dorsal_to_ventral = "dorsal-to-ventral"
    # Describes the directional orientation from the belly/lower (ventral) to the top/upper (dorsal) in an anatomical structure or body
    ventral_to_dorsal = "ventral-to-dorsal"
    # Describes the directional orientation from the top/upper (dorsal) to the palm of the hand (palmar) in a body
    dorsal_to_palmar = "dorsal-to-palmar"
    # Describes the directional orientation from the palm of the hand (palmar) to the top/upper (dorsal) in a body
    palmar_to_dorsal = "palmar-to-dorsal"
    # Describes the directional orientation from the top/upper (dorsal) to the sole of the foot (plantar) in a body
    dorsal_to_plantar = "dorsal-to-plantar"
    # Describes the directional orientation from the sole of the foot (plantar) to the top/upper (dorsal) in a body
    plantar_to_dorsal = "plantar-to-dorsal"
    # Describes the directional orientation from the nasal (rostral) to the tail (caudal) end of an anatomical structure, typically used in reference to the central nervous system
    rostral_to_caudal = "rostral-to-caudal"
    # Describes the directional orientation from the tail (caudal) to the nasal (rostral) end of an anatomical structure, typically used in reference to the central nervous system
    caudal_to_rostral = "caudal-to-rostral"
    # Describes the directional orientation from the head (cranial) to the tail (caudal) end of an anatomical structure or body
    cranial_to_caudal = "cranial-to-caudal"
    # Describes the directional orientation from the tail (caudal) to the head (cranial) end of an anatomical structure or body
    caudal_to_cranial = "caudal-to-cranial"
    # Describes the directional orientation from the center of the body to the periphery of an anatomical structure or limb
    proximal_to_distal = "proximal-to-distal"
    # Describes the directional orientation from the periphery of an anatomical structure or limb to the center of the body
    distal_to_proximal = "distal-to-proximal"


@dataclass
class AnatomicalOrientation:
    """Anatomical orientation specification for spatial axes."""

    value: AnatomicalOrientationValues
    type: Literal["anatomical"] = "anatomical"


# Convenience constants for common coordinate systems
LPS: dict[str, AnatomicalOrientation] = {
    "x": AnatomicalOrientation(
        type="anatomical", value=AnatomicalOrientationValues.right_to_left
    ),
    "y": AnatomicalOrientation(
        type="anatomical", value=AnatomicalOrientationValues.anterior_to_posterior
    ),
    "z": AnatomicalOrientation(
        type="anatomical", value=AnatomicalOrientationValues.inferior_to_superior
    ),
}
"""
LPS (Left-Posterior-Superior) coordinate system orientations.
In LPS, the axes increase from:
- X: right-to-left (L = Left)
- Y: anterior-to-posterior (P = Posterior)
- Z: inferior-to-superior (S = Superior)
This is the standard coordinate system used by ITK and many medical imaging applications.

Example usage:
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        axes_orientations=LPS
    )
"""

RAS: dict[str, AnatomicalOrientation] = {
    "x": AnatomicalOrientation(
        type="anatomical", value=AnatomicalOrientationValues.left_to_right
    ),
    "y": AnatomicalOrientation(
        type="anatomical", value=AnatomicalOrientationValues.posterior_to_anterior
    ),
    "z": AnatomicalOrientation(
        type="anatomical", value=AnatomicalOrientationValues.inferior_to_superior
    ),
}
"""
RAS (Right-Anterior-Superior) coordinate system orientations.
In RAS, the axes increase from:
- X: left-to-right (R = Right)
- Y: posterior-to-anterior (A = Anterior)
- Z: inferior-to-superior (S = Superior)
This coordinate system is commonly used in neuroimaging applications like FreeSurfer and FSL.

Example usage:
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        axes_orientations=RAS
    )
"""


def itk_lps_to_anatomical_orientation(
    axis_name: str,
) -> AnatomicalOrientation | None:
    """
    Convert ITK LPS coordinate system to anatomical orientation.

    ITK uses the LPS (Left-Posterior-Superior) coordinate system by default.
    In LPS, the axes increase from:
    - X: right-to-left (L = Left)
    - Y: anterior-to-posterior (P = Posterior)
    - Z: inferior-to-superior (S = Superior)

    Parameters
    ----------
    axis_name : str
        The axis name ('x', 'y', or 'z')

    Returns
    -------
    Optional[AnatomicalOrientation]
        The corresponding anatomical orientation, or None for non-spatial axes
    """
    return LPS.get(axis_name)


_LPS_ORIENTATION_PAIRS = [
    (
        AnatomicalOrientationValues.right_to_left,
        AnatomicalOrientationValues.left_to_right,
    ),
    (
        AnatomicalOrientationValues.anterior_to_posterior,
        AnatomicalOrientationValues.posterior_to_anterior,
    ),
    (
        AnatomicalOrientationValues.inferior_to_superior,
        AnatomicalOrientationValues.superior_to_inferior,
    ),
]
"""LPS orientation pairs: positive and negative orientations for each
physical axis in the LPS coordinate system.

Each row is ``(positive_orientation, negative_orientation)`` where
*positive* means the direction cosine value is >= 0 and *negative* means
it is < 0.

- Row 0 (L/R): positive -> RightToLeft, negative -> LeftToRight
- Row 1 (A/P): positive -> AnteriorToPosterior, negative -> PosteriorToAnterior
- Row 2 (I/S): positive -> InferiorToSuperior, negative -> SuperiorToInferior
"""


_MAX_SPATIAL_DIMENSIONS = 3


def itk_direction_to_anatomical_orientation(
    direction_column: list[float] | tuple[float, ...],
) -> AnatomicalOrientation:
    """Determine the anatomical orientation from a direction cosine vector.

    The direction vector is a column of the ITK direction matrix. Its
    dominant component (largest absolute value) determines which physical
    axis (L/R, A/P, or S/I) this image axis aligns with, and the sign
    of that component determines the direction.

    In LPS physical space:
    - Component 0: L/R axis (positive = Right->Left, negative = Left->Right)
    - Component 1: A/P axis (positive = Anterior->Posterior, negative = Posterior->Anterior)
    - Component 2: I/S axis (positive = Inferior->Superior, negative = Superior->Inferior)

    For 2D images, the direction column will have only 2 components, with the
    third component implicitly zero.

    Parameters
    ----------
    direction_column : list[float] | tuple[float, ...]
        Direction cosine vector for the image axis.
        Can be 2-element (for 2D images) or 3-element (for 3D images).
        Must have at least 2 elements.

    Returns
    -------
    AnatomicalOrientation
        The anatomical orientation corresponding to the dominant direction
        of this axis.

    Raises
    ------
    ValueError
        If direction_column has fewer than 2 elements.
    """
    if len(direction_column) < 2:
        msg = (
            f"Direction column must have at least 2 elements for 2D/3D images, "
            f"got {len(direction_column)}"
        )
        raise ValueError(msg)

    # Find the dominant physical axis (largest absolute component)
    n = min(len(direction_column), _MAX_SPATIAL_DIMENSIONS)
    dominant_axis = 0
    max_abs = abs(direction_column[0])
    for i in range(1, n):
        abs_val = abs(direction_column[i])
        if abs_val > max_abs:
            max_abs = abs_val
            dominant_axis = i

    # Positive component -> positive LPS direction, negative -> opposite
    pos, neg = _LPS_ORIENTATION_PAIRS[dominant_axis]
    value = pos if direction_column[dominant_axis] >= 0 else neg
    return AnatomicalOrientation(value=value)


def anatomical_orientation_to_itk_direction(
    orientation: AnatomicalOrientationValues,
) -> list[float] | None:
    """Convert an anatomical orientation to an ITK direction cosine vector.

    This is the reverse of :func:`itk_direction_to_anatomical_orientation`.
    Given an anatomical orientation value, it returns the unit direction
    column vector in ITK LPS physical space.

    Only the six LPS-compatible orientations are supported.  For any other
    orientation (e.g. dorsal, palmar, rostral, etc.) the function returns
    ``None``, signalling that the caller should fall back to an identity
    direction matrix.

    Parameters
    ----------
    orientation : AnatomicalOrientationValues
        The anatomical orientation value.

    Returns
    -------
    list[float] | None
        A 3-element unit direction column vector ``[lps_x, lps_y, lps_z]``,
        or ``None`` if the orientation is not one of the six LPS-compatible
        values.
    """
    for axis_index, (pos, neg) in enumerate(_LPS_ORIENTATION_PAIRS):
        if orientation == pos:
            col = [0.0, 0.0, 0.0]
            col[axis_index] = 1.0
            return col
        if orientation == neg:
            col = [0.0, 0.0, 0.0]
            col[axis_index] = -1.0
            return col
    return None


def is_rfc4_enabled(enabled_rfcs: list[int] | None) -> bool:
    """Check if RFC 4 is enabled in the list of enabled RFCs."""
    return enabled_rfcs is not None and 4 in enabled_rfcs


def add_anatomical_orientation_to_axis(
    axis_dict: dict, orientation: AnatomicalOrientation
) -> dict:
    """
    Add anatomical orientation to an axis dictionary.

    Parameters
    ----------
    axis_dict : dict
        The axis dictionary to modify
    orientation : AnatomicalOrientation
        The anatomical orientation to add

    Returns
    -------
    dict
        The modified axis dictionary
    """
    axis_dict["orientation"] = {
        "type": orientation.type,
        "value": orientation.value.value,
    }
    return axis_dict


def remove_anatomical_orientation_from_axis(axis_dict: dict) -> dict:
    """
    Remove anatomical orientation from an axis dictionary.

    Parameters
    ----------
    axis_dict : dict
        The axis dictionary to modify

    Returns
    -------
    dict
        The modified axis dictionary
    """
    if "orientation" in axis_dict:
        axis_dict.pop("orientation")
    return axis_dict
