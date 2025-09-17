# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC 4 implementation for anatomical orientation in OME-NGFF.

This module implements RFC 4 which adds anatomical orientation support
to OME-NGFF axes, based on the LinkML schema.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict
from enum import Enum


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
LPS: Dict[str, AnatomicalOrientation] = {
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

RAS: Dict[str, AnatomicalOrientation] = {
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
) -> Optional[AnatomicalOrientation]:
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


def is_rfc4_enabled(enabled_rfcs: Optional[list[int]]) -> bool:
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
