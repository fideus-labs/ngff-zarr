# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Dict, Tuple, Optional
import numpy as np

from .ngff_image import NgffImage
from .rfc4 import AnatomicalOrientation, AnatomicalOrientationValues


def decompose_affine_with_shear(affine):
    """Decompose affine transformation matrix into translation, scale, shear, and orientation."""
    # Affine top-left 3x3: linear (rotation, scale, shear), last column: translation
    matrix = affine[:3, :3]
    translation = affine[:3, 3]

    # Extract scale: norm of each column (preserves axis order)
    scale = np.linalg.norm(matrix, axis=0)

    # Normalize columns to remove scale for shear/orientation steps
    normed_matrix = matrix / scale

    # Shear extraction (per scipy/ITK/transforms3d conventions)
    shear_xy = np.dot(normed_matrix[:, 0], normed_matrix[:, 1])
    y_orth = normed_matrix[:, 1] - shear_xy * normed_matrix[:, 0]
    shear_y = np.linalg.norm(y_orth)
    shear_xz = np.dot(normed_matrix[:, 0], normed_matrix[:, 2])
    shear_yz = np.dot(normed_matrix[:, 1], normed_matrix[:, 2])
    z_orth = normed_matrix[:, 2] - shear_xz * normed_matrix[:, 0] - shear_yz * normed_matrix[:, 1]
    shear_z = np.linalg.norm(z_orth)

    shear = np.array([shear_xy, shear_xz, shear_yz])

    # Orthonormal rotation/orientation: Gram-Schmidt
    x = normed_matrix[:, 0]
    y = y_orth / shear_y
    z = z_orth / shear_z
    orientation = np.stack([x, y, z], axis=1)

    # Compose explicit scale and shear matrices
    scale_matrix = np.diag(scale)
    shear_matrix = np.array([[1, shear_xy, shear_xz],
                             [0,     1,   shear_yz],
                             [0,     0,      1    ]])

    # The "remaining affine" is shear * orientation
    remaining_affine_matrix = orientation @ shear_matrix

    return {
        "translation": translation,
        "scale": scale,                # pixel spacing in x,y,z order
        "shear": shear,                # [shear_xy, shear_xz, shear_yz]
        "orientation": orientation,    # rotation (orthonormal, columns x y z)
        "remaining_affine": remaining_affine_matrix
    }


def extract_spatial_metadata(img) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    """Extract translation, scale, and orientation from NIfTI spatial metadata."""
    # Get the affine transformation matrix
    affine = img.affine

    # Use axis-preserving decomposition
    decomposition = decompose_affine_with_shear(affine)

    # Get image dimensions
    shape = img.shape

    # Create scale and translation dictionaries for spatial dimensions
    if len(shape) >= 3:
        # NIfTI uses RAS+ coordinate system, but array indexing is in reverse order
        # The affine maps from voxel coordinates (i,j,k) to world coordinates (x,y,z)
        # Array dimensions are typically (z,y,x) or (t,z,y,x)
        spatial_dims = ['x', 'y', 'z']  # World coordinate order
        scale_dict = {dim: float(decomposition["scale"][i]) for i, dim in enumerate(spatial_dims)}
        translation_dict = {dim: float(decomposition["translation"][i]) for i, dim in enumerate(spatial_dims)}

        # Add time dimension if 4D
        if len(shape) == 4:
            scale_dict['t'] = 1.0
            translation_dict['t'] = 0.0
    else:
        raise ValueError(f"Image must have at least 3 dimensions, got {len(shape)}")

    return scale_dict, translation_dict, affine


def nibabel_image_to_ngff_image(
    nibabel_image,
    add_anatomical_orientation: bool = True,
) -> NgffImage:
    """Convert a nibabel image to an NgffImage, preserving spatial metadata.

    Args:
        nibabel_image: A nibabel image object
        add_anatomical_orientation: Whether to add anatomical orientation metadata.
                                   Only added if orientation and shear matrices are identity.

    Returns:
        NgffImage with spatial metadata from the NIfTI file
    """
    # Get image data as numpy array (not Dask)
    data = nibabel_image.get_fdata()

    # Extract spatial metadata
    scale_dict, translation_dict, affine = extract_spatial_metadata(nibabel_image)

    # Determine dimension names based on image shape
    if len(data.shape) == 3:
        dims = ['x', 'y', 'z']
    elif len(data.shape) == 4:
        dims = ['x', 'y', 'z', 't']
    elif len(data.shape) == 5:
        dims = ['x', 'y', 'z', 't', 'c']
    else:
        raise ValueError(f"Unsupported number of dimensions: {len(data.shape)}")

    # Create anatomical orientations for spatial dimensions (RAS) only if matrices are identity
    axes_orientations = None
    if add_anatomical_orientation:
        # Decompose the affine to check if orientation and shear are identity
        decomposition = decompose_affine_with_shear(affine)
        orientation_matrix = decomposition["orientation"]
        shear = decomposition["shear"]

        # Check if orientation is identity (within tolerance)
        identity_orientation = np.allclose(orientation_matrix, np.eye(3), atol=1e-6)
        # Check if shear is zero (within tolerance)
        zero_shear = np.allclose(shear, 0.0, atol=1e-6)

        if identity_orientation and zero_shear:
            axes_orientations = {}
            spatial_dims = [dim for dim in dims if dim in ['x', 'y', 'z']]

            for dim in spatial_dims:
                if dim == 'x':
                    axes_orientations[dim] = AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right)
                elif dim == 'y':
                    axes_orientations[dim] = AnatomicalOrientation(value=AnatomicalOrientationValues.posterior_to_anterior)
                elif dim == 'z':
                    axes_orientations[dim] = AnatomicalOrientation(value=AnatomicalOrientationValues.inferior_to_superior)

    # Create NgffImage
    ngff_img = NgffImage(
        data=data,  # Keep as numpy array
        dims=dims,
        scale=scale_dict,
        translation=translation_dict,
        name="nibabel_converted_image",
        axes_orientations=axes_orientations
    )

    return ngff_img