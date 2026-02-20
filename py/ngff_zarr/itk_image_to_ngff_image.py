# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import asdict

import dask.array
import numpy as np

from .ngff_image import NgffImage
from .rfc4 import itk_lps_to_anatomical_orientation, itk_direction_to_anatomical_orientation


def _flatten_direction(direction, n: int):
    """Return direction as a flat 1-D array of length ``n * n``.

    ``itk.dict_from_image`` returns a 2-D numpy array while
    ``itkwasm`` returns a flat array.  This helper normalises both
    and validates that the resulting size is ``n * n``.
    """
    flat = np.asarray(direction, dtype=float).reshape(-1)
    expected_size = n * n
    if flat.size != expected_size:
        raise ValueError(
            f"Direction has length {flat.size}, but expected {expected_size} (for n={n})."
        )
    return flat


def _is_identity_direction(direction, n: int) -> bool:
    """Check whether a direction matrix is the identity matrix."""
    flat = _flatten_direction(direction, n)
    for row in range(n):
        for col in range(n):
            expected = 1 if row == col else 0
            if flat[row * n + col] != expected:
                return False
    return True


def itk_image_to_ngff_image(
    itk_image,
    add_anatomical_orientation: bool = True,
    # axis_units: List[str] = None,
):
    """Convert an itk.Image or an itkwasm.Image to an NgffImage, preserving spatial metadata."""

    # # Orient 3D image so that direction is identity wrt RAI coordinates
    # image_dimension = image.GetImageDimension()
    # input_direction = np.array(image.GetDirection())
    # oriented_image = image
    # if anatomical_axes and image_dimension == 3 and not (np.eye(image_dimension) == input_direction).all():
    #     desired_orientation = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI
    #     oriented_image = itk.orient_image_filter(image, use_image_direction=True, desired_coordinate_orientation=desired_orientation)
    # elif anatomical_axes and image_dimension != 3:
    #     raise ValueError(f'Cannot use anatomical axes for input image of size {image_dimension}')

    image_dict = None

    try:
        import itk

        if isinstance(itk_image, (itk.Image, itk.VectorImage)):
            image_dict = itk.dict_from_image(itk_image)
    except ImportError:
        pass

    try:
        import itkwasm

        if isinstance(itk_image, itkwasm.Image):
            image_dict = asdict(itk_image)
    except ImportError:
        pass

    if image_dict is None:
        msg = (
            "Could not import itk or itkwasm or input is not itk.Image or itkwasm.Image"
        )
        raise RuntimeError(msg)

    data = dask.array.from_array(image_dict["data"])
    ndim = data.ndim
    if ndim == 3 and image_dict["imageType"]["components"] > 1:
        dims = ("y", "x", "c")
    elif ndim < 4:
        dims = ("z", "y", "x")[-ndim:]
    elif ndim < 5:
        dims = ("z", "y", "x", "c")
    elif ndim < 6:
        dims = ("t", "z", "y", "x", "c")
    all_spatial_dims = {"x", "y", "z"}
    spatial_dims = [dim for dim in dims if dim in all_spatial_dims]

    spacing = image_dict["spacing"]
    scale = {dim: spacing[::-1][idx] for idx, dim in enumerate(spatial_dims)}

    origin = image_dict["origin"]
    translation = {dim: origin[::-1][idx] for idx, dim in enumerate(spatial_dims)}

    # Add anatomical orientation if requested
    axes_orientations = None
    if add_anatomical_orientation:
        axes_orientations = {}
        direction = image_dict.get("direction")
        n_spatial = len(spatial_dims)
        itk_dim = image_dict["imageType"]["dimension"]

        has_direction = direction is not None and len(direction) > 0

        # The direction matrix from ITK has shape itk_dim × itk_dim.
        # We need the n_spatial × n_spatial spatial sub-matrix.
        if has_direction:
            flat_full = _flatten_direction(direction, itk_dim)
            # Extract the spatial sub-matrix (top-left n_spatial × n_spatial)
            flat_dir = np.array([
                flat_full[row * itk_dim + col]
                for row in range(n_spatial)
                for col in range(n_spatial)
            ])
        else:
            flat_dir = None

        has_non_identity_direction = flat_dir is not None and not _is_identity_direction(
            flat_dir, n_spatial
        )

        if has_non_identity_direction:
            for i, dim in enumerate(spatial_dims):
                itk_axis_index = n_spatial - 1 - i
                col = [
                    flat_dir[row * n_spatial + itk_axis_index]
                    for row in range(n_spatial)
                ]
                axes_orientations[dim] = itk_direction_to_anatomical_orientation(col)
        else:
            for dim in spatial_dims:
                orientation = itk_lps_to_anatomical_orientation(dim)
                if orientation is not None:
                    axes_orientations[dim] = orientation

    return NgffImage(
        data, dims, scale, translation, axes_orientations=axes_orientations
    )
