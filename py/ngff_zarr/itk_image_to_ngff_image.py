# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import asdict

import dask.array

from .ngff_image import NgffImage
from .rfc4 import (
    itk_direction_to_anatomical_orientation,
    itk_lps_to_anatomical_orientation,
)


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
    dimension = image_dict["imageType"]["dimension"]
    if not 1 <= dimension <= 4:
        raise ValueError(f"Unsupported ITK image dimension: {dimension}")
    # ITK axes run x, y, z, t; the array holds them reversed, components last.
    itk_axes = ("x", "y", "z", "t")[:dimension]
    dims = tuple(reversed(itk_axes))
    if data.ndim == dimension + 1:
        dims += ("c",)
    all_spatial_dims = {"x", "y", "z"}
    spatial_dims = [dim for dim in dims if dim in all_spatial_dims]

    spacing = image_dict["spacing"]
    scale = {dim: spacing[itk_axes.index(dim)] for dim in dims if dim in itk_axes}

    origin = image_dict["origin"]
    translation = {dim: origin[itk_axes.index(dim)] for dim in dims if dim in itk_axes}

    # Add anatomical orientation if requested
    axes_orientations = None
    if add_anatomical_orientation:
        import numpy as np

        direction = image_dict.get("direction")
        n_spatial = len(spatial_dims)

        # Check whether the direction matrix is available and non-identity
        has_direction = direction is not None and len(direction) > 0
        has_non_identity = False
        if has_direction:
            dir_array = np.asarray(direction).reshape(dimension, dimension)
            dir_array = dir_array[:n_spatial, :n_spatial]
            has_non_identity = not np.array_equal(dir_array, np.eye(n_spatial))

        axes_orientations = {}
        if has_non_identity:
            # Use the direction cosine matrix to determine each axis'
            # anatomical orientation. The direction matrix is stored
            # row-major: direction[row * n_spatial + col].
            #
            # spatial_dims is in array order (e.g. ["z", "y", "x"]) and
            # maps to reversed ITK axis indices:
            #   spatial_dims[0] = "z" -> ITK axis 2 -> direction column 2
            #   spatial_dims[1] = "y" -> ITK axis 1 -> direction column 1
            #   spatial_dims[2] = "x" -> ITK axis 0 -> direction column 0
            for i, dim in enumerate(spatial_dims):
                itk_axis_index = n_spatial - 1 - i
                col = [
                    float(dir_array[row, itk_axis_index]) for row in range(n_spatial)
                ]
                axes_orientations[dim] = itk_direction_to_anatomical_orientation(col)
        else:
            # Identity direction or no direction: fall back to LPS labels
            for dim in spatial_dims:
                orientation = itk_lps_to_anatomical_orientation(dim)
                if orientation is not None:
                    axes_orientations[dim] = orientation

    return NgffImage(
        data, dims, scale, translation, axes_orientations=axes_orientations
    )
