# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import asdict

import dask.array

from .ngff_image import NgffImage
from .rfc4 import itk_lps_to_anatomical_orientation


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
        for dim in spatial_dims:
            orientation = itk_lps_to_anatomical_orientation(dim)
            if orientation is not None:
                axes_orientations[dim] = orientation

    return NgffImage(
        data, dims, scale, translation, axes_orientations=axes_orientations
    )
