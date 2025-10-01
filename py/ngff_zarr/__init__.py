# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-FileCopyrightText: 2022-present Matt McCormick <matt@fideus.io>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__
from .cli_input_to_ngff_image import cli_input_to_ngff_image
from .config import config
from .detect_cli_io_backend import ConversionBackend, detect_cli_io_backend
from .from_ngff_zarr import from_ngff_zarr
from .itk_image_to_ngff_image import itk_image_to_ngff_image
from .nibabel_image_to_ngff_image import nibabel_image_to_ngff_image, extract_omero_metadata_from_nibabel
from .memory_usage import memory_usage
from .methods import Methods
from .multiscales import Multiscales
from .ngff_image import NgffImage
from .ngff_image_to_itk_image import ngff_image_to_itk_image
from .task_count import task_count
from .to_multiscales import to_multiscales
from .to_ngff_image import to_ngff_image
from .to_ngff_zarr import to_ngff_zarr
from .validate import validate
from .hcs import from_hcs_zarr, to_hcs_zarr, write_hcs_well_image, HCSPlate, HCSWell
from .v04.zarr_metadata import (
    AxesType,
    SpatialDims,
    SupportedDims,
    SpaceUnits,
    TimeUnits,
    Units,
    Axis,
    Identity,
    Scale,
    Translation,
    Transform,
    Dataset,
    Metadata,
    MethodMetadata,
    Omero,
    OmeroChannel,
    OmeroWindow,
    Plate,
    PlateAcquisition,
    PlateColumn,
    PlateRow,
    PlateWell,
    Well,
    WellImage,
)
from .rfc4 import (
    AnatomicalOrientation,
    AnatomicalOrientationValues,
    LPS,
    RAS,
    itk_lps_to_anatomical_orientation,
    is_rfc4_enabled,
    add_anatomical_orientation_to_axis,
    remove_anatomical_orientation_from_axis,
)

__all__ = [
    "__version__",
    "config",
    "NgffImage",
    "Multiscales",
    "to_ngff_image",
    "itk_image_to_ngff_image",
    "nibabel_image_to_ngff_image",
    "extract_omero_metadata_from_nibabel",
    "ngff_image_to_itk_image",
    "memory_usage",
    "task_count",
    "to_multiscales",
    "Methods",
    "to_ngff_zarr",
    "from_ngff_zarr",
    "detect_cli_io_backend",
    "ConversionBackend",
    "cli_input_to_ngff_image",
    "validate",
    "Metadata",
    "MethodMetadata",
    "AxesType",
    "SpatialDims",
    "SupportedDims",
    "SpaceUnits",
    "TimeUnits",
    "Units",
    "Axis",
    "Identity",
    "Scale",
    "Translation",
    "Transform",
    "Dataset",
    "Metadata",
    "Omero",
    "OmeroChannel",
    "OmeroWindow",
    # HCS (High Content Screening)
    "Plate",
    "PlateAcquisition",
    "PlateColumn",
    "PlateRow",
    "PlateWell",
    "Well",
    "WellImage",
    # HCS functions and classes
    "from_hcs_zarr",
    "to_hcs_zarr",
    "write_hcs_well_image",
    "HCSPlate",
    "HCSWell",
    # RFC 4 - Anatomical Orientation
    "AnatomicalOrientation",
    "AnatomicalOrientationValues",
    "LPS",
    "RAS",
    "itk_lps_to_anatomical_orientation",
    "is_rfc4_enabled",
    "add_anatomical_orientation_to_axis",
    "remove_anatomical_orientation_from_axis",
]
