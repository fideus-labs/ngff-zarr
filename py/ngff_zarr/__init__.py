# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-FileCopyrightText: 2022-present Matt McCormick <matt@fideus.io>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__
from ._supported_versions import SUPPORTED_VERSIONS
from .cli_input_to_ngff_image import cli_input_to_ngff_image
from .compute_omero import (
    GLASBEY_COLORS,
    compute_omero_from_multiscales,
    compute_omero_from_ngff_image,
)
from .config import config
from .detect_cli_io_backend import ConversionBackend, detect_cli_io_backend
from .from_ngff_zarr import from_ngff_zarr
from .hcs import (
    HCSPlate,
    HCSPlateWriter,
    HCSWell,
    from_hcs_zarr,
    to_hcs_zarr,
    write_hcs_well_image,
)
from .itk_image_to_ngff_image import itk_image_to_ngff_image
from .lif_to_ngff_image import (
    has_mosaic_dimension,
    lif_file_to_ngff_images,
    lif_to_hcs_plate,
    lif_to_ngff_image,
)
from .memory_usage import memory_usage
from .methods import Methods
from .multiscales import Multiscales
from .ngff_image import NgffImage
from .ngff_image_to_itk_image import ngff_image_to_itk_image
from .nibabel_image_to_ngff_image import (
    extract_omero_metadata_from_nibabel,
    nibabel_image_to_ngff_image,
)
from .rfc4 import (
    LPS,
    RAS,
    AnatomicalOrientation,
    AnatomicalOrientationValues,
    add_anatomical_orientation_to_axis,
    is_rfc4_enabled,
    itk_direction_to_anatomical_orientation,
    itk_lps_to_anatomical_orientation,
    remove_anatomical_orientation_from_axis,
)
from .rfc9_zip import (
    is_ozx_path,
    read_ozx_json_first,
    read_ozx_version,
    write_store_to_zip,
)
from .task_count import task_count
from .tiff_to_ngff_image import (
    tiff_file_to_ngff_images,
)
from .to_multiscales import to_multiscales
from .to_ngff_image import to_ngff_image
from .to_ngff_zarr import ScaleStrategy, to_ngff_zarr
from .v04.zarr_metadata import (
    AxesType,
    Axis,
    Dataset,
    Identity,
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
    Scale,
    SpaceUnits,
    SpatialDims,
    SupportedDims,
    TimeUnits,
    Transform,
    Translation,
    Units,
    Well,
    WellImage,
)
from .validate import validate

__all__ = [
    "__version__",
    "SUPPORTED_VERSIONS",
    "config",
    # OMERO computation
    "compute_omero_from_ngff_image",
    "compute_omero_from_multiscales",
    "GLASBEY_COLORS",
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
    "ScaleStrategy",
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
    "HCSPlateWriter",
    # RFC 4 - Anatomical Orientation
    "AnatomicalOrientation",
    "AnatomicalOrientationValues",
    "LPS",
    "RAS",
    "itk_direction_to_anatomical_orientation",
    "itk_lps_to_anatomical_orientation",
    "is_rfc4_enabled",
    "add_anatomical_orientation_to_axis",
    "remove_anatomical_orientation_from_axis",
    # RFC 9 - Zipped OME-Zarr (.ozx)
    "is_ozx_path",
    "read_ozx_json_first",
    "read_ozx_version",
    "write_store_to_zip",
    # LIF (Leica Image Format) support
    "lif_to_ngff_image",
    "lif_file_to_ngff_images",
    "lif_to_hcs_plate",
    "has_mosaic_dimension",
    # TIFF support
    "tiff_file_to_ngff_images",
]
