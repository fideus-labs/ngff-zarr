# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-FileCopyrightText: 2022-present Matt McCormick <matt@fideus.io>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__
from ._supported_versions import SUPPORTED_VERSIONS, V06_ONDISK_VERSION
from .cli_input_to_ngff_image import cli_input_to_ngff_image
from .codecs import codec_from_name, get_available_codecs
from .compute_omero import (
    GLASBEY_COLORS,
    compute_omero_from_multiscales,
    compute_omero_from_ngff_image,
)
from .config import config
from .detect_cli_io_backend import ConversionBackend, detect_cli_io_backend
from .displacement_field_transform import (
    itk_displacement_field_to_ngff_transform,
    ngff_displacement_field_to_itk_transform,
)
from .from_ngff_zarr import from_ngff_zarr, from_ome_zarr
from .hcs import (
    HCSPlate,
    HCSPlateWriter,
    HCSWell,
    from_hcs_zarr,
    to_hcs_zarr,
    write_hcs_well_image,
)
from .itk_image_to_ngff_image import itk_image_to_ngff_image
from .itk_transform_resample import itk_transform_resample
from .itk_transform_resample_bounding_box import (
    ResampleBoundingBox,
    itk_transform_resample_bounding_box,
)
from .itk_transform_to_ngff_transform import (
    itk_transform_to_ngff_matrix,
    itk_transform_to_ngff_transform,
)
from .lif_to_ngff_image import (
    has_mosaic_dimension,
    lif_file_to_ngff_images,
    lif_to_hcs_plate,
    lif_to_ngff_image,
)
from .memory_usage import memory_usage
from .methods import Methods
from .multiscales import Multiscales, NgffMultiscales
from .ngff_image import NgffImage
from .ngff_image_to_itk_image import ngff_image_to_itk_image
from .ngff_transform_to_itk_transform import ngff_transform_to_itk_transform
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
    anatomical_orientation_to_itk_direction,
    itk_direction_to_anatomical_orientation,
    itk_lps_to_anatomical_orientation,
    orientation_from_name,
    remove_anatomical_orientation_from_axis,
)
from .rfc9_zip import (
    is_ozx_path,
    read_ozx_json_first,
    read_ozx_version,
    write_store_to_zip,
)
from .structural_validation import (
    SpecRule,
    ValidateOptions,
    ValidationError,
    ValidationLevel,
    validate_plate,
    validate_structural,
    validate_well,
)
from .task_count import task_count
from .tiff_to_ngff_image import (
    tiff_file_to_ngff_images,
)
from .to_multiscales import to_multiscales
from .to_ngff_image import to_ngff_image
from .to_ngff_zarr import ScaleStrategy, to_ngff_zarr, to_ome_zarr
from .upgrade_ome_zarr import upgrade_ome_zarr
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
    "V06_ONDISK_VERSION",
    "config",
    # OMERO computation
    "compute_omero_from_ngff_image",
    "compute_omero_from_multiscales",
    "GLASBEY_COLORS",
    "NgffImage",
    "NgffMultiscales",
    "Multiscales",
    "to_ngff_image",
    "itk_image_to_ngff_image",
    "nibabel_image_to_ngff_image",
    "extract_omero_metadata_from_nibabel",
    "ngff_image_to_itk_image",
    # RFC 5 - Coordinate transformations and ITK
    "ngff_transform_to_itk_transform",
    "itk_transform_to_ngff_matrix",
    "itk_transform_to_ngff_transform",
    "itk_displacement_field_to_ngff_transform",
    "ngff_displacement_field_to_itk_transform",
    # Out-of-core resampling
    "itk_transform_resample",
    "itk_transform_resample_bounding_box",
    "ResampleBoundingBox",
    "memory_usage",
    "task_count",
    "to_multiscales",
    "Methods",
    "to_ome_zarr",
    "to_ngff_zarr",
    "ScaleStrategy",
    "from_ome_zarr",
    "from_ngff_zarr",
    "upgrade_ome_zarr",
    "detect_cli_io_backend",
    "ConversionBackend",
    "cli_input_to_ngff_image",
    "validate",
    # Structural validation (v0.4 spec MUSTs beyond JSON Schema)
    "validate_structural",
    "validate_plate",
    "validate_well",
    "SpecRule",
    "ValidationLevel",
    "ValidateOptions",
    "ValidationError",
    # Codec utilities
    "codec_from_name",
    "get_available_codecs",
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
    "itk_lps_to_anatomical_orientation",
    "itk_direction_to_anatomical_orientation",
    "anatomical_orientation_to_itk_direction",
    "orientation_from_name",
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
