# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import sys
from enum import Enum
from pathlib import Path
from typing import List

conversion_backends = [
    ("NGFF_ZARR", "ngff_zarr"),
    ("ZARR_ARRAY", "zarr"),
    ("NIBABEL", "nibabel"),
    ("ITKWASM", "itkwasm_image_io"),
    ("ITK", "itk"),
    ("TIFFFILE", "tifffile"),
    ("IMAGEIO", "imageio"),
]
conversion_backends_values = [b[1] for b in conversion_backends]
ConversionBackend = Enum("ConversionBackend", conversion_backends)


def _matches_extension(extension: str, supported_extensions: tuple) -> bool:
    """Check if extension ends with any of the supported extensions.

    Checks longer extensions first to ensure precise matching
    (e.g., '.ome.zarr' before '.zarr').
    """
    # Sort by length descending to check longer extensions first
    sorted_extensions = sorted(supported_extensions, key=len, reverse=True)
    return any(extension.endswith(ext) for ext in sorted_extensions)


def detect_cli_io_backend(input: List[str]) -> ConversionBackend:
    if (Path(input[0]) / ".zarray").exists():
        return ConversionBackend.ZARR_ARRAY

    extension = "".join(Path(input[0]).suffixes).lower()

    # RFC-9: Support .ozx (zipped OME-Zarr) files
    ngff_zarr_supported_extensions = (".zarr", ".ome.zarr", ".ozx")
    if _matches_extension(extension, ngff_zarr_supported_extensions):
        return ConversionBackend.NGFF_ZARR

    # Prioritize NIBABEL for NIfTI files
    nibabel_supported_extensions = (".nii", ".nii.gz")
    if _matches_extension(extension, nibabel_supported_extensions):
        return ConversionBackend.NIBABEL

    itkwasm_supported_extensions = (
        ".bmp",
        ".dcm",
        ".gipl",
        ".gipl.gz",
        ".hdf5",
        ".jpg",
        ".jpeg",
        ".iwi",
        ".iwi.cbor",
        ".iwi.cbor.zst",
        ".lsm",
        ".mnc",
        ".mnc.gz",
        ".mnc2",
        ".mgh",
        ".mhz",
        ".mha",
        ".mhd",
        ".mrc",
        ".nia",
        ".nii",
        ".nii.gz",
        ".hdr",
        ".nrrd",
        ".nhdr",
        ".png",
        ".pic",
        ".vtk",
        ".aim",
        ".isq",
        ".fdf",
    )
    if (
        _matches_extension(extension, itkwasm_supported_extensions)
        and len(input) == 1
        and Path(input[0]).is_file()
        and Path(input[0]).stat().st_size < 2e9
    ):
        return ConversionBackend.ITKWASM

    itk_supported_extensions = (
        ".bmp",
        ".dcm",
        ".gipl",
        ".gipl.gz",
        ".hdf5",
        ".jpg",
        ".jpeg",
        ".iwi",
        ".iwi.cbor",
        ".iwi.cbor.zst",
        ".lsm",
        ".mnc",
        ".mnc.gz",
        ".mnc2",
        ".mgh",
        ".mhz",
        ".mha",
        ".mhd",
        ".mrc",
        ".nia",
        ".nii",
        ".nii.gz",
        ".hdr",
        ".nrrd",
        ".nhdr",
        ".png",
        ".pic",
        ".vtk",
        ".isq",  # Requires pip install itk-ioscanco,
        ".aim",  # Requires pip install itk-ioscanco,
        ".fdf",  # Requires pip install itk-iofdf
    )

    if _matches_extension(extension, itk_supported_extensions):
        return ConversionBackend.ITK

    try:
        import tifffile

        tifffile_supported_extensions = tuple(
            f".{ext}" for ext in tifffile.TIFF.FILE_EXTENSIONS
        )
        if _matches_extension(extension, tifffile_supported_extensions):
            return ConversionBackend.TIFFFILE
    except ImportError:
        from rich import print

        print("[red]Please install the [i]tifffile[/i] package")
        sys.exit(1)

    return ConversionBackend.IMAGEIO
