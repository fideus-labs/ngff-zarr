# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import sys
from enum import Enum
from pathlib import Path
from typing import List

conversion_backends = [
    ("NGFF_ZARR", "ngff_zarr"),
    ("ZARR_ARRAY", "zarr"),
    ("ITKWASM", "itkwasm_image_io"),
    ("ITK", "itk"),
    ("TIFFFILE", "tifffile"),
    ("IMAGEIO", "imageio"),
]
conversion_backends_values = [b[1] for b in conversion_backends]
ConversionBackend = Enum("ConversionBackend", conversion_backends)


def detect_cli_io_backend(input: List[str]) -> ConversionBackend:
    if (Path(input[0]) / ".zarray").exists():
        return ConversionBackend.ZARR_ARRAY

    extension = "".join(Path(input[0]).suffixes).lower()

    ngff_zarr_supported_extensions = (".zarr", ".ome.zarr")
    if extension in ngff_zarr_supported_extensions:
        return ConversionBackend.NGFF_ZARR

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
        extension in itkwasm_supported_extensions
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

    if extension in itk_supported_extensions:
        return ConversionBackend.ITK

    try:
        import tifffile

        tifffile_supported_extensions = [
            f".{ext}" for ext in tifffile.TIFF.FILE_EXTENSIONS
        ]
        if extension in tifffile_supported_extensions:
            return ConversionBackend.TIFFFILE
    except ImportError:
        from rich import print

        print("[red]Please install the [i]tifffile[/i] package")
        sys.exit(1)

    return ConversionBackend.IMAGEIO
