# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from ngff_zarr import ConversionBackend, detect_cli_io_backend


def test_detect_ngff_zarr_input_backend():
    extension = ".ome.zarr"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.NGFF_ZARR


def test_detect_itk_input_backend():
    extension = ".nrrd"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.ITK


def test_detect_tifffile_input_backend():
    extension = ".svs"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.TIFFFILE


def test_detect_imageio_input_backend():
    extension = ".webm"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.IMAGEIO
