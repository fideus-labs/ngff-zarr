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


def test_detect_nibabel_input_backend():
    extension = ".nii.gz"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.NIBABEL


def test_detect_nibabel_input_backend_nii():
    extension = ".nii"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.NIBABEL


def test_detect_imageio_input_backend():
    extension = ".webm"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.IMAGEIO


def test_detect_ngff_zarr_input_backend_with_extra_dots():
    """Test that filenames with additional dots like '.qupath.ome.zarr' are correctly detected.

    Regression test for GitHub issue #272.
    """
    backend = detect_cli_io_backend(["602a12_z_stack.qupath.ome.zarr"])
    assert backend == ConversionBackend.NGFF_ZARR


def test_detect_nibabel_input_backend_with_extra_dots():
    """Test that filenames with additional dots like '.scan.nii.gz' are correctly detected."""
    backend = detect_cli_io_backend(["patient.scan.nii.gz"])
    assert backend == ConversionBackend.NIBABEL


def test_detect_liffile_input_backend_lif():
    """Test detection of .lif (Leica Image File) extension."""
    extension = ".lif"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.LIFFILE


def test_detect_liffile_input_backend_lof():
    """Test detection of .lof (Leica Object File) extension."""
    extension = ".lof"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.LIFFILE


def test_detect_liffile_input_backend_xlif():
    """Test detection of .xlif (XML Image File) extension."""
    extension = ".xlif"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.LIFFILE


def test_detect_liffile_input_backend_xlef():
    """Test detection of .xlef (XML Experiment File) extension."""
    extension = ".xlef"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.LIFFILE


def test_detect_liffile_input_backend_xlcf():
    """Test detection of .xlcf (XML Collection File) extension."""
    extension = ".xlcf"
    backend = detect_cli_io_backend(
        [
            f"file{extension}",
        ]
    )
    assert backend == ConversionBackend.LIFFILE
