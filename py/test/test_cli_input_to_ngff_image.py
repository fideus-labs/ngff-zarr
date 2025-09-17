# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import pytest
import zarr
from packaging import version

from ngff_zarr import ConversionBackend, cli_input_to_ngff_image

from ._data import test_data_dir

zarr_version = version.parse(zarr.__version__)


def test_cli_input_to_ngff_image_itk(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "cthead1.png",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.ITK, input)
    assert image.dims == ("y", "x")


def test_cli_input_to_ngff_image_itk_glob(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "lung_series" / "*.png",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.ITK, input)
    assert image.dims == ("z", "y", "x")


def test_cli_input_to_ngff_image_itk_list(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "lung_series" / "LIDC2-025.png",
        test_data_dir / "input" / "lung_series" / "LIDC2-026.png",
        test_data_dir / "input" / "lung_series" / "LIDC2-027.png",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.ITK, input)
    assert image.dims == ("z", "y", "x")


@pytest.mark.skipif(
    zarr_version >= version.parse("3.0.0b1"),
    reason="Skipping because Zarr version is greater than 3, ZarrTiffStore not yet supported",
)
def test_cli_input_to_ngff_image_tifffile(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "bat-cochlea-volume.tif",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.TIFFFILE, input)
    assert image.dims == ("z", "y", "x")


def test_cli_input_to_ngff_image_imageio(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "cthead1.png",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.IMAGEIO, input)
    assert image.dims == ("y", "x")
