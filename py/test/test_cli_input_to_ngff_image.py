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


def test_cli_input_to_ngff_image_nibabel(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "mri_denoised.nii.gz",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.NIBABEL, input)
    assert tuple(image.dims) == ("x", "y", "z")
    assert image.data.shape == (256, 256, 256)
    # Check that data is numpy array, not dask
    import numpy as np

    assert isinstance(image.data, np.ndarray)


def test_cli_input_to_ngff_image_imageio(input_images):  # noqa: ARG001
    input = [
        test_data_dir / "input" / "cthead1.png",
    ]
    image = cli_input_to_ngff_image(ConversionBackend.IMAGEIO, input)
    assert image.dims == ("y", "x")


def test_cli_input_to_ngff_image_imageio_spacing_mismatch():
    """Test that imageio backend handles spacing with fewer values than dims.

    This tests the fix for GitHub issue #270 where props.spacing from imageio
    has fewer elements than the image dimensions, causing an IndexError.
    """
    import tempfile
    import numpy as np

    # Create a 2D test image
    data_2d = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        import imageio.v3 as iio

        iio.imwrite(f.name, data_2d)

        # Test that the imageio backend doesn't crash with spacing
        image = cli_input_to_ngff_image(ConversionBackend.IMAGEIO, [f.name])
        assert image.dims == ("y", "x")
        # Scale should only contain spatial dimensions
        assert set(image.scale.keys()).issubset({"x", "y", "z"})
