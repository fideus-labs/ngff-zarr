# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import itk
import itkwasm
import numpy as np
from ngff_zarr import itk_image_to_ngff_image

from ._data import test_data_dir

rng = np.random.default_rng(12345)


def test_2d_itk_image(input_images):  # noqa: ARG001
    itk_image = itk.imread(test_data_dir / "input" / "cthead1.png")
    ngff_image = itk_image_to_ngff_image(itk_image)
    assert np.array_equal(np.asarray(itk_image), np.asarray(ngff_image.data))
    assert ngff_image.dims == ("y", "x")
    assert ngff_image.scale["x"] == 1.0
    assert ngff_image.scale["y"] == 1.0
    assert ngff_image.translation["x"] == 0.0
    assert ngff_image.translation["y"] == 0.0
    assert ngff_image.name == "image"
    assert ngff_image.axes_units is None


def test_2d_rgb_itk_image(input_images):  # noqa: ARG001
    array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)
    assert np.array_equal(np.asarray(itk_image), array)
    assert np.array_equal(np.asarray(ngff_image.data), array)
    assert ngff_image.dims == ("y", "x", "c")
    assert ngff_image.scale["x"] == 1.0
    assert ngff_image.scale["y"] == 1.0
    assert ngff_image.translation["x"] == 0.0
    assert ngff_image.translation["y"] == 0.0
    assert ngff_image.name == "image"
    assert ngff_image.axes_units is None


def test_2d_itk_vector_image(input_images):  # noqa: ARG001
    array = rng.random(size=(224, 224, 3), dtype=np.float32)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)
    assert np.array_equal(itk.array_from_image(itk_image), array)
    assert np.array_equal(np.asarray(ngff_image.data), array)
    assert ngff_image.dims == ("y", "x", "c")
    assert ngff_image.scale["x"] == 1.0
    assert ngff_image.scale["y"] == 1.0
    assert ngff_image.translation["x"] == 0.0
    assert ngff_image.translation["y"] == 0.0
    assert ngff_image.name == "image"
    assert ngff_image.axes_units is None


def test_3d_itk_vector_image(input_images):  # noqa: ARG001
    array = rng.random(size=(224, 224, 128, 3), dtype=np.float32)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)
    assert np.array_equal(itk.array_from_image(itk_image), array)
    assert np.array_equal(np.asarray(ngff_image.data), array)
    assert ngff_image.dims == ("z", "y", "x", "c")
    assert ngff_image.scale["x"] == 1.0
    assert ngff_image.scale["y"] == 1.0
    assert ngff_image.scale["z"] == 1.0
    assert ngff_image.translation["x"] == 0.0
    assert ngff_image.translation["y"] == 0.0
    assert ngff_image.translation["z"] == 0.0
    assert ngff_image.name == "image"
    assert ngff_image.axes_units is None


def test_2d_itkwasm_image(input_images):  # noqa: ARG001
    itk_image = itk.imread(test_data_dir / "input" / "cthead1.png")
    itk_image_dict = itk.dict_from_image(itk_image)
    itkwasm_image = itkwasm.Image(**itk_image_dict)
    ngff_image = itk_image_to_ngff_image(itkwasm_image)
    assert np.array_equal(np.asarray(itk_image), np.asarray(ngff_image.data))
    assert ngff_image.dims == ("y", "x")
    assert ngff_image.scale["x"] == 1.0
    assert ngff_image.scale["y"] == 1.0
    assert ngff_image.translation["x"] == 0.0
    assert ngff_image.translation["y"] == 0.0
    assert ngff_image.name == "image"
    assert ngff_image.axes_units is None
