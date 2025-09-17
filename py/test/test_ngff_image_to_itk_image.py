# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import itk
import itkwasm
import numpy as np
from ngff_zarr import itk_image_to_ngff_image, ngff_image_to_itk_image, from_ngff_zarr

from ._data import test_data_dir

rng = np.random.default_rng(12345)


def test_2d_itk_image(input_images):  # noqa: ARG001
    itk_image = itk.imread(test_data_dir / "input" / "cthead1.png")
    ngff_image = itk_image_to_ngff_image(itk_image)
    itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    diff = itk.comparison_image_filter(itk_image, itk_image_back)
    assert np.sum(np.asarray(diff)) == 0.0


def test_2d_rgb_itk_image(input_images):  # noqa: ARG001
    array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)  # noqa: F841
    # Requires fixes for itk
    # itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    # diff = itk.comparison_image_filter(itk_image, itk_image_back)
    # assert np.sum(np.asarray(diff)) == 0.0


def test_2d_itk_vector_image(input_images):  # noqa: ARG001
    array = rng.random(size=(224, 224, 3), dtype=np.float32)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)  # noqa: F841
    # Requires fixes for itk
    # itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    # assert np.array_equal(itk.array_from_image(itk_image), itk.array_from_image(itk_image_back))


def test_3d_itk_image(input_images):  # noqa: ARG001
    array = rng.integers(0, 255, size=(32, 32, 32), dtype=np.uint8)
    itk_image = itk.image_from_array(array, is_vector=False)
    ngff_image = itk_image_to_ngff_image(itk_image)
    itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    diff = itk.comparison_image_filter(itk_image, itk_image_back)
    assert np.sum(np.asarray(diff)) == 0.0


def test_3d_itk_vector_image(input_images):  # noqa: ARG001
    array = rng.random(size=(224, 224, 128, 3), dtype=np.float32)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)  # noqa: F841
    # Requires fixes for itk
    # itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    # assert np.array_equal(itk.array_from_image(itk_image), itk.array_from_image(itk_image_back))


def test_2d_itkwasm_image(input_images):  # noqa: ARG001
    itk_image = itk.imread(test_data_dir / "input" / "cthead1.png")
    itk_image_dict = itk.dict_from_image(itk_image)
    itkwasm_image = itkwasm.Image(**itk_image_dict)
    ngff_image = itk_image_to_ngff_image(itkwasm_image)
    itkwasm_image_back = ngff_image_to_itk_image(ngff_image)
    assert np.array_equal(
        np.asarray(itkwasm_image.data), np.asarray(itkwasm_image_back.data)
    )


def test_t_index(input_images):  # noqa: ARG001
    dataset_name = "13457537"
    store_path = test_data_dir / "input" / f"{dataset_name}.zarr"
    multiscales = from_ngff_zarr(store_path)
    ngff_image = multiscales.images[0]

    itk_image = ngff_image_to_itk_image(ngff_image)

    assert itk_image.imageType.dimension == 4
    assert itk_image.imageType.components == 6
    assert len(itk_image.size) == 4
    assert len(itk_image.spacing) == 4
    assert len(itk_image.origin) == 4
    assert len(itk_image.direction) == 4
    assert itk_image.data.shape == (18, 12, 223, 198, 6)

    itk_image = ngff_image_to_itk_image(ngff_image, t_index=0)

    assert itk_image.imageType.dimension == 3
    assert itk_image.imageType.components == 6
    assert len(itk_image.size) == 3
    assert len(itk_image.spacing) == 3
    assert len(itk_image.origin) == 3
    assert len(itk_image.direction) == 3
    assert itk_image.data.shape == (12, 223, 198, 6)


def test_c_index(input_images):  # noqa: ARG001
    dataset_name = "13457537"
    store_path = test_data_dir / "input" / f"{dataset_name}.zarr"
    multiscales = from_ngff_zarr(store_path)
    ngff_image = multiscales.images[0]

    itk_image = ngff_image_to_itk_image(ngff_image)

    assert itk_image.imageType.dimension == 4
    assert itk_image.imageType.components == 6
    assert len(itk_image.size) == 4
    assert len(itk_image.spacing) == 4
    assert len(itk_image.origin) == 4
    assert len(itk_image.direction) == 4
    assert itk_image.data.shape == (18, 12, 223, 198, 6)

    itk_image = ngff_image_to_itk_image(ngff_image, c_index=0)

    assert itk_image.imageType.dimension == 4
    assert itk_image.imageType.components == 1
    assert len(itk_image.size) == 4
    assert len(itk_image.spacing) == 4
    assert len(itk_image.origin) == 4
    assert len(itk_image.direction) == 4
    assert itk_image.data.shape == (18, 12, 223, 198)

    itk_image = ngff_image_to_itk_image(ngff_image, t_index=0, c_index=0)

    assert itk_image.imageType.dimension == 3
    assert itk_image.imageType.components == 1
    assert len(itk_image.size) == 3
    assert len(itk_image.spacing) == 3
    assert len(itk_image.origin) == 3
    assert len(itk_image.direction) == 3
    assert itk_image.data.shape == (12, 223, 198)
