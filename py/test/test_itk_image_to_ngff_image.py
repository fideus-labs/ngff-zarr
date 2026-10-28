# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import warnings

import itk
import itkwasm
import numpy as np
import pytest
from ngff_zarr import itk_image_to_ngff_image

from ._data import input_images, test_data_dir  # noqa: F401

rng = np.random.default_rng(12345)


def test_2d_itk_image():
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


def test_2d_rgb_itk_image():
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


def test_2d_itk_vector_image():
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


def test_3d_itk_vector_image():
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


def test_2d_itkwasm_image():
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


def _itkwasm_4d(components=1):
    shape = (2, 3, 4, 5) + ((components,) if components > 1 else ())
    return itkwasm.Image(
        imageType=itkwasm.ImageType(
            dimension=4,
            componentType="float32",
            pixelType="Scalar" if components == 1 else "VariableLengthVector",
            components=components,
        ),
        spacing=[0.5, 0.6, 0.7, 10.0],
        origin=[1.0, 2.0, 3.0, 40.0],
        direction=np.eye(4),
        size=[5, 4, 3, 2],
        data=np.zeros(shape, dtype=np.float32),
    )


def test_4d_scalar_itkwasm_image_has_a_time_axis():
    """An x, y, z, t image (e.g. fMRI) keeps t as time, not as z or c."""
    ngff_image = itk_image_to_ngff_image(_itkwasm_4d())

    assert ngff_image.dims == ("t", "z", "y", "x")
    assert ngff_image.scale == {"t": 10.0, "z": 0.7, "y": 0.6, "x": 0.5}
    assert ngff_image.translation == {"t": 40.0, "z": 3.0, "y": 2.0, "x": 1.0}


def test_4d_itkwasm_image_round_trips():
    from ngff_zarr import ngff_image_to_itk_image

    itk_image = ngff_image_to_itk_image(itk_image_to_ngff_image(_itkwasm_4d()))

    assert list(itk_image.spacing) == [0.5, 0.6, 0.7, 10.0]
    assert list(itk_image.origin) == [1.0, 2.0, 3.0, 40.0]


def test_4d_vector_itkwasm_image():
    ngff_image = itk_image_to_ngff_image(_itkwasm_4d(components=3))

    assert ngff_image.dims == ("t", "z", "y", "x", "c")
    assert ngff_image.scale == {"t": 10.0, "z": 0.7, "y": 0.6, "x": 0.5}


def _oriented(direction):
    itk_image = itk.image_from_array(np.zeros((4, 5, 6), np.float32))
    itk_image.SetDirection(itk.matrix_from_array(np.asarray(direction, np.float64)))
    return itk_image


_FLIPPED = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
_OBLIQUE = [
    [np.cos(0.35), -np.sin(0.35), 0],
    [np.sin(0.35), np.cos(0.35), 0],
    [0, 0, 1],
]


def test_an_oblique_direction_warns_that_it_is_placed_axis_aligned():
    # Snapped to the nearest axis-aligned grid, an oblique pair's rigid result
    # converted to RFC-5 landed 19 voxels off without a word.
    with pytest.warns(UserWarning, match="oblique"):
        itk_image_to_ngff_image(_oriented(_OBLIQUE))


def test_an_integer_shear_is_not_an_orientation():
    with pytest.warns(UserWarning, match="oblique"):
        itk_image_to_ngff_image(_oriented([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))


def test_a_reversed_time_axis_warns():
    itk_image = itk.image_from_array(np.zeros((3, 4, 5, 6), np.float32))
    itk_image.SetDirection(itk.matrix_from_array(np.diag([1.0, 1.0, 1.0, -1.0])))
    with pytest.warns(UserWarning, match="time"):
        itk_image_to_ngff_image(itk_image)


def test_a_direction_dropped_without_orientation_warns():
    with pytest.warns(UserWarning, match="add_anatomical_orientation is False"):
        itk_image_to_ngff_image(_oriented(_FLIPPED), add_anatomical_orientation=False)


@pytest.mark.parametrize(
    ("direction", "oriented"),
    [(np.eye(3), True), (np.eye(3), False), (_FLIPPED, True)],
)
def test_a_direction_the_image_carries_does_not_warn(direction, oriented):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        itk_image_to_ngff_image(
            _oriented(direction), add_anatomical_orientation=oriented
        )
