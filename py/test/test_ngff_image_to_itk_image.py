# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import itk
import itkwasm
import numpy as np
from ngff_zarr import from_ngff_zarr, itk_image_to_ngff_image, ngff_image_to_itk_image

from ._data import test_data_dir

rng = np.random.default_rng(12345)


def test_2d_itk_image(input_images):
    itk_image = itk.imread(test_data_dir / "input" / "cthead1.png")
    ngff_image = itk_image_to_ngff_image(itk_image)
    itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    diff = itk.comparison_image_filter(itk_image, itk_image_back)
    assert np.sum(np.asarray(diff)) == 0.0


def test_2d_rgb_itk_image(input_images):
    array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)  # noqa: F841
    # Requires fixes for itk
    # itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    # diff = itk.comparison_image_filter(itk_image, itk_image_back)
    # assert np.sum(np.asarray(diff)) == 0.0


def test_2d_itk_vector_image(input_images):
    array = rng.random(size=(224, 224, 3), dtype=np.float32)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)  # noqa: F841
    # Requires fixes for itk
    # itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    # assert np.array_equal(itk.array_from_image(itk_image), itk.array_from_image(itk_image_back))


def test_3d_itk_image(input_images):
    array = rng.integers(0, 255, size=(32, 32, 32), dtype=np.uint8)
    itk_image = itk.image_from_array(array, is_vector=False)
    ngff_image = itk_image_to_ngff_image(itk_image)
    itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    diff = itk.comparison_image_filter(itk_image, itk_image_back)
    assert np.sum(np.asarray(diff)) == 0.0


def test_3d_itk_vector_image(input_images):
    array = rng.random(size=(224, 224, 128, 3), dtype=np.float32)
    itk_image = itk.image_from_array(array, is_vector=True)
    ngff_image = itk_image_to_ngff_image(itk_image)  # noqa: F841
    # Requires fixes for itk
    # itk_image_back = ngff_image_to_itk_image(ngff_image, wasm=False)
    # assert np.array_equal(itk.array_from_image(itk_image), itk.array_from_image(itk_image_back))


def test_2d_itkwasm_image(input_images):
    itk_image = itk.imread(test_data_dir / "input" / "cthead1.png")
    itk_image_dict = itk.dict_from_image(itk_image)
    itkwasm_image = itkwasm.Image(**itk_image_dict)
    ngff_image = itk_image_to_ngff_image(itkwasm_image)
    itkwasm_image_back = ngff_image_to_itk_image(ngff_image)
    assert np.array_equal(
        np.asarray(itkwasm_image.data), np.asarray(itkwasm_image_back.data)
    )


def test_t_index(input_images):
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


def test_c_index(input_images):
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


def test_t_index_with_none_axes_units():
    """Test t_index extraction when axes_units is None."""
    import dask.array as da
    from ngff_zarr import NgffImage

    # Create a 4D image with t, z, y, x dimensions
    data = da.zeros((2, 4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("t", "z", "y", "x"),
        scale={"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation={"t": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        axes_units=None,  # This is the key part - axes_units is None
    )

    # This should not raise TypeError
    itk_image = ngff_image_to_itk_image(ngff_image, t_index=0)
    assert itk_image is not None
    # Verify that t_index actually reduced dimensions (from 4D to 3D)
    assert itk_image.imageType.dimension == 3
    assert len(itk_image.size) == 3
    assert itk_image.data.shape == (4, 8, 8)  # (z, y, x) after removing t


def test_c_index_with_none_axes_units():
    """Test c_index extraction when axes_units is None."""
    import dask.array as da
    from ngff_zarr import NgffImage

    # Create a 4D image with c, z, y, x dimensions
    data = da.zeros((3, 4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("c", "z", "y", "x"),
        scale={"c": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation={"c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        axes_units=None,  # This is the key part - axes_units is None
    )

    # This should not raise TypeError
    itk_image = ngff_image_to_itk_image(ngff_image, c_index=0)
    assert itk_image is not None
    # Verify that c_index actually reduced components (from 3 to 1)
    assert itk_image.imageType.dimension == 3
    assert itk_image.imageType.components == 1
    assert len(itk_image.size) == 3
    assert itk_image.data.shape == (4, 8, 8)  # (z, y, x) after removing c


def test_t_and_c_index_with_none_axes_units():
    """Test both t_index and c_index extraction when axes_units is None."""
    import dask.array as da
    from ngff_zarr import NgffImage

    # Create a 5D image with t, c, z, y, x dimensions
    data = da.zeros((2, 3, 4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("t", "c", "z", "y", "x"),
        scale={"t": 1.0, "c": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        axes_units=None,  # This is the key part - axes_units is None
    )

    # This should not raise TypeError
    itk_image = ngff_image_to_itk_image(ngff_image, t_index=0, c_index=0)
    assert itk_image is not None
    # Verify that both t_index and c_index reduced dimensions (from 5D to 3D)
    assert itk_image.imageType.dimension == 3
    assert itk_image.imageType.components == 1
    assert len(itk_image.size) == 3
    assert itk_image.data.shape == (4, 8, 8)  # (z, y, x) after removing t and c


def test_t_index_with_partial_axes_units():
    """Test t_index extraction when axes_units has only some dimensions."""
    import dask.array as da
    from ngff_zarr import NgffImage

    # Create a 4D image with t, z, y, x dimensions
    # axes_units only has 'z', missing 't', 'y', 'x'
    data = da.zeros((2, 4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("t", "z", "y", "x"),
        scale={"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation={"t": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        axes_units={"z": "micrometer"},  # Partial mapping - only 'z' has units
    )

    # This should not raise KeyError
    itk_image = ngff_image_to_itk_image(ngff_image, t_index=0)
    assert itk_image is not None
    # Verify that t_index actually reduced dimensions (from 4D to 3D)
    assert itk_image.imageType.dimension == 3
    assert len(itk_image.size) == 3
    assert itk_image.data.shape == (4, 8, 8)  # (z, y, x) after removing t


def test_c_index_with_empty_axes_units():
    """Test c_index extraction when axes_units is an empty dict."""
    import dask.array as da
    from ngff_zarr import NgffImage

    # Create a 4D image with c, z, y, x dimensions
    data = da.zeros((3, 4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("c", "z", "y", "x"),
        scale={"c": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation={"c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        axes_units={},  # Empty dict - should be preserved as empty dict
    )

    # This should not raise KeyError
    itk_image = ngff_image_to_itk_image(ngff_image, c_index=0)
    assert itk_image is not None
    # Verify that c_index actually reduced components (from 3 to 1)
    assert itk_image.imageType.dimension == 3
    assert itk_image.imageType.components == 1
    assert len(itk_image.size) == 3
    assert itk_image.data.shape == (4, 8, 8)  # (z, y, x) after removing c


# --- Tests for direction matrix from anatomical orientation ---


def test_direction_matrix_without_orientation():
    """NgffImage without axes_orientations produces identity direction."""
    import dask.array as da
    from ngff_zarr import NgffImage

    data = da.zeros((4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    )

    itk_image = ngff_image_to_itk_image(ngff_image)

    assert np.array_equal(np.asarray(itk_image.direction).reshape(3, 3), np.eye(3))


def test_direction_matrix_from_lps_orientation():
    """NgffImage with LPS orientations produces identity direction."""
    import dask.array as da
    from ngff_zarr import LPS, NgffImage

    data = da.zeros((4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        axes_orientations=LPS,
    )

    itk_image = ngff_image_to_itk_image(ngff_image)

    assert np.array_equal(np.asarray(itk_image.direction).reshape(3, 3), np.eye(3))


def test_direction_matrix_from_ras_orientation():
    """NgffImage with RAS orientations produces flipped X/Y direction."""
    import dask.array as da
    from ngff_zarr import RAS, NgffImage

    data = da.zeros((4, 8, 8), dtype=np.uint8)
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        axes_orientations=RAS,
    )

    itk_image = ngff_image_to_itk_image(ngff_image)

    # RAS: X is left-to-right (-1), Y is posterior-to-anterior (-1),
    # Z is inferior-to-superior (+1)
    expected = np.diag([-1.0, -1.0, 1.0])
    assert np.array_equal(np.asarray(itk_image.direction).reshape(3, 3), expected)


def test_direction_matrix_from_permuted_orientation():
    """NgffImage with permuted orientations produces non-identity direction."""
    import dask.array as da
    from ngff_zarr import AnatomicalOrientation, AnatomicalOrientationValues, NgffImage

    data = da.zeros((4, 8, 8), dtype=np.uint8)
    # x maps to A/P (Y physical), y maps to I/S (Z physical),
    # z maps to R/L (X physical)
    axes_orientations = {
        "x": AnatomicalOrientation(
            value=AnatomicalOrientationValues.anterior_to_posterior
        ),
        "y": AnatomicalOrientation(
            value=AnatomicalOrientationValues.inferior_to_superior
        ),
        "z": AnatomicalOrientation(value=AnatomicalOrientationValues.right_to_left),
    }
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        axes_orientations=axes_orientations,
    )

    itk_image = ngff_image_to_itk_image(ngff_image)

    direction = np.asarray(itk_image.direction).reshape(3, 3)
    # itk_dims sorted = [x, y, z]
    # col 0 (x) -> A/P -> [0, 1, 0]
    # col 1 (y) -> I/S -> [0, 0, 1]
    # col 2 (z) -> R/L -> [1, 0, 0]
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert np.array_equal(direction, expected)


def test_direction_matrix_non_lps_fallback():
    """NgffImage with non-LPS orientation falls back to identity."""
    import dask.array as da
    from ngff_zarr import AnatomicalOrientation, AnatomicalOrientationValues, NgffImage

    data = da.zeros((4, 8, 8), dtype=np.uint8)
    axes_orientations = {
        "x": AnatomicalOrientation(value=AnatomicalOrientationValues.right_to_left),
        "y": AnatomicalOrientation(value=AnatomicalOrientationValues.dorsal_to_ventral),
        "z": AnatomicalOrientation(
            value=AnatomicalOrientationValues.inferior_to_superior
        ),
    }
    ngff_image = NgffImage(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        axes_orientations=axes_orientations,
    )

    itk_image = ngff_image_to_itk_image(ngff_image)

    # Should fall back to identity since dorsal_to_ventral is not LPS
    assert np.array_equal(np.asarray(itk_image.direction).reshape(3, 3), np.eye(3))


def test_direction_matrix_2d():
    """2D NgffImage with orientations produces correct 2x2 direction."""
    import dask.array as da
    from ngff_zarr import RAS, NgffImage

    data = da.zeros((8, 8), dtype=np.uint8)
    # Use RAS x and y only
    axes_orientations = {
        "x": RAS["x"],
        "y": RAS["y"],
    }
    ngff_image = NgffImage(
        data=data,
        dims=("y", "x"),
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
        axes_orientations=axes_orientations,
    )

    itk_image = ngff_image_to_itk_image(ngff_image)

    expected = np.diag([-1.0, -1.0])
    assert np.array_equal(np.asarray(itk_image.direction).reshape(2, 2), expected)


def test_direction_round_trip_identity():
    """Round-trip ITK -> NgffImage -> ITK preserves identity direction."""
    img = itkwasm.Image(
        imageType=itkwasm.ImageType(
            dimension=3,
            componentType="uint8",
            pixelType="Scalar",
            components=1,
        ),
        name="test",
        origin=[0.0, 0.0, 0.0],
        spacing=[1.0, 1.0, 1.0],
        direction=np.eye(3, dtype=np.float64).flatten(),
        size=[4, 8, 8],
        data=np.zeros((8, 8, 4), dtype=np.uint8),
    )

    ngff = itk_image_to_ngff_image(img)
    itk_back = ngff_image_to_itk_image(ngff)

    assert np.array_equal(np.asarray(itk_back.direction).reshape(3, 3), np.eye(3))


def test_direction_round_trip_flipped_y():
    """Round-trip ITK -> NgffImage -> ITK preserves flipped-Y direction."""
    original_direction = np.diag([1.0, -1.0, 1.0])
    img = itkwasm.Image(
        imageType=itkwasm.ImageType(
            dimension=3,
            componentType="uint8",
            pixelType="Scalar",
            components=1,
        ),
        name="test",
        origin=[0.0, 0.0, 0.0],
        spacing=[1.0, 1.0, 1.0],
        direction=original_direction.flatten(),
        size=[4, 8, 8],
        data=np.zeros((8, 8, 4), dtype=np.uint8),
    )

    ngff = itk_image_to_ngff_image(img)
    itk_back = ngff_image_to_itk_image(ngff)

    assert np.array_equal(
        np.asarray(itk_back.direction).reshape(3, 3), original_direction
    )
