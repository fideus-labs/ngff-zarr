# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import platform

import dask.array as da
import pytest
from ngff_zarr import Methods, config, to_multiscales, to_ngff_image

from ._data import verify_against_baseline

_on_x86 = platform.machine().lower() in ("x86_64", "amd64")


@pytest.mark.skipif(
    not _on_x86,
    reason="baselines are generated on x86_64; scipy Gaussian "
    "floating point differs on other architectures",
)
def test_gaussian_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/DASK_IMAGE_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.DASK_IMAGE_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    baseline_name = "auto/DASK_IMAGE_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, method=Methods.DASK_IMAGE_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_gaussian_isotropic_scale_factors_two_components(input_images):
    dataset_name = "brain_two_components"
    image = input_images[dataset_name]
    baseline_name = "2_4/DASK_IMAGE_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.DASK_IMAGE_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_non_spatial_dims_with_auto_scale_factors():
    """Regression test for issue #328.

    Tests that to_multiscales() handles non-spatial dimensions (e.g., 'c')
    correctly when using auto scale factors and the large image caching path.
    The issue occurred when dict-style scale factors introduced non-spatial
    dims into dim_factors.
    """
    # Create a dummy array with channel dimension as in issue #328
    dummy_array = da.random.random((4, 15000, 15000), chunks=(1, 1000, 1000))

    ngff_image = to_ngff_image(
        data=dummy_array,
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": 0.65, "x": 0.65},
        axes_units={"x": "micrometer", "y": "micrometer"},
    )

    # This should complete without KeyError for non-spatial dims
    # Using default (auto) scale_factors triggers the code path that had the bug
    ngff_multiscales = to_multiscales(
        data=ngff_image,
        cache=False,  # Test without caching first
        method=Methods.DASK_IMAGE_GAUSSIAN,
    )

    # Verify that we got multiple scales
    assert len(ngff_multiscales.images) > 1

    # Verify that all scales have the correct dims and shapes
    original_shape = (4, 15000, 15000)
    for i, scale_image in enumerate(ngff_multiscales.images):
        assert scale_image.dims == ("c", "y", "x")
        # Verify that non-spatial scale is preserved
        assert scale_image.scale["c"] == 1.0
        # Verify that spatial scales are increasing
        assert scale_image.scale["y"] >= 0.65
        assert scale_image.scale["x"] >= 0.65

        # Verify shapes: channel dim should not change, spatial dims should decrease
        c_dim, y_dim, x_dim = scale_image.data.shape
        assert c_dim == original_shape[0], (
            f"Channel dimension should not change at scale {i}"
        )
        assert y_dim <= original_shape[1], (
            f"Y dimension should decrease or stay same at scale {i}"
        )
        assert x_dim <= original_shape[2], (
            f"X dimension should decrease or stay same at scale {i}"
        )


def test_non_spatial_dims_with_caching():
    """Regression test for issue #328 with caching enabled.

    Tests that the large image serialization path (enabled by cache=True
    or low memory_target) correctly handles non-spatial dimensions.
    """
    # Create a smaller array but force caching with low memory target
    dummy_array = da.random.random((2, 512, 512), chunks=(1, 128, 128))

    ngff_image = to_ngff_image(
        data=dummy_array,
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": 1.0, "x": 1.0},
    )

    # Save original memory target
    original_memory_target = config.memory_target
    try:
        # Set a very low memory target to force the large image path
        config.memory_target = 1024  # 1KB to force caching path

        ngff_multiscales = to_multiscales(
            data=ngff_image, scale_factors=[2, 4], method=Methods.DASK_IMAGE_GAUSSIAN
        )

        # Verify that we got the expected scales
        assert len(ngff_multiscales.images) == 3  # original + 2 downsampled

        # Verify metadata preservation and shapes
        expected_shapes = [(2, 512, 512), (2, 256, 256), (2, 128, 128)]
        for i, (scale_image, expected_shape) in enumerate(
            zip(ngff_multiscales.images, expected_shapes)
        ):
            assert scale_image.dims == ("c", "y", "x")
            assert scale_image.scale["c"] == 1.0
            # Verify shape
            actual_shape = scale_image.data.shape
            assert actual_shape == expected_shape, (
                f"Scale {i}: expected {expected_shape}, got {actual_shape}"
            )
    finally:
        # Restore original memory target
        config.memory_target = original_memory_target


def test_non_spatial_translation_preservation():
    """Test that non-spatial translations are preserved during downsampling."""
    dummy_array = da.random.random((3, 256, 256), chunks=(1, 128, 128))

    ngff_image = to_ngff_image(
        data=dummy_array,
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": 1.0, "x": 1.0},
        translation={"c": 5.0, "y": 10.0, "x": 20.0},
    )

    ngff_multiscales = to_multiscales(
        data=ngff_image, scale_factors=[2], method=Methods.DASK_IMAGE_GAUSSIAN
    )

    # Check that non-spatial translation is preserved
    for scale_image in ngff_multiscales.images:
        assert scale_image.translation["c"] == 5.0
        # Spatial translations may change
        assert "y" in scale_image.translation
        assert "x" in scale_image.translation


#     dataset_name = "cthead1"
#     image = input_images[dataset_name]
#     baseline_name = "2_3/DASK_IMAGE_GAUSSIAN"
#     multiscale = to_multiscale(image, [2, 3], method=Methods.DASK_IMAGE_GAUSSIAN)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)

#     dataset_name = "small_head"
#     image = input_images[dataset_name]
#     baseline_name = "2_3_4/DASK_IMAGE_GAUSSIAN"
#     multiscale = to_multiscale(image, [2, 3, 4], method=Methods.DASK_IMAGE_GAUSSIAN)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)


# def test_gaussian_anisotropic_scale_factors(input_images):
#     dataset_name = "cthead1"
#     image = input_images[dataset_name]
#     scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
#     multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_GAUSSIAN)
#     baseline_name = "x2y4_x1y2/DASK_IMAGE_GAUSSIAN"
#     verify_against_baseline(dataset_name, baseline_name, multiscale)

#     dataset_name = "small_head"
#     image = input_images[dataset_name]
#     scale_factors = [
#         {"x": 3, "y": 2, "z": 4},
#         {"x": 2, "y": 2, "z": 2},
#         {"x": 1, "y": 2, "z": 1},
#     ]
#     multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_GAUSSIAN)
#     baseline_name = "x3y2z4_x2y2z2_x1y2z1/DASK_IMAGE_GAUSSIAN"
#     verify_against_baseline(dataset_name, baseline_name, multiscale)


# def test_label_nearest_isotropic_scale_factors(input_images):
#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     baseline_name = "2_4/DASK_IMAGE_NEAREST"
#     multiscale = to_multiscale(image, [2, 4], method=Methods.DASK_IMAGE_NEAREST)
#     store_new_image(dataset_name, baseline_name, multiscale)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)

#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     baseline_name = "2_3/DASK_IMAGE_NEAREST"
#     multiscale = to_multiscale(image, [2, 3], method=Methods.DASK_IMAGE_NEAREST)
#     store_new_image(dataset_name, baseline_name, multiscale)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)


# def test_label_nearest_anisotropic_scale_factors(input_images):
#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
#     multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_NEAREST)
#     baseline_name = "x2y4_x1y2/DASK_IMAGE_NEAREST"
#     store_new_image(dataset_name, baseline_name, multiscale)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)


# def test_label_mode_isotropic_scale_factors(input_images):
#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     baseline_name = "2_4/DASK_IMAGE_MODE"
#     multiscale = to_multiscale(image, [2, 4], method=Methods.DASK_IMAGE_MODE)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)

#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     baseline_name = "2_3/DASK_IMAGE_MODE"
#     multiscale = to_multiscale(image, [2, 3], method=Methods.DASK_IMAGE_MODE)
#     verify_against_baseline(dataset_name, baseline_name, multiscale)


# def test_label_mode_anisotropic_scale_factors(input_images):
#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     scale_factors = [{"x": 2, "y": 4}, {"x": 1, "y": 2}]
#     multiscale = to_multiscale(image, scale_factors, method=Methods.DASK_IMAGE_MODE)
#     baseline_name = "x2y4_x1y2/DASK_IMAGE_MODE"
#     verify_against_baseline(dataset_name, baseline_name, multiscale)
