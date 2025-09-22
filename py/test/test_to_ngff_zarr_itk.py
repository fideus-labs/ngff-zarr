# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from ngff_zarr import Methods, to_multiscales, to_ngff_image

from ._data import verify_against_baseline
import platform


def test_bin_shrink_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITK_BIN_SHRINK.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITK_BIN_SHRINK)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    baseline_name = "auto/ITK_BIN_SHRINK.zarr"
    multiscales = to_multiscales(image, method=Methods.ITK_BIN_SHRINK)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_bin_shrink_tzyxc():
    import dask.array as da

    test_array = da.ones((96, 64, 64, 64, 2), chunks=(1, 64, 64, 64, 1), dtype="uint8")
    img = to_ngff_image(
        test_array,
        dims=list("tzyxc"),
        scale={
            "t": 100_000.0,
            "z": 0.98,
            "y": 0.98,
            "x": 0.98,
            "c": 1.0,
        },
        axes_units={
            "t": "millisecond",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        },
        name="000x_000y_000z",
    )

    # expect a ValueError
    try:
        to_multiscales(
            img,
            scale_factors=[{"z": 2, "y": 2, "x": 2}, {"z": 4, "y": 4, "x": 4}],
            method=Methods.ITK_BIN_SHRINK,
        )
        assert False, "Expected ValueError for non-spatial dimensions"
    except ValueError:
        pass


def test_gaussian_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITK_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITK_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    baseline_name = "auto/ITK_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, method=Methods.ITK_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)


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
