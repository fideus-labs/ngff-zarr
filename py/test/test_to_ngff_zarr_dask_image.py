# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from ngff_zarr import Methods, to_multiscales

from ._data import verify_against_baseline


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
