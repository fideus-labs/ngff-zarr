# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
from zarr.storage import MemoryStore

from ngff_zarr import (
    Methods,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
    cli_input_to_ngff_image,
    ConversionBackend,
)

from ._data import verify_against_baseline, test_data_dir

_HAVE_CUCIM = False
try:
    import itkwasm_downsample_cucim  # noqa: F401

    _HAVE_CUCIM = True
except ImportError:
    pass


def test_downsample_czyx():
    data = np.random.randint(0, 256, 262144).reshape((2, 32, 64, 64)).astype(np.uint8)
    image = to_ngff_image(data, dims=["c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "c"
    assert multiscales.images[1].data.shape[0] == 2
    assert multiscales.images[1].data.shape[1] == 16


def test_downsample_zycx():
    data = np.random.randint(0, 256, 262144).reshape((32, 64, 2, 64)).astype(np.uint8)
    image = to_ngff_image(data, dims=["z", "y", "c", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "z"
    assert multiscales.images[0].dims[2] == "c"
    assert multiscales.images[1].data.shape[0] == 16
    assert multiscales.images[1].data.shape[2] == 2


def test_downsample_cxyz():
    data = np.random.randint(0, 256, 262144).reshape((2, 64, 64, 32)).astype(np.uint8)
    image = to_ngff_image(data, dims=["c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "c"
    assert multiscales.images[1].data.shape[0] == 2
    assert multiscales.images[1].data.shape[1] == 32


def test_downsample_tczyx():
    data = (
        np.random.randint(0, 256, 524288).reshape((2, 2, 32, 64, 64)).astype(np.uint8)
    )
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "t"
    assert multiscales.images[0].dims[1] == "c"
    assert multiscales.images[1].data.shape[0] == 2
    assert multiscales.images[1].data.shape[1] == 2
    assert multiscales.images[1].data.shape[2] == 16


def test_downsample_tzycx():
    data = np.random.randint(0, 256, 524288).reshape((2, 32, 64, 2, 64))
    image = to_ngff_image(data, dims=["t", "z", "y", "c", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "t"
    assert multiscales.images[0].dims[1] == "z"
    assert multiscales.images[0].dims[3] == "c"
    assert multiscales.images[1].data.shape[0] == 2
    assert multiscales.images[1].data.shape[1] == 16
    assert multiscales.images[1].data.shape[3] == 2


def test_downsample_tcxyz():
    data = np.random.randint(0, 256, 524288).reshape((2, 2, 64, 64, 32))
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "t"
    assert multiscales.images[0].dims[1] == "c"
    assert multiscales.images[1].data.shape[0] == 2
    assert multiscales.images[1].data.shape[1] == 2
    assert multiscales.images[1].data.shape[2] == 32


def test_bin_shrink_tczyx():
    data = (
        np.random.randint(0, 256, 524288).reshape((2, 2, 32, 64, 64)).astype(np.uint8)
    )
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[2, 4], chunks=32, method=Methods.ITKWASM_BIN_SHRINK
    )
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)
    assert multiscales.images[0].dims[0] == "t"
    assert multiscales.images[0].dims[1] == "c"
    assert multiscales.images[1].data.shape[0] == 2
    assert multiscales.images[1].data.shape[1] == 2
    assert multiscales.images[1].data.shape[2] == 16


def test_bin_shrink_tzyxc():
    import dask.array as da

    test_array = da.ones(
        (96, 256, 256, 256, 2), chunks=(1, 128, 128, 128, 1), dtype="uint16"
    )
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

    multiscales = to_multiscales(
        img,
        scale_factors=[{"z": 2, "y": 2, "x": 2}, {"z": 4, "y": 4, "x": 4}],
        method=Methods.ITKWASM_BIN_SHRINK,
    )
    assert len(multiscales.images) == 3


def test_bin_shrink_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    if _HAVE_CUCIM:
        baseline_name = "2_4/ITKWASM_BIN_SHRINK_CUCIM.zarr"
    else:
        baseline_name = "2_4/ITKWASM_BIN_SHRINK.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_BIN_SHRINK)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    if _HAVE_CUCIM:
        baseline_name = "auto/ITKWASM_BIN_SHRINK_CUCIM.zarr"
    else:
        baseline_name = "auto/ITKWASM_BIN_SHRINK.zarr"
    multiscales = to_multiscales(image, method=Methods.ITKWASM_BIN_SHRINK)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_gaussian_isotropic_scale_factors(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    if _HAVE_CUCIM:
        baseline_name = "2_4/ITKWASM_GAUSSIAN_CUCIM.zarr"
    else:
        baseline_name = "2_4/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    if _HAVE_CUCIM:
        baseline_name = "auto/ITKWASM_GAUSSIAN_CUCIM.zarr"
    else:
        baseline_name = "auto/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, method=Methods.ITKWASM_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    if _HAVE_CUCIM:
        baseline_name = "2_3/ITKWASM_GAUSSIAN_CUCIM.zarr"
    else:
        baseline_name = "2_3/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 3], method=Methods.ITKWASM_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)

    dataset_name = "MR-head"
    image = input_images[dataset_name]
    if _HAVE_CUCIM:
        baseline_name = "2_3_4/ITKWASM_GAUSSIAN_CUCIM.zarr"
    else:
        baseline_name = "2_3_4/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 3, 4], method=Methods.ITKWASM_GAUSSIAN)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales)


# def test_gaussian_anisotropic_scale_factors(input_images):
#     dataset_name = "cthead1"
#     image = input_images[dataset_name]
#     scale_factors = [{"x": 2, "y": 2}, {"x": 2, "y": 4}]
#     multiscales = to_multiscales(image, scale_factors, method=Methods.ITKWASM_GAUSSIAN)
#     baseline_name = "x2y2_x2y4/ITKWASM_GAUSSIAN.zarr"
#     verify_against_baseline(dataset_name, baseline_name, multiscales)

# dataset_name = "MR-head"
# image = input_images[dataset_name]
# scale_factors = [
#     {"x": 3, "y": 2, "z": 4},
#     {"x": 6, "y": 4, "z": 4},
#     {"x": 6, "y": 8, "z": 4},
# ]
# multiscales = to_multiscales(image, scale_factors, method=Methods.ITKWASM_GAUSSIAN)
# baseline_name = "x3y2z4_6242z4_x6y8z4/ITKWASM_GAUSSIAN.zarr"
# verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_label_image_isotropic_scale_factors(input_images):
    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITKWASM_LABEL_IMAGE.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_LABEL_IMAGE)
    version = "0.4"
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales, version=version)

    dataset_name = "2th_cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_3/ITKWASM_LABEL_IMAGE.zarr"
    multiscales = to_multiscales(image, [2, 3], method=Methods.ITKWASM_LABEL_IMAGE)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(dataset_name, baseline_name, multiscales, version=version)


# def test_label_image_anisotropic_scale_factors(input_images):
#     dataset_name = "2th_cthead1"
#     image = input_images[dataset_name]
#     scale_factors = [{"x": 2, "y": 2}, {"x": 2, "y": 4}]
#     multiscales = to_multiscales(
#         image, scale_factors, method=Methods.ITKWASM_LABEL_IMAGE
#     )
#     baseline_name = "x2y4_x2y4/ITKWASM_LABEL_IMAGE.zarr"
#     store_new_multiscales(dataset_name, baseline_name, multiscales)
#     verify_against_baseline(dataset_name, baseline_name, multiscales)


def test_multichannel_tiff(input_images):
    dataset_name = "ThreeChannels_Resolution_Level7.tif"
    dataset_path = test_data_dir / "input" / dataset_name
    image = cli_input_to_ngff_image(
        ConversionBackend.TIFFFILE,
        [
            dataset_path,
        ],
    )
    from rich import print

    print(image)
    multiscales = to_multiscales(
        image,
        [
            2,
        ],
        method=Methods.ITKWASM_GAUSSIAN,
    )
    print(multiscales)
    store = MemoryStore()
    to_ngff_zarr(store, multiscales)

    # baseline_name = "multichannel_tiff/ITKWASM_GAUSSIAN.zarr"
    # multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_GAUSSIAN)
    # # store_new_multiscales(dataset_name, baseline_name, multiscales)
    # verify_against_baseline(dataset_name, baseline_name, multiscales)
