# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dask_image import imread
from ngff_zarr import config, to_multiscales, to_ngff_image, to_ngff_zarr
from zarr.storage import MemoryStore


def test_large_image_serialization(input_images):
    default_mem_target = config.memory_target
    config.memory_target = int(1e6)

    dataset_name = "lung_series"
    data = imread.imread(input_images[dataset_name])
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 2.5, "y": 1.40625, "x": 1.40625},
        translation={"z": 332.5, "y": 360.0, "x": 0.0},
        name="LIDC2",
    )
    multiscales = to_multiscales(image)
    # baseline_name = "auto/memory_target_1e6.zarr"
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    test_store = MemoryStore()
    to_ngff_zarr(test_store, multiscales)
    # verify_against_baseline(dataset_name, baseline_name, multiscales)

    config.memory_target = default_mem_target
