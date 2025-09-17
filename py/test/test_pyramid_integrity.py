# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array as da
from ngff_zarr import Multiscales, to_multiscales, NgffImage, Methods


def test_check_pyramid():
    """
    When creating downscaled versions of images, subsequent resolutions were
    not correctly being calculated. This test checks that the image volume
    occupies the same space, and it checks that the scale factors are consistent
    with the provided scale factors.
    """
    data = da.zeros(shape=(512, 512))
    for method in Methods:
        image = NgffImage(
            data=data,
            dims=["y", "x"],
            scale={"y": 0.25, "x": 0.25},
            translation={"y": 0, "x": 0},
        )
        scale_factors = [2, 4, 8, 16, 32]
        multiscales: Multiscales = to_multiscales(
            image,
            scale_factors=scale_factors,
            method=method,
        )
        dims = image.data.shape

        for sf, image in enumerate(multiscales.images):
            if sf == 0:
                continue
            scales = [image.scale[k] for k in image.scale]
            for i, d in enumerate(dims):
                assert d * 0.25 == scales[i] * image.data.shape[i]
                assert scales[i] == scale_factors[sf - 1] * 0.25
