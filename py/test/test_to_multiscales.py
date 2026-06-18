# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from ngff_zarr.to_multiscales import to_multiscales
from ngff_zarr.to_ngff_image import to_ngff_image

rng = np.random.default_rng(12345)


@pytest.mark.parametrize(
    "shape, chunk_shape",
    [
        (
            (1, 30, 1024, 1024),
            (1, 30, 65, 65),
        ),
        (
            (1, 125, 1024, 1024),
            (1, 50, 51, 50),
        ),
    ],
)
def test_to_multiscales_metadata_synced_with_data(shape, chunk_shape):
    array = rng.random(size=shape, dtype=np.float32) * 100.0
    input_image = to_ngff_image(array, dims=["t", "z", "y", "x"])
    multiscales = to_multiscales(
        input_image, scale_factors=max(chunk_shape), chunks=chunk_shape
    )
    for i, dataset in enumerate(multiscales.metadata.datasets):
        image = multiscales.images[i]
        toplevel_meta_scale = dataset.coordinateTransformations[0].scale

        image_scale_spatial_only = [image.scale[d] for d in ["z", "y", "x"]]
        assert image_scale_spatial_only == toplevel_meta_scale[1:]

        # Assert scale factors are applied to the correct dimensions
        assert image.data.shape[2] == image.data.shape[3]  # 512 != 1024


def test_downsamples_when_size_is_exactly_double_chunk():
    # Regression test for
    # https://github.com/fideus-labs/ngff-zarr/issues/551
    # When the image size is exactly twice the chunk size, to_multiscales()
    # should still produce a downsampled level rather than only the original.
    array = rng.integers(low=0, high=2**16, size=(128, 128, 128), dtype=np.uint16)
    input_image = to_ngff_image(array)
    multiscales = to_multiscales(input_image, scale_factors=4, chunks=64)

    assert len(multiscales.images) == 2
    assert multiscales.images[0].data.shape == (128, 128, 128)
    assert multiscales.images[1].data.shape == (64, 64, 64)
