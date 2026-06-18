# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from ngff_zarr.to_multiscales import _ngff_image_scale_factors
from ngff_zarr.to_ngff_image import to_ngff_image

rng = np.random.default_rng(12345)


@pytest.mark.parametrize(
    "shape, expected_factors",
    [
        ((30, 30), []),
        ((520, 520), [{"x": 2, "y": 2}, {"x": 4, "y": 4}, {"x": 8, "y": 8}]),
        (
            (10, 530, 530),
            [
                {"x": 2, "y": 2, "z": 1},
                {"x": 4, "y": 4, "z": 1},
                {"x": 8, "y": 8, "z": 1},
            ],
        ),
        # Image size exactly twice the chunk size should still downsample once.
        # Regression tests for
        # https://github.com/fideus-labs/ngff-zarr/issues/551
        # Cubic 3D at the exact 2x boundary.
        ((128, 128, 128), [{"x": 2, "y": 2, "z": 2}]),
        # 2D at the exact 2x boundary.
        ((128, 128), [{"x": 2, "y": 2}]),
        # Anisotropic: the max dim is exactly 2x the chunk while z sits at the
        # chunk size, so only y/x downsample.
        ((64, 128, 128), [{"x": 2, "y": 2, "z": 1}]),
    ],
)
def test_scale_factors(shape, expected_factors):
    array = rng.random(size=shape, dtype=np.float32) * 100.0
    image = to_ngff_image(array)
    chunk_length = 64
    image.data = image.data.rechunk(chunk_length)
    chunk_dims = dict.fromkeys(image.dims, chunk_length)
    scale_factors = _ngff_image_scale_factors(image, chunk_length, chunk_dims)
    assert scale_factors == expected_factors


@pytest.mark.parametrize(
    "shape, chunks, expected_factors",
    [
        (
            (1, 30, 1024, 1024),
            (1, 30, 65, 65),
            [
                {"x": 2, "y": 2, "z": 1},
                {"x": 4, "y": 4, "z": 1},
                {"x": 8, "y": 8, "z": 1},
            ],
        ),
        (
            (1, 125, 1024, 1024),
            (1, 50, 51, 50),
            [
                {"x": 2, "y": 2, "z": 1},
                {"x": 4, "y": 4, "z": 1},
                {"x": 8, "y": 8, "z": 1},
                {"x": 16, "y": 16, "z": 2},
            ],
        ),
    ],
)
def test_scale_factors_with_chunk_shape(shape, chunks, expected_factors):
    array = rng.random(size=shape, dtype=np.float32) * 100.0
    image = to_ngff_image(array, dims=["t", "z", "y", "x"])
    out_chunks = {d: chunks[i] for i, d in enumerate(image.dims)}
    scale_factors = _ngff_image_scale_factors(image, max(chunks), out_chunks)
    assert scale_factors == expected_factors
