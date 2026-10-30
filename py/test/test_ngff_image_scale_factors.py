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


@pytest.mark.parametrize(
    "scale, expected_factors",
    [
        # Isotropic: every axis halves together, as before.
        (
            {"z": 1.0, "y": 1.0, "x": 1.0},
            [{"x": 2, "y": 2, "z": 2}, {"x": 4, "y": 4, "z": 4}],
        ),
        # Light-sheet: z is already about 3x coarser, so it waits until y/x
        # catch up instead of keeping the anisotropy at every level.
        (
            {"z": 2.031, "y": 0.65, "x": 0.65},
            [
                {"x": 2, "y": 2, "z": 1},
                {"x": 4, "y": 4, "z": 2},
                # y/x reached the chunk size; z goes on alone.
                {"x": 4, "y": 4, "z": 4},
            ],
        ),
    ],
)
def test_scale_factors_follow_the_spacing(scale, expected_factors):
    array = rng.random(size=(256, 256, 256), dtype=np.float32)
    image = to_ngff_image(array, dims=["z", "y", "x"], scale=scale)
    chunk_dims = dict.fromkeys(image.dims, 64)
    assert _ngff_image_scale_factors(image, 64, chunk_dims) == expected_factors


def test_a_coarse_axis_halves_alone_when_the_finer_ones_are_at_their_chunk():
    # z is 4x coarser, so the spacing rule holds it back; y and x are already
    # one chunk long. Without z going on alone there would be no level at all.
    array = rng.random(size=(256, 64, 64), dtype=np.float32)
    image = to_ngff_image(
        array, dims=["z", "y", "x"], scale={"z": 4.0, "y": 1.0, "x": 1.0}
    )
    chunk_dims = dict.fromkeys(image.dims, 64)
    assert _ngff_image_scale_factors(image, 64, chunk_dims)[0] == {
        "z": 2,
        "y": 1,
        "x": 1,
    }


def test_scale_factors_compare_the_spacing_in_one_unit():
    # 2 micrometer along z is coarser than 650 nanometer along x: on the raw
    # numbers z looked 300x finer, and was halved first.
    array = rng.random(size=(256, 256, 256), dtype=np.float32)
    image = to_ngff_image(
        array,
        dims=["z", "y", "x"],
        scale={"z": 2.031, "y": 650.0, "x": 650.0},
        axes_units={"z": "micrometer", "y": "nanometer", "x": "nanometer"},
    )
    chunk_dims = dict.fromkeys(image.dims, 64)
    assert _ngff_image_scale_factors(image, 64, chunk_dims)[0] == {
        "x": 2,
        "y": 2,
        "z": 1,
    }
