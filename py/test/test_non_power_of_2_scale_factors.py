# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test for non-power-of-2 scale factors (GitHub issue #264)."""

import tempfile

import numpy as np
import pytest

import ngff_zarr as nz


def test_non_power_of_2_scale_factors():
    """
    Test that scale_factors that are not powers of 2 produce correct shapes.

    Regression test for https://github.com/fideus-labs/ngff-zarr/issues/264
    """
    data = np.random.randint(0, 256, size=(120, 120), dtype=np.uint8)
    image = nz.to_ngff_image(data, scale={"y": 1.0, "x": 1.0})

    multiscales = nz.to_multiscales(image, scale_factors=[2, 3, 4])

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test.ome.zarr"
        nz.to_ngff_zarr(zarr_path, multiscales)
        result = nz.from_ngff_zarr(zarr_path)

        # Original: 120x120
        assert result.images[0].data.shape == (120, 120)
        # Factor 2: 120/2 = 60
        assert result.images[1].data.shape == (60, 60)
        # Factor 3: 120/3 = 40
        assert result.images[2].data.shape == (
            40,
            40,
        ), f"Expected (40, 40) but got {result.images[2].data.shape}"
        # Factor 4: 120/4 = 30
        assert result.images[3].data.shape == (
            30,
            30,
        ), f"Expected (30, 30) but got {result.images[3].data.shape}"


@pytest.mark.parametrize(
    "scale_factors,expected_shapes",
    [
        ([2, 3, 4], [(120, 120), (60, 60), (40, 40), (30, 30)]),
        ([3, 6, 9], [(120, 120), (40, 40), (20, 20), (13, 13)]),
        ([2, 5, 10], [(120, 120), (60, 60), (24, 24), (12, 12)]),
        ([3], [(120, 120), (40, 40)]),
        ([5], [(120, 120), (24, 24)]),
    ],
)
def test_non_power_of_2_scale_factors_parametrized(scale_factors, expected_shapes):
    """
    Parametrized test for various non-power-of-2 scale factor combinations.

    Regression test for https://github.com/fideus-labs/ngff-zarr/issues/264
    """
    data = np.random.randint(0, 256, size=(120, 120), dtype=np.uint8)
    image = nz.to_ngff_image(data, scale={"y": 1.0, "x": 1.0})

    multiscales = nz.to_multiscales(image, scale_factors=scale_factors)

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test.ome.zarr"
        nz.to_ngff_zarr(zarr_path, multiscales)
        result = nz.from_ngff_zarr(zarr_path)

        for i, expected_shape in enumerate(expected_shapes):
            actual_shape = result.images[i].data.shape
            assert actual_shape == expected_shape, (
                f"Scale level {i} (factor={scale_factors[i - 1] if i > 0 else 1}): "
                f"expected {expected_shape}, got {actual_shape}"
            )


def test_non_power_of_2_scale_factors_3d():
    """
    Test non-power-of-2 scale factors with 3D data.

    Regression test for https://github.com/fideus-labs/ngff-zarr/issues/264
    """
    data = np.random.randint(0, 256, size=(60, 120, 120), dtype=np.uint8)
    image = nz.to_ngff_image(data, scale={"z": 2.0, "y": 1.0, "x": 1.0})

    multiscales = nz.to_multiscales(image, scale_factors=[2, 3, 6])

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test.ome.zarr"
        nz.to_ngff_zarr(zarr_path, multiscales)
        result = nz.from_ngff_zarr(zarr_path)

        # Original: (60, 120, 120)
        assert result.images[0].data.shape == (60, 120, 120)
        # Factor 2: (30, 60, 60)
        assert result.images[1].data.shape == (30, 60, 60)
        # Factor 3: (20, 40, 40)
        assert result.images[2].data.shape == (
            20,
            40,
            40,
        ), f"Expected (20, 40, 40) but got {result.images[2].data.shape}"
        # Factor 6: (10, 20, 20)
        assert result.images[3].data.shape == (
            10,
            20,
            20,
        ), f"Expected (10, 20, 20) but got {result.images[3].data.shape}"


def test_non_power_of_2_with_dict_scale_factors():
    """
    Test non-power-of-2 scale factors specified as per-dimension dicts.

    Regression test for https://github.com/fideus-labs/ngff-zarr/issues/264
    """
    data = np.random.randint(0, 256, size=(120, 90), dtype=np.uint8)
    image = nz.to_ngff_image(data, scale={"y": 1.0, "x": 1.0})

    # Different scale factors per dimension
    scale_factors = [
        {"y": 2, "x": 3},  # -> (60, 30)
        {"y": 3, "x": 9},  # -> (40, 10)
    ]
    multiscales = nz.to_multiscales(image, scale_factors=scale_factors)

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test.ome.zarr"
        nz.to_ngff_zarr(zarr_path, multiscales)
        result = nz.from_ngff_zarr(zarr_path)

        # Original: (120, 90)
        assert result.images[0].data.shape == (120, 90)
        # Factor {y: 2, x: 3}: (60, 30)
        assert result.images[1].data.shape == (
            60,
            30,
        ), f"Expected (60, 30) but got {result.images[1].data.shape}"
        # Factor {y: 3, x: 9}: (40, 10)
        assert result.images[2].data.shape == (
            40,
            10,
        ), f"Expected (40, 10) but got {result.images[2].data.shape}"
