# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Regression tests for GH #577.

Issue: https://github.com/fideus-labs/ngff-zarr/issues/577

Accelerated itkwasm backends such as cuCIM can return CuPy arrays.  The
downsampling dask graph is NumPy-backed, so each backend result must be moved
to host memory before NumPy operations or dask block assembly consume it.
"""

from types import SimpleNamespace

import itkwasm
import itkwasm_downsample
import numpy as np
from ngff_zarr.methods._itkwasm import (
    _itkwasm_blur_and_downsample,
    _itkwasm_chunk_bin_shrink,
)


def test_itkwasm_gaussian_normalizes_accelerator_output(monkeypatch):
    accelerator_data = object()
    host_data = np.full((4, 4), 2.4, dtype=np.float32)
    normalized = []

    def fake_downsample(*args, **kwargs):
        return SimpleNamespace(data=accelerator_data)

    def fake_to_numpy(data):
        normalized.append(data)
        return host_data

    monkeypatch.setattr(itkwasm_downsample, "downsample", fake_downsample)
    monkeypatch.setattr(itkwasm, "array_like_to_numpy_array", fake_to_numpy)

    result = _itkwasm_blur_and_downsample(
        np.full((8, 8), 2, dtype=np.uint8),
        shrink_factors=[2, 2],
        kernel_radius=[0, 0],
        smoothing="gaussian",
    )

    assert normalized == [accelerator_data]
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.full((4, 4), 2, dtype=np.uint8))


def test_itkwasm_bin_shrink_normalizes_accelerator_output(monkeypatch):
    accelerator_data = object()
    host_data = np.arange(16, dtype=np.uint8).reshape((4, 4))
    normalized = []

    def fake_downsample(*args, **kwargs):
        return SimpleNamespace(data=accelerator_data)

    def fake_to_numpy(data):
        normalized.append(data)
        return host_data

    monkeypatch.setattr(itkwasm_downsample, "downsample_bin_shrink", fake_downsample)
    monkeypatch.setattr(itkwasm, "array_like_to_numpy_array", fake_to_numpy)

    result = _itkwasm_chunk_bin_shrink(
        np.ones((8, 8), dtype=np.uint8),
        shrink_factors=[2, 2],
    )

    assert normalized == [accelerator_data]
    assert result is host_data
