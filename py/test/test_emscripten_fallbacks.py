# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Code paths that must work with no zarrista installed, as on Pyodide."""

import importlib
import sys

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import NgffImage, resample, to_ngff_image
from ngff_zarr._matrix_transform_arrays import resolve_matrix_transforms
from ngff_zarr.v06.zarr_metadata import Translation


def test_to_ngff_image_does_not_import_zarrista_on_emscripten(monkeypatch):
    monkeypatch.setattr("ngff_zarr._zarrista_utils._USE_JS", True)
    monkeypatch.setitem(sys.modules, "zarrista", None)

    image = to_ngff_image(np.zeros((4, 8, 8), dtype=np.uint8))

    assert image.data.compute().shape == (4, 8, 8)


def test_an_unreadable_matrix_is_reported_without_zarrista(monkeypatch):
    def unreadable(*_):
        raise OSError("unreadable")

    monkeypatch.setitem(sys.modules, "zarrista.exceptions", None)
    monkeypatch.setattr("ngff_zarr._zarrista_utils.open_lazy_array", unreadable)

    with pytest.raises(ValueError, match="'params/r', which could not be read"):
        resolve_matrix_transforms([{"type": "rotation", "path": "params/r"}], "s")


def test_resample_sizes_its_blocks_without_threads_on_emscripten(monkeypatch):
    pytest.importorskip("itkwasm_downsample")
    values = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    image = NgffImage(
        data=da.from_array(values, chunks=(16, 16)),
        dims=("y", "x"),
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    transform = Translation(translation=[2.0, -3.0])
    expected = np.asarray(resample(transform, image, image).data)

    class NoThreads:
        def __init__(self, *_args):
            raise RuntimeError("can't start new thread")

    # The package re-exports the function under the module's name.
    module = importlib.import_module("ngff_zarr.resample")
    monkeypatch.setattr(module, "_USE_JS", True)
    monkeypatch.setattr(module, "ThreadPoolExecutor", NoThreads)

    result = resample(transform, image, image)
    np.testing.assert_array_equal(np.asarray(result.data), expected)
