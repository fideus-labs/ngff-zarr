# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Code paths that must work with no zarrista installed, as on Pyodide."""

import sys

import numpy as np
import pytest
from ngff_zarr import to_ngff_image
from ngff_zarr._matrix_transform_arrays import resolve_matrix_transforms


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
