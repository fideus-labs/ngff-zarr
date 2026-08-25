# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""TIFF inputs must convert without zarr-python.

tifffile's ``aszarr`` stores import zarr-python (>= 3), so the TIFF paths
build lazy dask arrays from TIFF page primitives instead
(``_series_level_to_dask``). These tests pin that the conversion never
imports zarr and that concurrent per-page chunk reads are race-free.
"""

import subprocess
import sys

import numpy as np
import pytest

_NO_ZARR_SCRIPT = """
import sys
import importlib.abc


class _ZarrBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "zarr" or fullname.startswith("zarr."):
            raise ImportError("zarr import blocked by test")
        return None


sys.meta_path.insert(0, _ZarrBlocker())

import numpy as np
import tifffile

tiff_path = sys.argv[1]
vol = np.arange(5 * 16 * 16, dtype="uint16").reshape(5, 16, 16)
tifffile.imwrite(tiff_path, vol, metadata={"axes": "ZYX"})

from ngff_zarr import tiff_file_to_ngff_images

((name, image),) = tiff_file_to_ngff_images(tiff_path)
np.testing.assert_array_equal(np.asarray(image.data), vol)
assert "zarr" not in sys.modules
"""


def test_tiff_conversion_does_not_import_zarr(tmp_path):
    """The whole TIFF read path works with zarr imports blocked."""
    pytest.importorskip("tifffile")

    result = subprocess.run(
        [sys.executable, "-c", _NO_ZARR_SCRIPT, str(tmp_path / "vol.tif")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_multi_page_tiff_threaded_reads(tmp_path):
    """Per-page chunks read correctly under the threaded scheduler.

    ``tiff_file_to_ngff_images`` closes the TIFF before the data is
    computed, so this also covers reopening the file handle on first read.
    """
    tifffile = pytest.importorskip("tifffile")

    from ngff_zarr import tiff_file_to_ngff_images

    rng = np.random.default_rng(7)
    vol = rng.integers(0, 255, size=(16, 64, 64), dtype=np.uint8)
    tiff_path = tmp_path / "vol.tif"
    tifffile.imwrite(tiff_path, vol, metadata={"axes": "ZYX"})

    ((_, image),) = tiff_file_to_ngff_images(tiff_path)
    # One chunk per stored page so each chunk decodes a single plane.
    assert image.data.chunks[0] == (1,) * 16
    # Concurrent chunk reads must serialize on the file handle lock;
    # repeat to give a race a chance to surface.
    for _ in range(10):
        np.testing.assert_array_equal(image.data.compute(scheduler="threads"), vol)
