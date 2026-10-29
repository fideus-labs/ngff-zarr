# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Checks run inside Pyodide by run.mjs; a failing assert fails the run."""

import sys

import ngff_zarr
import numpy as np
from dask.base import get_scheduler
from dask.local import get_sync
from ngff_zarr import to_ngff_image
from pyodide.ffi import can_run_sync

assert sys.platform == "emscripten", sys.platform
assert can_run_sync(), "JSPI unavailable: run with Node >= 25"

image = to_ngff_image(np.zeros((4, 8, 8), dtype=np.uint8))
assert image.dims == ("z", "y", "x"), image.dims
assert get_scheduler(collections=[image.data]) is get_sync
assert image.data.compute().shape == (4, 8, 8)
print(f"ngff_zarr {ngff_zarr.__version__} on Pyodide, Python {sys.version}")
