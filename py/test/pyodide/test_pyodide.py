# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Checks run inside Pyodide by run.mjs; a failing assert fails the run."""

import hashlib
import os
import shutil
import sys
from pathlib import Path

import ngff_zarr
import numpy as np
import zarr
from dask.base import get_scheduler
from dask.local import get_sync
from itkwasm.pyodide import JsPackageConfig
from itkwasm_downsample_emscripten.js_package import js_package
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
from ngff_zarr._js_zarr import JsZarrStore, create_js_array, open_js_array, open_js_lazy
from pyodide.ffi import can_run_sync

assert sys.platform == "emscripten", sys.platform
assert can_run_sync(), "JSPI unavailable: run with Node >= 25"

image = to_ngff_image(np.zeros((4, 8, 8), dtype=np.uint8))
assert image.dims == ("z", "y", "x"), image.dims
assert get_scheduler(collections=[image.data]) is get_sync
assert image.data.compute().shape == (4, 8, 8)

# Phase 2: the zarrita bridge against stores zarr-python wrote, and vice versa.
root = Path("/tmp/js_zarr")
shutil.rmtree(root, ignore_errors=True)
data = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)

z = zarr.create_array(
    str(root / "in"),
    name="0",
    shape=data.shape,
    dtype=data.dtype,
    chunks=(2, 3, 4),
    shards=(4, 6, 8),
)
z[:] = data
store = JsZarrStore(root / "in")
arr = open_js_array(store, "0")
assert arr.shape == data.shape
assert arr.dtype == data.dtype
assert arr.chunks == (2, 3, 4)
assert arr.write_chunks == (4, 6, 8)
assert arr.is_sharded
np.testing.assert_array_equal(arr[:], data)
np.testing.assert_array_equal(arr[1:3, ::2, 5], data[1:3, ::2, 5])
np.testing.assert_array_equal(arr[3, 1:4, ::-1], data[3, 1:4, ::-1])
lazy = open_js_lazy(store.with_prefix("0"))
assert lazy.chunksize == (2, 3, 4)
np.testing.assert_array_equal(lazy.compute(), data)

for dtype in ("uint8", "int64", "float32", "bool"):
    values = (data % 7 if dtype != "bool" else data % 2).astype(dtype)
    for fmt, kwargs in (
        (3, {"chunks_per_shard": 2, "dimension_names": ("z", "y", "x")}),
        (3, {}),
        (2, {}),
    ):
        if dtype == "bool" and kwargs:
            # shortcut: fizarrita's workers have no bool codec and zarrita's
            # own set() refuses sharded arrays, so sharded bool arrays cannot
            # be written on Pyodide; drop this skip once fizarrita adds bool.
            continue
        name = f"{dtype}_v{fmt}_{len(kwargs)}"
        out = create_js_array(
            root / "out",
            name,
            values.shape,
            dtype,
            (2, 3, 4),
            fmt,
            compressor={"id": "blosc"},
            **kwargs,
        )
        out[:2] = values[:2]
        out[2:] = values[2:]
        out[:, 0] = values[:, 0]  # integer index: the value drops that axis
        out[0, 0, :4] = values[3, 5, 4:]
        expected = values.copy()
        expected[0, 0, :4] = values[3, 5, 4:]
        back = zarr.open_array(str(root / "out" / name), mode="r")[:]
        np.testing.assert_array_equal(back, expected)
        np.testing.assert_array_equal(
            open_js_array(JsZarrStore(root / "out"), name)[:], expected
        )

# Phase 3 + 4: the public API end to end, itkwasm downsampling included. /out
# is a host directory run.mjs also serves over HTTP, so the stores come back
# three ways: local path, URL, and zarr-python as the independent reader.
out = Path("/out")
http_base = os.environ["NGFF_ZARR_HTTP_BASE"]
# The itkwasm wrapper imports the package's browser build; Node runs its Node
# build instead through the shim run.mjs names here.
js_package.config = JsPackageConfig(os.environ["NGFF_ZARR_ITKWASM_MODULE"])
volume = np.random.default_rng(0).integers(0, 4096, (64, 64, 64), dtype=np.uint16)
image = to_ngff_image(
    volume,
    dims=("z", "y", "x"),
    scale={"z": 2.0, "y": 0.5, "x": 0.5},
    translation={"z": 1.0, "y": 0.0, "x": 0.0},
)
multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)
levels = [level.data.compute() for level in multiscales.images]
assert [level.shape for level in levels] == [(64,) * 3, (32,) * 3, (16,) * 3]
# The same pyramid from the WASI build: test_js_zarr.py::PYRAMID_DIGESTS.
digests = [hashlib.sha256(level.tobytes()).hexdigest()[:16] for level in levels]
assert digests == ["257df0ff4ff55204", "7bd266d79dec4343", "fd60f9720d24832c"], digests


def check(read):
    first = read.images[0]
    assert first.dims == ("z", "y", "x"), first.dims
    assert first.scale == {"z": 2.0, "y": 0.5, "x": 0.5}, first.scale
    assert first.translation == {"z": 1.0, "y": 0.0, "x": 0.0}, first.translation
    assert first.data.chunksize == (32, 32, 32), first.data.chunksize
    assert read.images[1].scale == {"z": 4.0, "y": 1.0, "x": 1.0}, read.images[1].scale
    for image, level in zip(read.images, levels, strict=True):
        np.testing.assert_array_equal(image.data.compute(), level)


for name, kwargs in (
    ("v04", {"version": "0.4"}),
    ("v05", {"version": "0.5"}),
    ("sharded", {"version": "0.5", "chunks_per_shard": 2}),
):
    path = out / f"{name}.ome.zarr"
    to_ome_zarr(path, multiscales, **kwargs)
    check(from_ome_zarr(path))
    check(from_ome_zarr(f"{http_base}/{name}.ome.zarr"))
    level0 = zarr.open_group(str(path), mode="r")[multiscales.metadata.datasets[0].path]
    np.testing.assert_array_equal(level0[:], volume)

to_ome_zarr(out / "img.ozx", multiscales, version="0.5")
check(from_ome_zarr(out / "img.ozx"))

print(f"ngff_zarr {ngff_zarr.__version__} on Pyodide, Python {sys.version}")
