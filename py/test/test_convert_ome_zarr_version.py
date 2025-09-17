# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from packaging import version
from pathlib import Path

import pytest

import zarr.storage
import zarr

from ngff_zarr import to_ngff_zarr, from_ngff_zarr


zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def test_convert_0_4_to_0_5():
    test_store = Path(__file__).parent / "data" / "input" / "v04" / "6001240.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.4")
    store = zarr.storage.MemoryStore()
    version = "0.5"
    to_ngff_zarr(store, multiscales, version=version)
    from_ngff_zarr(store, validate=True, version=version)


def test_convert_0_5_to_0_4():
    test_store = Path(__file__).parent / "data" / "input" / "v04" / "6001240.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.4")
    store = zarr.storage.MemoryStore()
    version = "0.5"
    to_ngff_zarr(store, multiscales, version=version)
    multiscales = from_ngff_zarr(store, validate=True, version=version)

    version = "0.4"
    new_store = zarr.storage.MemoryStore()
    to_ngff_zarr(new_store, multiscales, version=version)
    from_ngff_zarr(new_store, validate=True, version=version)
