# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path
import sys

import numpy as np
import zarr
from ngff_zarr import (
    Multiscales,
    to_multiscales,
    to_ngff_zarr,
    validate,
    from_ngff_zarr,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)
zarr_version_major = zarr_version.major


def check_valid_ngff(multiscale: Multiscales):
    store = zarr.storage.MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscale, version=version)
    format_kwargs = {}
    if version and zarr_version_major >= 3:
        format_kwargs = {"zarr_format": 2} if version == "0.4" else {"zarr_format": 3}
    root = zarr.open_group(store, mode="r", **format_kwargs)

    validate(root.attrs.asdict())
    # Need to add NGFF metadata property
    # validate(ngff, strict=True)

    from_ngff_zarr(store, validate=True, version=version)


def test_y_x_valid_ngff():
    array = np.random.random((32, 16))
    multiscale = to_multiscales(array, [2, 4])

    check_valid_ngff(multiscale)


def test_validate_0_1():
    test_store = Path(__file__).parent / "data" / "input" / "v01" / "6001251.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.1")
    if sys.byteorder == "little":
        assert multiscales.images[0].data.dtype.byteorder == "<"
    else:
        assert multiscales.images[0].data.dtype.byteorder == ">"


def test_validate_0_1_no_version():
    test_store = Path(__file__).parent / "data" / "input" / "v01" / "6001251.zarr"
    from_ngff_zarr(test_store, validate=True, version="0.1")


def test_validate_0_2():
    test_store = Path(__file__).parent / "data" / "input" / "v02" / "6001240.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.2")
    print(multiscales)


def test_validate_0_2_no_version():
    test_store = Path(__file__).parent / "data" / "input" / "v02" / "6001240.zarr"
    from_ngff_zarr(test_store, validate=True)


def test_validate_0_3():
    test_store = Path(__file__).parent / "data" / "input" / "v03" / "9528933.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.3")
    print(multiscales)


def test_validate_0_3_no_version():
    test_store = Path(__file__).parent / "data" / "input" / "v03" / "9528933.zarr"
    from_ngff_zarr(test_store, validate=True)
