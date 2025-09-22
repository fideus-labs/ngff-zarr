# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import json
from packaging import version
import tempfile

import pytest

import zarr
import numpy as np

from ngff_zarr import to_multiscales, to_ngff_zarr, to_ngff_image

zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def test_dimension_separator_0_4():
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    nz_img = to_ngff_image(data=img, dims=["y", "x"], name="test_img")
    ms = to_multiscales(
        data=nz_img,
        chunks=64,
    )

    version = "0.4"
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, ms, version=version)
        with open(tmpdir + "/.zmetadata") as f:
            zarr_json = json.load(f)
        assert zarr_json["metadata"][".zgroup"]["zarr_format"] == 2
        metadata = zarr_json["metadata"]
        separator = metadata["scale0/test_img/.zarray"]["dimension_separator"]
        assert separator == "/"
