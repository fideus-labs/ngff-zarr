# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import os
import tempfile

import numpy as np
import pytest
import zarr
from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr
from packaging import version

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

        # Open the array directly by path — on zarr-python 3 the arrays are
        # written as zarr v3 format even for OME-Zarr 0.4 stores.
        arr_path = os.path.join(tmpdir, "scale0", "test_img")
        arr = zarr.open_array(store=arr_path, mode="r")
        encoding = arr.metadata.chunk_key_encoding
        assert encoding.separator == "/"
