# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
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

        # Verify dimension_separator via the zarr API (works across zarr versions)
        arr = zarr.open_array(
            store=tmpdir, path="scale0/test_img", mode="r", zarr_format=2
        )
        if hasattr(arr.metadata, "dimension_separator"):
            # zarr-python 2.x
            assert arr.metadata.dimension_separator == "/"
        else:
            # zarr-python 3.x: chunk_key_encoding with separator
            encoding = arr.metadata.chunk_key_encoding
            assert encoding.separator == "/"
