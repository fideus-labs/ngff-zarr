# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest
import zarr
import zarr.storage
from ngff_zarr import from_ome_zarr, to_ome_zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


@pytest.mark.parametrize(
        "input_version, output_version",
        [
            ("0.4", "0.5"),
            ("0.5", "0.4"),
            ("0.5", "0.6"),
            ("0.6", "0.5"),
        ],
)
def test_conversion(input_version, output_version):
    test_store = Path(__file__).parent / "data" / "input" / "v04" / "6001240.zarr"
    multiscales = from_ome_zarr(test_store, validate=True, version="0.4")

    store = zarr.storage.MemoryStore()
    to_ome_zarr(store, multiscales, version=input_version)
    from_ome_zarr(store, validate=True, version=input_version)

    new_store = zarr.storage.MemoryStore()
    to_ome_zarr(new_store, multiscales, version=output_version)
    from_ome_zarr(new_store, validate=True, version=output_version)


if __name__ == "__main__":
    pytest.main([__file__])