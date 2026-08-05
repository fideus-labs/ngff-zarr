# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A missing baseline must never read as a passing comparison."""

import pytest
from ngff_zarr import Methods, to_multiscales, to_ngff_image

from ._data import store_equals, verify_against_baseline


def _tiny_multiscales():
    import numpy as np

    image = to_ngff_image(np.zeros((32, 32), dtype=np.uint8), dims=("y", "x"))
    return to_multiscales(image, [2], method=Methods.ITKWASM_GAUSSIAN)


def test_missing_baseline_skips_rather_than_passes():
    with pytest.raises(BaseException) as excinfo:
        verify_against_baseline(
            "does-not-exist", "no/such/baseline.zarr", _tiny_multiscales()
        )
    assert excinfo.typename == "Skipped", (
        f"a missing baseline must skip, got {excinfo.typename}"
    )


def test_store_equals_requires_identical_key_sets():
    """An empty baseline, or an extra key on either side, must not pass."""
    import zarr
    from ngff_zarr import to_ngff_zarr
    from zarr.storage import MemoryStore

    multiscales = _tiny_multiscales()
    populated = MemoryStore()
    to_ngff_zarr(populated, multiscales, version="0.4")

    assert not store_equals(MemoryStore(), populated), (
        "an empty baseline compared equal to a populated store"
    )

    test_store = MemoryStore()
    to_ngff_zarr(test_store, multiscales, version="0.4")
    assert store_equals(populated, test_store), (
        "two identical writes should compare equal"
    )

    zarr.create(shape=(1,), store=test_store, path="extra", zarr_format=2)
    assert not store_equals(populated, test_store), (
        "an extra array in the test store went unnoticed"
    )
