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


def test_empty_baseline_would_compare_equal_to_anything():
    """Why the guard is needed: store_equals cannot detect an empty baseline."""
    from ngff_zarr import to_ngff_zarr
    from zarr.storage import MemoryStore

    populated = MemoryStore()
    to_ngff_zarr(populated, _tiny_multiscales(), version="0.4")

    assert store_equals(MemoryStore(), populated), (
        "store_equals is expected to report an empty baseline as equal; "
        "the guard in verify_against_baseline is what makes that safe"
    )
