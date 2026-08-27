# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array as da
import numpy as np
from ngff_zarr import Methods, NgffImage, to_multiscales
from ngff_zarr.methods._support import _dim_scale_factors, _incremental_factor

DIMS = ("c", "z", "y", "x")


def _image(shape):
    return NgffImage(
        da.zeros(shape, dtype=np.uint8, chunks=shape),
        DIMS,
        dict.fromkeys(DIMS, 1.0),
        dict.fromkeys(DIMS, 0.0),
    )


def test_requested_factor_is_kept_when_it_reaches_the_target():
    # 15 / 4 floors to 3, and 4 already gets there: 5 would too, but was not asked for.
    assert _incremental_factor(15, 3, 4) == 4
    assert _incremental_factor(61, 15, 4) == 4
    assert _incremental_factor(514, 128, 4) == 4


def test_a_floored_ladder_step_is_used_only_when_it_lands_on_the_target():
    # A [2, 5] ladder steps by 2.5 and floors to 2. A floored step is smaller
    # than the exact ratio, so it can only leave the level larger than the
    # target; it is used only when it lands on the target exactly, and the
    # search answers whenever it misses.
    for original in range(4, 400):
        for previous_absolute, absolute in ((2, 5), (3, 7), (4, 10), (2, 11)):
            previous = int(original / previous_absolute)
            target = int(original / absolute)
            if previous < 1 or target < 1:
                continue
            nominal = absolute // previous_absolute
            if int(previous / nominal) == target:
                assert _incremental_factor(previous, target, nominal) == nominal


def test_search_runs_only_when_the_requested_factor_misses():
    # A [2, 3] ladder: level 1 holds 7 of 15 samples and level 2 must hold 5;
    # no integer factor gets there and the search answers as before.
    assert _incremental_factor(7, 5, 1) == 2
    assert _incremental_factor(15, 0, 4) == 1


def test_odd_extents_keep_the_requested_factor():
    image = _image((2, 15, 17, 19))
    previous = dict.fromkeys(DIMS, 1)
    factors = _dim_scale_factors(
        DIMS, 4, previous, original_image=image, previous_image=image
    )
    assert factors == {"z": 4, "y": 4, "x": 4}
    factors = _dim_scale_factors(
        DIMS, {"z": 4, "y": 2}, previous, original_image=image, previous_image=image
    )
    assert factors == {"z": 4, "y": 2, "c": 1, "x": 1}


def test_pyramid_scales_follow_the_requested_factors():
    multiscales = to_multiscales(
        _image((1, 61, 64, 64)),
        scale_factors=[4, 16],
        method=Methods.ITKWASM_BIN_SHRINK,
        cache=False,
    )
    assert [image.data.shape for image in multiscales.images] == [
        (1, 61, 64, 64),
        (1, 15, 16, 16),
        (1, 3, 4, 4),
    ]
    assert [image.scale["z"] for image in multiscales.images] == [1.0, 4.0, 16.0]
    assert multiscales.images[2].translation["z"] == 7.5
