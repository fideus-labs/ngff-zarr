# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dataclasses

import dask.array as da
import numpy as np
import zarr
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr


def _multiscales():
    data = np.random.default_rng(0).random((1, 64, 64, 64)).astype(np.float32)
    image = to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )
    return to_multiscales(image, scale_factors=[2], chunks=(1, 32, 32, 32), cache=False)


def _constant_level(shape=(1, 32, 32, 32), value=7.0):
    return da.full(shape, value, chunks=(1, 32, 32, 32), dtype=np.float32)


def test_replaced_level_is_written_as_given(tmp_path):
    multiscales = _multiscales()
    multiscales.images[1] = dataclasses.replace(
        multiscales.images[1], data=_constant_level()
    )
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.4")

    level1 = zarr.open_group(path, mode="r")["scale1/image"][:]
    np.testing.assert_array_equal(level1, 7.0)


def test_level_data_assigned_in_place_is_written_as_given(tmp_path):
    multiscales = _multiscales()
    multiscales.images[1].data = _constant_level()
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.4")

    level1 = zarr.open_group(path, mode="r")["scale1/image"][:]
    np.testing.assert_array_equal(level1, 7.0)


def test_generated_levels_still_match_the_pipeline(tmp_path):
    multiscales = _multiscales()
    expected = multiscales.images[1].data.compute()
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.4")

    level1 = zarr.open_group(path, mode="r")["scale1/image"][:]
    np.testing.assert_allclose(level1, expected, rtol=1e-5, atol=1e-6)
    assert len(from_ome_zarr(path).images) == 2


def test_hand_built_multiscales_has_no_generated_keys(tmp_path):
    multiscales = _multiscales()
    rebuilt = dataclasses.replace(multiscales, generated_data_keys=None)
    rebuilt.images[1] = dataclasses.replace(rebuilt.images[1], data=_constant_level())
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, rebuilt, version="0.4")

    level1 = zarr.open_group(path, mode="r")["scale1/image"][:]
    np.testing.assert_array_equal(level1, 7.0)
