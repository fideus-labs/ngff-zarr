# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import zarr
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)
needs_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def _multiscales(scale_factors=()):
    data = np.random.default_rng(0).random((1, 64, 64, 64)).astype(np.float32)
    image = to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )
    return to_multiscales(
        image, scale_factors=list(scale_factors), chunks=(1, 32, 32, 32), cache=False
    )


@pytest.mark.parametrize(
    "ngff_version", ["0.4", pytest.param("0.5", marks=needs_zarr_v3)]
)
def test_foreign_root_attrs_survive_overwrite(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(), version=ngff_version)
    zarr.open_group(path, mode="r+").attrs["mine"] = {"k": 1}

    to_ome_zarr(path, _multiscales(), overwrite=True, version=ngff_version)

    attrs = dict(zarr.open_group(path, mode="r").attrs)
    assert attrs["mine"] == {"k": 1}
    assert len(from_ome_zarr(path).images) == 1


@needs_zarr_v3
def test_version_change_drops_stale_ome_keys(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(), version="0.4")
    zarr.open_group(path, mode="r+").attrs["mine"] = {"k": 1}

    to_ome_zarr(path, _multiscales(), overwrite=True, version="0.5")

    attrs = dict(zarr.open_group(path, mode="r").attrs)
    assert attrs["mine"] == {"k": 1}
    assert "ome" in attrs
    assert "multiscales" not in attrs


def test_overwrite_still_clears_previous_arrays(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(scale_factors=[2]), version="0.4")
    assert "scale1" in zarr.open_group(path, mode="r")

    to_ome_zarr(path, _multiscales(), overwrite=True, version="0.4")

    assert "scale1" not in zarr.open_group(path, mode="r")


def test_overwrite_of_missing_store_creates_it(tmp_path):
    path = str(tmp_path / "new.ome.zarr")
    to_ome_zarr(path, _multiscales(), overwrite=True, version="0.4")
    assert len(from_ome_zarr(path).images) == 1
