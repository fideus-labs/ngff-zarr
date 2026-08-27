# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import zarr
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
from ngff_zarr._zarrista_utils import open_lazy_array
from packaging import version

zarr_version = version.parse(zarr.__version__)


def _multiscales(rng_seed=0, scale_factors=()):
    data = np.random.default_rng(rng_seed).random((1, 64, 64, 64)).astype(np.float32)
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
    "ngff_version",
    [
        "0.4",
        pytest.param(
            "0.5",
            marks=pytest.mark.skipif(
                zarr_version < version.parse("3.0.0b1"),
                reason="zarr version < 3.0.0b1",
            ),
        ),
    ],
)
def test_overwrite_of_read_store_raises(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(), version=ngff_version)

    base = from_ome_zarr(path).images[0]
    multiscales = to_multiscales(
        base, scale_factors=[2], chunks=(1, 32, 32, 32), cache=False
    )
    with pytest.raises(ValueError, match="reads its data from that store"):
        to_ome_zarr(path, multiscales, overwrite=True, version=ngff_version)

    group = "scale0/image"
    assert zarr.open_group(path, mode="r")[group][:].max() > 0.0


def test_read_store_written_to_other_path(tmp_path):
    source = str(tmp_path / "source.ome.zarr")
    to_ome_zarr(source, _multiscales(), version="0.4")

    base = from_ome_zarr(source).images[0]
    multiscales = to_multiscales(
        base, scale_factors=[2], chunks=(1, 32, 32, 32), cache=False
    )
    destination = str(tmp_path / "destination.ome.zarr")
    to_ome_zarr(destination, multiscales, overwrite=True, version="0.4")

    written = zarr.open_group(destination, mode="r")["scale0/image"][:]
    original = zarr.open_group(source, mode="r")["scale0/image"][:]
    np.testing.assert_array_equal(written, original)


def test_in_memory_multiscales_overwrites_in_place(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    multiscales = _multiscales()
    to_ome_zarr(path, multiscales, version="0.4")
    to_ome_zarr(path, multiscales, overwrite=True, version="0.4")

    assert zarr.open_group(path, mode="r")["scale0/image"][:].max() > 0.0


def test_materialized_data_overwrites_in_place(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(), version="0.4")

    base = from_ome_zarr(path).images[0]
    base.data = base.data.persist()
    multiscales = to_multiscales(base, scale_factors=[], cache=False)
    to_ome_zarr(path, multiscales, overwrite=True, version="0.4")

    assert zarr.open_group(path, mode="r")["scale0/image"][:].max() > 0.0


def test_image_opened_from_the_array_path_raises(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(), version="0.4")
    image = to_ngff_image(
        open_lazy_array(f"{path}/scale0/image"),
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )
    multiscales = to_multiscales(image, scale_factors=[], cache=False)
    with pytest.raises(ValueError, match="reads its data from that store"):
        to_ome_zarr(path, multiscales, overwrite=True, version="0.4")


def test_level_alone_in_reading_the_store_raises(tmp_path):
    """A level reading the store is caught even when no other level shares it.

    The walk skips a layer another level already carried it past, so a level
    whose read is its own has to be reached on its own turn.
    """
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(scale_factors=[2]), version="0.4")

    multiscales = _multiscales(rng_seed=1, scale_factors=[2])
    multiscales.images[1].data = open_lazy_array(f"{path}/scale1/image")
    with pytest.raises(ValueError, match=r"images\[1\] reads its data"):
        to_ome_zarr(path, multiscales, overwrite=True, version="0.4")
