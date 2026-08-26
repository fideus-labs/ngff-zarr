# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import (
    NgffImage,
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
    update_root_attributes,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)
needs_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)
VERSIONS = ["0.4", pytest.param("0.5", marks=needs_zarr_v3)]
DIRECTION = {"acquisition": {"direction": [0, -1, 0, 1, 0, 0, 0, 0, 1]}}


def _multiscales(scale_factors=()):
    data = np.random.default_rng(0).random((1, 32, 32, 32)).astype(np.float32)
    image = to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )
    return to_multiscales(
        image, scale_factors=list(scale_factors), chunks=(1, 16, 16, 16), cache=False
    )


@pytest.mark.parametrize("ngff_version", VERSIONS)
def test_root_attributes_are_written_beside_the_ome_keys(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    multiscales = _multiscales()
    multiscales.root_attributes = dict(DIRECTION)
    to_ome_zarr(path, multiscales, version=ngff_version)

    attrs = dict(zarr.open_group(path, mode="r").attrs)
    assert attrs["acquisition"] == DIRECTION["acquisition"]
    assert ("ome" in attrs) == (ngff_version != "0.4")
    assert from_ome_zarr(path).root_attributes == DIRECTION


def test_root_attributes_follow_a_round_trip(tmp_path):
    source = str(tmp_path / "source.ome.zarr")
    multiscales = _multiscales()
    multiscales.root_attributes = dict(DIRECTION)
    to_ome_zarr(source, multiscales, version="0.4")

    read = from_ome_zarr(source)
    assert read.root_attributes == DIRECTION
    assert "multiscales" not in read.root_attributes

    copy = str(tmp_path / "copy.ome.zarr")
    to_ome_zarr(copy, read, version="0.4")
    assert from_ome_zarr(copy).root_attributes == DIRECTION


def test_a_store_without_foreign_keys_reads_back_none(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, _multiscales(), version="0.4")
    assert from_ome_zarr(path).root_attributes is None


def test_ome_keys_are_refused(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    multiscales = _multiscales()
    multiscales.root_attributes = {"multiscales": []}
    with pytest.raises(ValueError, match="root_attributes cannot set"):
        to_ome_zarr(path, multiscales, version="0.4")

    to_ome_zarr(path, _multiscales(), version="0.4")
    with pytest.raises(ValueError, match="root_attributes cannot set"):
        update_root_attributes(path, {"ome": {}})


@pytest.mark.parametrize("ngff_version", VERSIONS)
def test_update_merges_and_refreshes_the_consolidated_metadata(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    multiscales = _multiscales()
    multiscales.root_attributes = {"a": 1, "b": 2}
    to_ome_zarr(path, multiscales, version=ngff_version)

    update_root_attributes(path, {"b": 3, "c": 4})

    assert from_ome_zarr(path).root_attributes == {"a": 1, "b": 3, "c": 4}
    # zarr-python reads the consolidated copy first.
    attrs = dict(zarr.open_group(path, mode="r").attrs)
    assert (attrs["a"], attrs["b"], attrs["c"]) == (1, 3, 4)
    assert len(from_ome_zarr(path).images) == 1


def test_update_needs_a_store(tmp_path):
    with pytest.raises(ValueError, match="root group"):
        update_root_attributes(str(tmp_path / "missing.ome.zarr"), {"a": 1})


def test_explicit_attributes_win_over_the_kept_ones(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    multiscales = _multiscales()
    multiscales.root_attributes = {"a": 1, "keep": True}
    to_ome_zarr(path, multiscales, version="0.4")

    again = _multiscales()
    again.root_attributes = {"a": 2}
    to_ome_zarr(path, again, overwrite=True, version="0.4")

    assert from_ome_zarr(path).root_attributes == {"a": 2, "keep": True}


def test_metadata_only_store_carries_them_and_an_append_keeps_them(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    image = NgffImage(
        da.zeros((1, 32, 32, 32), dtype=np.float32, chunks=(1, 16, 16, 16)),
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )
    multiscales = to_multiscales(
        image, scale_factors=[], chunks=(1, 16, 16, 16), cache=False
    )
    multiscales.root_attributes = dict(DIRECTION)
    to_ome_zarr(path, multiscales, version="0.4", metadata_only=True)
    assert from_ome_zarr(path).root_attributes == DIRECTION

    base = from_ome_zarr(path).images[0]
    pyramid = to_multiscales(
        base, scale_factors=[2], chunks=(1, 16, 16, 16), cache=False
    )
    to_ome_zarr(path, pyramid, version="0.4", overwrite=False, start_level=1)

    read = from_ome_zarr(path)
    assert read.root_attributes == DIRECTION
    assert len(read.images) == 2
