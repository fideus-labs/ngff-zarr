# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import NgffImage, from_ome_zarr, open_array, to_multiscales, to_ome_zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)
needs_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)
VERSIONS = ["0.4", pytest.param("0.5", marks=needs_zarr_v3)]
SHAPE = (1, 32, 32, 32)
CHUNKS = (1, 16, 16, 16)


def _empty_store(path, ngff_version="0.4", **kwargs):
    image = NgffImage(
        da.zeros(SHAPE, dtype=np.float32, chunks=CHUNKS),
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )
    multiscales = to_multiscales(image, scale_factors=[], chunks=CHUNKS, cache=False)
    to_ome_zarr(path, multiscales, version=ngff_version, metadata_only=True, **kwargs)
    return multiscales.metadata.datasets[0].path


@pytest.mark.parametrize("ngff_version", VERSIONS)
def test_handle_describes_the_array(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    level0 = open_array(path, _empty_store(path, ngff_version))
    assert level0.shape == SHAPE
    assert level0.dtype == np.float32
    assert level0.chunks == CHUNKS
    assert level0.path == "scale0/image"
    assert level0.to_dask().shape == SHAPE


@pytest.mark.parametrize("ngff_version", VERSIONS)
def test_regions_written_through_the_handle_read_back(tmp_path, ngff_version):
    path = str(tmp_path / "image.ome.zarr")
    values = np.random.default_rng(0).random(SHAPE).astype(np.float32)
    level0 = open_array(path, _empty_store(path, ngff_version))
    for start in range(0, SHAPE[1], 16):
        level0[:, start : start + 16] = values[:, start : start + 16]

    np.testing.assert_array_equal(level0[:, 16:32], values[:, 16:32])
    np.testing.assert_array_equal(from_ome_zarr(path).images[0].data.compute(), values)
    np.testing.assert_array_equal(
        zarr.open_array(path, path="scale0/image", mode="r")[:], values
    )


def test_scalars_and_integer_indices_follow_numpy(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    level0 = open_array(path, _empty_store(path))
    level0[:, 3] = 7.0
    level0[0, 0:2, 0:2, 0:2] = np.full((2, 2, 2), 1.5, dtype=np.float32)

    plane = level0[:, 3]
    assert plane.shape == (1, 32, 32)
    assert plane.min() == 7.0
    np.testing.assert_array_equal(level0[0, 0:2, 0:2, 0:2], 1.5)
    assert level0[0, 4:8].max() == 0.0


@needs_zarr_v3
def test_sharded_store_reports_the_inner_chunks(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    level0 = open_array(path, _empty_store(path, "0.5", chunks_per_shard=2))
    assert level0.chunks == CHUNKS
    assert zarr.open_array(path, path="scale0/image", mode="r").shards == SHAPE


def test_missing_and_group_paths_are_refused(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    _empty_store(path)
    with pytest.raises(FileNotFoundError, match="No array"):
        open_array(path, "scale7/image")
    with pytest.raises(ValueError, match="group, not an array"):
        open_array(path, "scale0")
