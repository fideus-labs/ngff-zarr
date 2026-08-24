# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import time
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr
from ngff_zarr import (
    Methods,
    NgffImage,
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)
needs_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)
_METADATA_FILES = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}


def _image(data):
    return to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        translation={"c": 0, "z": 0, "y": 0, "x": 0},
    )


def _chunk_files(path):
    return [
        file
        for file in Path(path).rglob("*")
        if file.is_file() and file.name not in _METADATA_FILES
    ]


@pytest.mark.parametrize(
    "ngff_version",
    [
        "0.4",
        pytest.param("0.5", marks=needs_zarr_v3),
        pytest.param("0.6", marks=needs_zarr_v3),
    ],
)
def test_terabyte_store_is_created_without_computing(tmp_path, ngff_version):
    data = da.zeros((1, 4096, 8192, 8192), dtype=np.float32, chunks=(1, 512, 512, 512))
    multiscales = to_multiscales(
        _image(data),
        scale_factors=[2, 4],
        chunks=(1, 512, 512, 512),
        method=Methods.ITKWASM_BIN_SHRINK,
        cache=False,
    )
    path = str(tmp_path / "image.ome.zarr")

    start = time.perf_counter()
    to_ome_zarr(path, multiscales, version=ngff_version, metadata_only=True)
    assert time.perf_counter() - start < 10.0

    assert _chunk_files(path) == []
    root = zarr.open_group(path, mode="r")
    for level, image in enumerate(multiscales.images):
        array = root[f"scale{level}/image"]
        assert array.shape == image.data.shape
        assert array.dtype == np.float32
        assert array.chunks == (1, 512, 512, 512)
    assert len(from_ome_zarr(path).images) == 3


def test_requested_chunks_are_kept(tmp_path):
    data = da.zeros((2, 40, 96, 96), dtype=np.uint16, chunks=(1, 8, 32, 32))
    multiscales = to_multiscales(
        _image(data), scale_factors=[2], chunks=(1, 8, 32, 32), cache=False
    )
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.4", metadata_only=True)

    root = zarr.open_group(path, mode="r")
    assert root["scale0/image"].chunks == (1, 8, 32, 32)
    assert root["scale1/image"].chunks == (1, 8, 32, 32)
    assert root["scale1/image"].shape == (2, 20, 48, 48)


def test_regions_filled_afterwards_round_trip(tmp_path):
    shape = (1, 64, 64, 64)
    values = np.random.default_rng(0).random(shape).astype(np.float32)
    data = da.zeros(shape, dtype=np.float32, chunks=(1, 16, 64, 64))
    multiscales = to_multiscales(
        _image(data), scale_factors=[2], chunks=(1, 16, 64, 64), cache=False
    )
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.4", metadata_only=True)

    level0 = zarr.open_array(path, path="scale0/image", mode="r+")
    for start in range(0, shape[1], 16):
        level0[:, start : start + 16] = values[:, start : start + 16]

    read = from_ome_zarr(path)
    np.testing.assert_array_equal(read.images[0].data.compute(), values)
    assert read.images[1].data.shape == (1, 32, 32, 32)
    assert read.images[1].data.compute().max() == 0.0


@needs_zarr_v3
def test_sharding_and_compression_are_applied(tmp_path):
    from zarr.codecs import BloscCodec

    data = da.zeros((1, 64, 128, 128), dtype=np.uint8, chunks=(1, 32, 32, 32))
    multiscales = to_multiscales(
        _image(data), scale_factors=[], chunks=(1, 32, 32, 32), cache=False
    )
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(
        path,
        multiscales,
        version="0.5",
        metadata_only=True,
        chunks_per_shard=2,
        compressors=[BloscCodec(cname="lz4")],
    )

    array = zarr.open_array(path, path="scale0/image", mode="r")
    assert array.shards == (1, 64, 64, 64)
    assert array.chunks == (1, 32, 32, 32)
    assert any(isinstance(codec, BloscCodec) for codec in array.compressors)
    assert _chunk_files(path) == []


def test_ozx_output_is_refused(tmp_path):
    data = da.zeros((1, 8, 8, 8), dtype=np.uint8, chunks=(1, 8, 8, 8))
    multiscales = to_multiscales(_image(data), scale_factors=[], cache=False)
    with pytest.raises(ValueError, match="metadata_only"):
        to_ome_zarr(
            str(tmp_path / "image.ozx"), multiscales, version="0.5", metadata_only=True
        )


def test_default_still_writes_data(tmp_path):
    values = np.random.default_rng(0).random((1, 16, 16, 16)).astype(np.float32)
    multiscales = to_multiscales(_image(values), scale_factors=[], cache=False)
    path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.4")
    assert _chunk_files(path) != []


@needs_zarr_v3
def test_displacement_field_store(tmp_path):
    data = da.zeros((2, 4096, 4096), dtype=np.float32, chunks=(2, 512, 512))
    image = NgffImage(
        data,
        dims=["c", "y", "x"],
        scale={"c": 1.0, "y": 0.5, "x": 0.5},
        translation={"c": 0.0, "y": 0.0, "x": 0.0},
        name="displacement_field",
        axes_types={"c": "displacement"},
    )
    multiscales = to_multiscales(image, scale_factors=[], cache=False)
    path = str(tmp_path / "field.ome.zarr")
    to_ome_zarr(path, multiscales, version="0.6", metadata_only=True)

    assert _chunk_files(path) == []
    read = from_ome_zarr(path)
    axes = read.metadata.intrinsic_coordinate_system.axes
    assert [axis.type for axis in axes] == ["displacement", "space", "space"]
    assert read.images[0].data.shape == (2, 4096, 4096)
