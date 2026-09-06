# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import json
import tempfile

import pytest
import zarr
from dask_image import imread
from ngff_zarr import Methods, config, to_multiscales, to_ngff_image, to_ngff_zarr
from packaging import version

from ._data import verify_against_baseline

zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def test_path_store_sharding(input_images, tmp_path):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4_chunks64/RFC3_GAUSSIAN.zarr"
    chunks = (64, 64)
    multiscales = to_multiscales(
        image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
    )
    store = tmp_path / "sharding_v04.zarr"

    chunks_per_shard = 2
    version = "0.4"
    with pytest.raises(ValueError):
        to_ngff_zarr(
            store, multiscales, version=version, chunks_per_shard=chunks_per_shard
        )

    version = "0.5"
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(
            tmpdir, multiscales, version=version, chunks_per_shard=chunks_per_shard
        )
        with open(tmpdir + "/zarr.json") as f:
            zarr_json = json.load(f)
        assert zarr_json["zarr_format"] == 3
        metadata = zarr_json["consolidated_metadata"]["metadata"]
        scale0 = metadata["scale0/image"]
        assert scale0["shape"][0] == 256
        assert scale0["shape"][1] == 256
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][0] == 128
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][1] == 128
        assert scale0["codecs"][0]["name"] == "sharding_indexed"
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][0] == 64
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][1] == 64

        verify_against_baseline(
            dataset_name, baseline_name, multiscales, version=version
        )

    chunks_per_shard = (2, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(
            tmpdir, multiscales, version=version, chunks_per_shard=chunks_per_shard
        )
        with open(tmpdir + "/zarr.json") as f:
            zarr_json = json.load(f)
        assert zarr_json["zarr_format"] == 3
        metadata = zarr_json["consolidated_metadata"]["metadata"]
        scale0 = metadata["scale0/image"]
        assert scale0["shape"][0] == 256
        assert scale0["shape"][1] == 256
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][0] == 128
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][1] == 64
        assert scale0["codecs"][0]["name"] == "sharding_indexed"
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][0] == 64
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][1] == 64

        verify_against_baseline(
            dataset_name, baseline_name, multiscales, version=version
        )

    chunks_per_shard = {"y": 2, "x": 1}
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(
            tmpdir, multiscales, version=version, chunks_per_shard=chunks_per_shard
        )
        with open(tmpdir + "/zarr.json") as f:
            zarr_json = json.load(f)
        assert zarr_json["zarr_format"] == 3
        metadata = zarr_json["consolidated_metadata"]["metadata"]
        scale0 = metadata["scale0/image"]
        assert scale0["shape"][0] == 256
        assert scale0["shape"][1] == 256
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][0] == 128
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][1] == 64
        assert scale0["codecs"][0]["name"] == "sharding_indexed"
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][0] == 64
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][1] == 64

        verify_against_baseline(
            dataset_name, baseline_name, multiscales, version=version
        )


@pytest.mark.filterwarnings("ignore:use_tensorstore is deprecated:DeprecationWarning")
def test_zarrista_sharding(input_images):
    pytest.importorskip("zarrista")

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4_chunks64/RFC3_GAUSSIAN.zarr"
    chunks = (64, 64)
    multiscales = to_multiscales(
        image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
    )

    chunks_per_shard = 2
    version = "0.5"
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(
            tmpdir,
            multiscales,
            version=version,
            chunks_per_shard=chunks_per_shard,
            use_tensorstore=True,
        )
        with open(tmpdir + "/zarr.json") as f:
            zarr_json = json.load(f)
        assert zarr_json["zarr_format"] == 3
        metadata = zarr_json["consolidated_metadata"]["metadata"]
        scale0 = metadata["scale0/image"]
        assert scale0["shape"][0] == 256
        assert scale0["shape"][1] == 256
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][0] == 128
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][1] == 128
        assert scale0["codecs"][0]["name"] == "sharding_indexed"
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][0] == 64
        assert scale0["codecs"][0]["configuration"]["chunk_shape"][1] == 64

        verify_against_baseline(
            dataset_name, baseline_name, multiscales, version=version
        )


@pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)
def test_large_image_serialization_with_sharding(input_images, tmp_path):
    default_mem_target = config.memory_target
    config.memory_target = int(1e6)

    dataset_name = "lung_series"
    data = imread.imread(input_images[dataset_name])
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 2.5, "y": 1.40625, "x": 1.40625},
        translation={"z": 332.5, "y": 360.0, "x": 0.0},
        name="LIDC2",
    )
    multiscales = to_multiscales(image)
    # baseline_name = "auto/memory_target_1e6.zarr"
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    test_store = tmp_path / "large_image.zarr"
    chunks_per_shard = 1
    to_ngff_zarr(
        test_store, multiscales, version="0.5", chunks_per_shard=chunks_per_shard
    )
    # verify_against_baseline(dataset_name, baseline_name, multiscales)

    config.memory_target = default_mem_target


def _stored_grid(store):
    """The stored shard shape and the inner chunk shape of one scale array."""
    for path in sorted(store.rglob("zarr.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("node_type") != "array":
            continue
        inner = next(
            codec["configuration"]["chunk_shape"]
            for codec in metadata["codecs"]
            if codec["name"] == "sharding_indexed"
        )
        return metadata["chunk_grid"]["configuration"]["chunk_shape"], inner
    raise AssertionError(f"no array metadata under {store}")


@pytest.mark.parametrize("path", ["small", "large", "metadata_only"])
def test_a_shard_may_run_past_the_axis(tmp_path, path):
    """The requested chunk survives an axis that factors badly (gh-733).

    A shard shortened to the axis has to be divided evenly by the inner chunk,
    and 275 (5 * 5 * 11) has no divisor near 256, so the chunk collapsed to 55
    while a prime 271 collapsed to the whole shard. Zarr v3 permits a partial
    final shard, so the shard is stored as asked instead. Every write path
    describes the array the same way.
    """
    import dask.array as da
    import numpy as np

    data = da.zeros((2, 12, 275, 271), dtype=np.uint8, chunks=(1, 1, 256, 256))
    image = to_ngff_image(data, dims=["c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[], chunks={"c": 1, "z": 1, "y": 256, "x": 256}
    )
    store = tmp_path / f"{path}.ome.zarr"
    chunks_per_shard = {"z": 10, "y": 2, "x": 2, "c": 2}

    default_mem_target = config.memory_target
    if path == "large":
        # Below the 1.8 MB payload, so the write takes the large-array path.
        config.memory_target = 1024 * 1024
    try:
        to_ngff_zarr(
            str(store),
            multiscales,
            chunks_per_shard=chunks_per_shard,
            metadata_only=path == "metadata_only",
        )
    finally:
        config.memory_target = default_mem_target

    assert _stored_grid(store) == ([2, 10, 512, 512], [1, 1, 256, 256])


def test_a_shard_is_capped_at_the_chunks_that_cover_the_axis(tmp_path):
    """No shard grid is larger than the array needs (gh-733).

    Two chunks per shard on an axis one chunk long would store a shard grid
    twice the array; the shard is held to the chunks that cover the axis.
    """
    import dask.array as da
    import numpy as np

    data = da.zeros((1, 3, 29, 185), dtype=np.uint8, chunks=(1, 3, 8, 8))
    image = to_ngff_image(data, dims=["t", "c", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[], chunks=8)
    store = tmp_path / "capped.ome.zarr"
    to_ngff_zarr(str(store), multiscales, chunks_per_shard=2)

    shard, inner = _stored_grid(store)
    assert shard == [1, 3, 16, 16]
    assert inner == [1, 3, 8, 8]
