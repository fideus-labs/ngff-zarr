import json
from packaging import version
import tempfile

import pytest

import zarr.storage
from zarr.storage import MemoryStore
import zarr

from dask_image import imread

from ngff_zarr import Methods, to_multiscales, to_ngff_zarr, config, to_ngff_image

from ._data import verify_against_baseline

zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def test_zarr_python_sharding(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/RFC3_GAUSSIAN.zarr"
    chunks = (64, 64)
    multiscales = to_multiscales(
        image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
    )
    store = zarr.storage.MemoryStore()

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


def test_tensorstore_sharding(input_images):
    pytest.importorskip("tensorstore")

    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/RFC3_GAUSSIAN.zarr"
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
def test_large_image_serialization_with_sharding(input_images):
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
    test_store = MemoryStore()
    chunks_per_shard = 1
    to_ngff_zarr(
        test_store, multiscales, version="0.5", chunks_per_shard=chunks_per_shard
    )
    # verify_against_baseline(dataset_name, baseline_name, multiscales)

    config.memory_target = default_mem_target
