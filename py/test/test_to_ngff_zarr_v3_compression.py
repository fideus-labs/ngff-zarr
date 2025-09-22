# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import json
from packaging import version
import tempfile

import pytest

import zarr.storage
import zarr


from ngff_zarr import Methods, to_multiscales, to_ngff_zarr

from ._data import verify_against_baseline

zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)


def test_zarr_v3_compression(input_images):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/RFC3_GAUSSIAN.zarr"
    chunks = (64, 64)
    multiscales = to_multiscales(
        image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
    )

    compressors = zarr.codecs.BloscCodec(
        cname="zlib", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
    )

    # Test writing OME-Zarr 0.5 with compression
    version = "0.5"
    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, multiscales, version=version, compressors=compressors)

        with open(tmpdir + "/zarr.json") as f:
            zarr_json = json.load(f)
        assert zarr_json["zarr_format"] == 3
        metadata = zarr_json["consolidated_metadata"]["metadata"]

        scale0 = metadata["scale0/image"]

        assert scale0["codecs"][1]["name"] == "blosc"
        assert scale0["codecs"][1]["configuration"]["cname"] == "zlib"
        assert scale0["codecs"][1]["configuration"]["clevel"] == 5
        assert scale0["codecs"][1]["configuration"]["shuffle"] == "shuffle"

        scale1 = metadata["scale1/image"]

        assert scale1["codecs"][1]["name"] == "blosc"
        assert scale1["codecs"][1]["configuration"]["cname"] == "zlib"
        assert scale1["codecs"][1]["configuration"]["clevel"] == 5
        assert scale1["codecs"][1]["configuration"]["shuffle"] == "shuffle"

        scale2 = metadata["scale2/image"]

        assert scale2["codecs"][1]["name"] == "blosc"
        assert scale2["codecs"][1]["configuration"]["cname"] == "zlib"
        assert scale2["codecs"][1]["configuration"]["clevel"] == 5
        assert scale2["codecs"][1]["configuration"]["shuffle"] == "shuffle"

        verify_against_baseline(
            dataset_name, baseline_name, multiscales, version=version
        )


def test_zarr_v3_compression_with_sharding(input_images):
    """Test Zarr v3 compression combined with sharding functionality"""
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/RFC3_GAUSSIAN.zarr"
    chunks = (64, 64)
    multiscales = to_multiscales(
        image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
    )

    compressors = zarr.codecs.BloscCodec(
        cname="zlib", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
    )

    version = "0.5"
    chunks_per_shard = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(
            tmpdir,
            multiscales,
            version=version,
            compressors=compressors,
            chunks_per_shard=chunks_per_shard,
        )

        with open(tmpdir + "/zarr.json") as f:
            zarr_json = json.load(f)
        assert zarr_json["zarr_format"] == 3
        metadata = zarr_json["consolidated_metadata"]["metadata"]

        scale0 = metadata["scale0/image"]
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][0] == 128
        assert scale0["chunk_grid"]["configuration"]["chunk_shape"][1] == 128

        sharding_codec = scale0["codecs"][0]
        assert sharding_codec["name"] == "sharding_indexed"
        assert sharding_codec["configuration"]["chunk_shape"][0] == 64
        assert sharding_codec["configuration"]["chunk_shape"][1] == 64

        blosc_codec = sharding_codec["configuration"]["codecs"][1]
        assert blosc_codec["name"] == "blosc"
        assert blosc_codec["configuration"]["cname"] == "zlib"
        assert blosc_codec["configuration"]["clevel"] == 5
        assert blosc_codec["configuration"]["shuffle"] == "shuffle"

        scale1 = metadata["scale1/image"]
        sharding_codec = scale1["codecs"][0]
        assert sharding_codec["name"] == "sharding_indexed"

        blosc_codec = sharding_codec["configuration"]["codecs"][1]
        assert blosc_codec["name"] == "blosc"
        assert blosc_codec["configuration"]["cname"] == "zlib"
        assert blosc_codec["configuration"]["clevel"] == 5
        assert blosc_codec["configuration"]["shuffle"] == "shuffle"

        verify_against_baseline(
            dataset_name, baseline_name, multiscales, version=version
        )


def test_zarr_v3_compression_rejected_for_ome_zarr_04(input_images):
    """Test that compressors argument is rejected for OME-Zarr version 0.4"""
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    chunks = (64, 64)
    multiscales = to_multiscales(
        image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
    )
    store = zarr.storage.MemoryStore()

    compressors = zarr.codecs.BloscCodec(
        cname="zlib", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
    )

    # Setting `compressors` argument for OME-Zarr 0.4 should raise an error
    version = "0.4"
    with pytest.raises(ValueError, match="not supported for OME-Zarr version 0.4"):
        to_ngff_zarr(store, multiscales, version=version, compressors=compressors)
