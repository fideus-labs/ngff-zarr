# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
import json
import asyncio
from packaging import version

import zarr
import pooch
import pytest
from itkwasm_image_io import imread
from ngff_zarr import itk_image_to_ngff_image, to_ngff_zarr
from ngff_zarr._zarr_kwargs import zarr_kwargs

from zarr.storage import MemoryStore
from deepdiff import DeepDiff

test_data_ipfs_cid = "bafybeigkjwg2dr7lvjgpns5nt65kfqfse2qdzqpohe4dvcvwj7j4xf3xbe"
test_data_sha256 = "ae6a3aa6e6c0faf4ec8c628c125e1125226ef0b6743158a121d22d7ab2e1d0e8"

test_dir = Path(__file__).resolve().parent
extract_dir = "data"
test_data_dir = test_dir / extract_dir

zarr_version = version.parse(zarr.__version__)
zarr_version_major = zarr_version.major


@pytest.fixture(scope="package")
def input_images():
    untar = pooch.Untar(extract_dir=extract_dir)
    pooch.retrieve(
        fname="data.tar.gz",
        path=test_dir,
        # url="https://github.com/thewtex/ngff-zarr/releases/download/v0.15.0/ngff-zarr-0.15.0-test-data.tar.gz",
        url=f"https://itk.mypinata.cloud/ipfs/{test_data_ipfs_cid}/data.tar.gz",
        # url=f"https://{test_data_ipfs_cid}.ipfs.w3s.link/ipfs/{test_data_ipfs_cid}/data.tar.gz",
        known_hash=f"sha256:{test_data_sha256}",
        processor=untar,
    )
    result = {}

    image = imread(test_data_dir / "input" / "cthead1.png")
    image_ngff = itk_image_to_ngff_image(image)
    result["cthead1"] = image_ngff

    result["lung_series"] = test_data_dir / "input" / "lung_series" / "*"

    image = imread(
        test_data_dir / "input" / "brain_two_components.nrrd",
    )
    image_ngff = itk_image_to_ngff_image(image)
    result["brain_two_components"] = image_ngff

    image = imread(test_data_dir / "input" / "2th_cthead1.png")
    image_ngff = itk_image_to_ngff_image(image)
    result["2th_cthead1"] = image_ngff

    image = imread(test_data_dir / "input" / "MR-head.nrrd")
    image_ngff = itk_image_to_ngff_image(image)
    result["MR-head"] = image_ngff

    return result


async def collect_values(async_gen):
    return [item async for item in async_gen]


def store_keys(store):
    zarr_version = version.parse(zarr.__version__)
    if zarr_version >= version.parse("3.0.0b1"):
        keys = asyncio.run(collect_values(store.list()))
    else:
        keys = store.keys()
    return set(keys)


async def async_store_contents(store, keys):
    return {k: (await store.get(k)).to_bytes() for k in keys}


async def async_memory_store_contents(store, keys):
    from zarr.core.buffer import default_buffer_prototype

    return {
        k: (await store.get(k, default_buffer_prototype())).to_bytes() for k in keys
    }


def store_contents(store, keys):
    zarr_version = version.parse(zarr.__version__)
    if zarr_version >= version.parse("3.0.0b1"):
        if isinstance(store, MemoryStore):
            contents = asyncio.run(async_memory_store_contents(store, keys))
        else:
            contents = asyncio.run(async_store_contents(store, keys))
    else:
        contents = {k: store[k] for k in keys}
    return contents


def store_equals(baseline_store, test_store):
    baseline_keys = store_keys(baseline_store)
    test_keys = store_keys(test_store)
    json_keys = {".zmetadata", ".zattrs", ".zgroup", "zarr.json"}
    baseline_contents = store_contents(baseline_store, baseline_keys)
    test_contents = store_contents(test_store, test_keys)

    for k in baseline_keys:
        if k in json_keys:
            baseline_metadata = json.loads(baseline_contents[k].decode("utf-8"))
            test_metadata = json.loads(test_contents[k].decode("utf-8"))

            diff = DeepDiff(baseline_metadata, test_metadata, ignore_order=True)
            if diff:
                sys.stderr.write("Metadata in {k} files do not match\n")
                sys.stderr.write(f"Differences: {diff}\n")
                return False
        else:
            if k not in test_keys:
                sys.stderr.write(f"baseline key {k} not in test keys\n")
                sys.stderr.write(f"test keys: {test_keys}\n")
                return False
            if (
                baseline_contents.get(k) != test_contents.get(k)
                and ".zattrs" not in k
                and ".zgroup" not in k
                and "zarr.json" not in k
            ):
                sys.stderr.write(f"test value != baseline value for key {k}\n")
                sys.stderr.write(f"baseline: {baseline_contents[k]}, \n")
                sys.stderr.write(f"test: {test_contents[k]}, \n")
                return False
    return True


def verify_against_baseline(dataset_name, baseline_name, multiscales, version="0.4"):
    try:
        from zarr.storage import DirectoryStore

        baseline_store = DirectoryStore(
            test_data_dir / f"baseline/v{version}/{dataset_name}/{baseline_name}",
            **zarr_kwargs,
        )
    except ImportError:
        from zarr.storage import LocalStore

        baseline_store = LocalStore(
            test_data_dir / f"baseline/v{version}/{dataset_name}/{baseline_name}"
        )

    test_store = MemoryStore()
    to_ngff_zarr(test_store, multiscales, version=version)
    assert store_equals(baseline_store, test_store)


def store_new_multiscales(dataset_name, baseline_name, multiscales, version="0.4"):
    """Helper method for writing output results to disk
    for later upload as test baseline"""
    try:
        from zarr.storage import DirectoryStore

        store = DirectoryStore(
            test_data_dir
            / f"baseline/zarr{zarr_version_major}/v{version}/{dataset_name}/{baseline_name}",
            **zarr_kwargs,
        )
    except ImportError:
        from zarr.storage import LocalStore

        store = LocalStore(
            test_data_dir
            / f"baseline/zarr{zarr_version_major}/v{version}/{dataset_name}/{baseline_name}"
        )
    to_ngff_zarr(store, multiscales, version=version)
