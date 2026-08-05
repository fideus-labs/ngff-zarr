# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import asyncio
import json
import sys
import time
from pathlib import Path

import pooch
import pytest
import zarr
from deepdiff import DeepDiff
from itkwasm_image_io import imread
from ngff_zarr import itk_image_to_ngff_image, to_ngff_zarr
from ngff_zarr._zarr_kwargs import zarr_kwargs
from packaging import version
from zarr.storage import MemoryStore

test_data_ipfs_cid = "bafybeifqibhcomn4u42aqrgvttyfteysbspvzez5sbezcqj5yylzzafpma"
test_data_sha256 = "e24fc764f562b68724665a9d36fafb64661b9cb433653d12597dd66ed9f28439"

test_dir = Path(__file__).resolve().parent
extract_dir = "data"
test_data_dir = test_dir / extract_dir

zarr_version = version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

# The test-data archive is downloaded from a GitHub release at the start of the
# test session. CI runners occasionally hit transient network failures (DNS
# blips, connection timeouts to github.com); without retries a single blip
# fails the entire test matrix. See the retrying downloader below.
#
# (connect, read) timeout in seconds: fail fast on an unreachable host but
# still allow a slow-but-alive download to finish.
_DOWNLOAD_TIMEOUT = (30, 120)
_DOWNLOAD_MAX_ATTEMPTS = 4
_DOWNLOAD_BACKOFF_SECONDS = 3.0


def _retrying_downloader(
    max_attempts=_DOWNLOAD_MAX_ATTEMPTS,
    backoff=_DOWNLOAD_BACKOFF_SECONDS,
    timeout=_DOWNLOAD_TIMEOUT,
):
    """Build a pooch downloader that retries transient network failures.

    Wraps :class:`pooch.HTTPDownloader`, retrying on connection/timeout errors
    and 5xx/429 responses with exponential backoff. Non-transient errors (such
    as a 404) are raised immediately. ``pooch`` hands the downloader a fresh
    temporary path each call and ``HTTPDownloader`` reopens it with ``"w+b"``,
    so every retry restarts the download from a clean file.
    """
    import requests.exceptions

    transient_errors = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )

    def _is_transient(exc):
        if isinstance(exc, transient_errors):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            response = exc.response
            return response is not None and (
                response.status_code >= 500 or response.status_code == 429
            )
        return False

    def download(url, output_file, pooch_, check_only=False):
        http_downloader = pooch.HTTPDownloader(timeout=timeout)
        for attempt in range(1, max_attempts + 1):
            try:
                return http_downloader(url, output_file, pooch_, check_only=check_only)
            except requests.exceptions.RequestException as exc:
                if check_only or attempt == max_attempts or not _is_transient(exc):
                    raise
                wait = backoff * 2 ** (attempt - 1)
                sys.stderr.write(
                    f"Download of {url} failed "
                    f"(attempt {attempt}/{max_attempts}): {exc}. "
                    f"Retrying in {wait:.0f}s...\n"
                )
                time.sleep(wait)
        return None

    return download


@pytest.fixture(scope="package")
def input_images():
    untar = pooch.Untar(extract_dir=extract_dir)
    pooch.retrieve(
        fname="data.tar.gz",
        path=test_dir,
        url="https://github.com/fideus-labs/ngff-zarr/releases/download/testing-data/ngff-zarr-testing-data-v0.21.0.tar.gz",
        # url=f"https://itk.mypinata.cloud/ipfs/{test_data_ipfs_cid}",
        # url=f"https://{test_data_ipfs_cid}.ipfs.w3s.link/ipfs/{test_data_ipfs_cid}/data.tar.gz",
        known_hash=f"sha256:{test_data_sha256}",
        processor=untar,
        downloader=_retrying_downloader(),
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


_METADATA_SUFFIXES = (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")


def _is_metadata_key(key):
    return key.endswith(_METADATA_SUFFIXES)


def _drop_codecs(node):
    """Drop the codec in use, which zarr picks and changes between releases.

    zstd until 3.2, blosc/lz4 from 3.3. ngff-zarr does not choose it, so
    comparing it compares zarr rather than this library.
    """
    if isinstance(node, list):
        return [_drop_codecs(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: _drop_codecs(value)
        for key, value in node.items()
        if key not in ("compressor", "compressors", "codecs")
    }


def _array_paths(keys, contents):
    """Group paths that hold an array, for both zarr formats."""
    paths = set()
    for key in keys:
        if key.endswith(".zarray"):
            paths.add(key[: -len("/.zarray")] if "/" in key else "")
        elif key.endswith("zarr.json"):
            try:
                meta = json.loads(contents[key].decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if meta.get("node_type") == "array":
                paths.add(key[: -len("/zarr.json")] if "/" in key else "")
    return paths


def _arrays_equal(baseline_store, test_store, path):
    """Compare decompressed values, so the codec in use does not matter."""
    import numpy as np

    kwargs = {"mode": "r"}
    if path:
        kwargs["path"] = path
    baseline = zarr.open_array(store=baseline_store, **kwargs)
    test = zarr.open_array(store=test_store, **kwargs)

    if baseline.shape != test.shape:
        sys.stderr.write(f"shape differs at {path or '/'}: ")
        sys.stderr.write(f"{baseline.shape} != {test.shape}\n")
        return False
    if baseline.dtype != test.dtype:
        sys.stderr.write(f"dtype differs at {path or '/'}: ")
        sys.stderr.write(f"{baseline.dtype} != {test.dtype}\n")
        return False
    if not np.array_equal(np.asarray(baseline[...]), np.asarray(test[...])):
        sys.stderr.write(f"values differ at {path or '/'}\n")
        return False
    return True


def store_equals(baseline_store, test_store):
    """Compare two stores by value rather than by stored bytes.

    Chunks are compressed, so identical data encoded by different codecs, or
    by different releases of the same codec, yields different bytes. Compare
    decompressed arrays, and parse the metadata rather than diffing its bytes.
    """
    baseline_keys = store_keys(baseline_store)
    test_keys = store_keys(test_store)

    # Key sets must match in both directions: a missing key hides data, an
    # extra one is data the baseline never blessed.
    if baseline_keys != test_keys:
        missing = sorted(baseline_keys - test_keys)
        extra = sorted(test_keys - baseline_keys)
        if missing:
            sys.stderr.write(f"baseline keys not in test store: {missing}\n")
        if extra:
            sys.stderr.write(f"test keys not in baseline: {extra}\n")
        return False

    baseline_contents = store_contents(baseline_store, baseline_keys)
    test_contents = store_contents(test_store, test_keys)

    for k in sorted(baseline_keys):
        if not _is_metadata_key(k):
            continue

        baseline_metadata = _drop_codecs(
            json.loads(baseline_contents[k].decode("utf-8"))
        )
        test_metadata = _drop_codecs(json.loads(test_contents[k].decode("utf-8")))
        diff = DeepDiff(baseline_metadata, test_metadata, ignore_order=True)
        if diff:
            sys.stderr.write(f"Metadata in {k} does not match\n")
            sys.stderr.write(f"Differences: {diff}\n")
            return False

    for path in sorted(_array_paths(baseline_keys, baseline_contents)):
        if not _arrays_equal(baseline_store, test_store, path):
            return False
    return True


def verify_against_baseline(dataset_name, baseline_name, multiscales, version="0.4"):
    baseline_path = (
        test_data_dir
        / f"baseline/zarr{zarr_version_major}/v{version}/{dataset_name}/{baseline_name}"
    )

    # store_equals walks the baseline's keys, so an absent one compares equal
    # to anything. Skip instead: a missing baseline is a gap, not a pass.
    if not baseline_path.is_dir():
        pytest.skip(
            f"No baseline at {baseline_path.relative_to(test_data_dir)} "
            f"(zarr {zarr.__version__}); nothing to compare against."
        )

    try:
        from zarr.storage import DirectoryStore

        baseline_store = DirectoryStore(baseline_path, **zarr_kwargs)
    except ImportError:
        from zarr.storage import LocalStore

        baseline_store = LocalStore(baseline_path)

    # A directory that exists but holds no arrays (a truncated extraction,
    # say) would also compare vacuously. Demand at least one array.
    baseline_keys = store_keys(baseline_store)
    assert _array_paths(baseline_keys, store_contents(baseline_store, baseline_keys)), (
        f"baseline at {baseline_path.relative_to(test_data_dir)} holds no arrays; "
        "the comparison would pass vacuously"
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
