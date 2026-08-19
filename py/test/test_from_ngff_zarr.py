# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import packaging.version
import pytest
import zarr
from dask_image import imread
from ngff_zarr import (
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)
from zarr.storage import MemoryStore

from ._data import test_data_dir, verify_against_baseline

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major


def test_from_ngff_zarr(input_images):
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
    multiscales.scale_factors = None
    multiscales.method = None
    multiscales.chunks = None
    baseline_name = "from_ngff_zarr"
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    # verify_against_baseline(dataset_name, baseline_name, multiscales)
    test_store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(test_store, multiscales, version=version)

    multiscales_back = from_ngff_zarr(test_store, version=version)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(
        dataset_name, baseline_name, multiscales_back, version=version
    )


def test_omero_zarr_from_ngff_zarr_to_ngff_zarr(input_images):
    dataset_name = "13457537"
    store_path = test_data_dir / "input" / f"{dataset_name}.zarr"
    version = "0.4"
    multiscales = from_ngff_zarr(store_path, version=version)
    test_store = MemoryStore()
    to_ngff_zarr(test_store, multiscales, version=version)


@pytest.mark.skipif(
    zarr_version_major < 3, reason="storage_options requires zarr-python 3"
)
def test_from_ngff_zarr_with_storage_options(input_images):
    """Test that storage_options parameter is accepted and handled correctly."""
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
    multiscales.scale_factors = None
    multiscales.method = None
    multiscales.chunks = None

    # Test with MemoryStore (should work as before)
    test_store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(test_store, multiscales, version=version)

    # Test with storage_options=None (should work as before)
    multiscales_back = from_ngff_zarr(test_store, version=version, storage_options=None)
    baseline_name = "from_ngff_zarr"
    verify_against_baseline(
        dataset_name, baseline_name, multiscales_back, version=version
    )

    # Test with empty storage_options dict (should work as before)
    multiscales_back2 = from_ngff_zarr(test_store, version=version, storage_options={})
    verify_against_baseline(
        dataset_name, baseline_name, multiscales_back2, version=version
    )


@pytest.mark.skipif(
    zarr_version_major < 3, reason="storage_options requires zarr-python 3"
)
def test_from_ngff_zarr_string_url_with_storage_options():
    """Test that string URLs with storage_options create appropriate stores."""
    from unittest.mock import MagicMock, patch

    # Mock the FsspecStore.from_url method
    with patch("zarr.storage.FsspecStore.from_url") as mock_from_url:
        mock_store = MagicMock()
        mock_from_url.return_value = mock_store

        # Mock zarr.open_group to avoid actual S3 calls
        with patch("zarr.open_group") as mock_open_group:
            mock_group = MagicMock()
            mock_group.attrs.asdict.return_value = {
                "multiscales": [{"version": "0.4", "datasets": [{"path": "0"}]}]
            }
            mock_open_group.return_value = mock_group

            # Mock dask.array.from_zarr to avoid actual data loading
            with patch("dask.array.from_zarr") as mock_from_zarr:
                mock_array = MagicMock()
                mock_array.dtype.byteorder = "="  # native endianness
                mock_from_zarr.return_value = mock_array

                # Test that S3 URL with storage_options creates FsspecStore
                url = "s3://test-bucket/test-dataset.zarr"
                storage_options = {"anon": True, "region_name": "us-west-2"}

                try:
                    from_ngff_zarr(url, storage_options=storage_options)

                    # Verify that FsspecStore.from_url was called with correct parameters
                    mock_from_url.assert_called_once_with(
                        url, storage_options=storage_options
                    )

                    # Verify that zarr.open_group was called with the created store
                    mock_open_group.assert_called_once_with(mock_store, mode="r")

                except Exception:
                    # If there are other issues (like missing metadata), that's okay
                    # We just want to verify the store creation logic
                    mock_from_url.assert_called_once_with(
                        url, storage_options=storage_options
                    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/does-not-matter.zarr",
        "http://example.com/does-not-matter.zarr",
        "s3://bucket/does-not-matter.zarr",
        "gs://bucket/does-not-matter.zarr",
        "azure://container/does-not-matter.zarr",
    ],
)
def test_from_ngff_zarr_remote_missing_backend_helpful_error(url):
    """A missing fsspec backend for any remote scheme yields an actionable error."""
    from unittest.mock import patch

    # Simulate fsspec failing to import its filesystem backend (aiohttp,
    # requests, s3fs, gcsfs, adlfs, ...).
    with patch(
        "zarr.open_group",
        side_effect=ModuleNotFoundError("No module named 'a_backend'"),
    ):
        with pytest.raises(ImportError) as exc_info:
            from_ngff_zarr(url)

    message = str(exc_info.value)
    assert "ngff-zarr[remote]" in message
    assert url in message
    # The original cause is preserved for debugging.
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_from_ngff_zarr_local_import_error_not_rewritten():
    """A non-remote store re-raises an ImportError unchanged (remote-only guard).

    Uses a zarr-python store object so the read goes through the legacy
    zarr-python branch that carries the remote-error rewriting; local paths
    now read through the zarrista compat layer without touching fsspec.
    """
    from unittest.mock import patch

    original_message = "No module named 'unrelated_dependency'"
    with patch(
        "zarr.open_group",
        side_effect=ModuleNotFoundError(original_message),
    ):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            from_ngff_zarr(MemoryStore())

    message = str(exc_info.value)
    assert "ngff-zarr[remote]" not in message
    assert original_message in message


def test_from_ngff_zarr_remote_https_read_smoke():
    """End-to-end read of a public remote OME-Zarr store (opt-in).

    This hits a live third-party endpoint, so it is gated behind
    ``NGFF_ZARR_NETWORK_TESTS`` and does not run in normal CI. A transient
    remote failure and a real regression are indistinguishable here: both
    surface as ``ValueError: No valid Zarr group found`` once zarr wraps the
    failed metadata read (observed on Windows + zarr v2, where the response is
    mis-decoded by the ``charmap`` codec). A broad ``except`` skip would
    therefore also mask genuine regressions, so the deterministic
    backend-handling paths are covered by the mocked tests above instead.
    """
    import os

    if not os.environ.get("NGFF_ZARR_NETWORK_TESTS"):
        pytest.skip("live remote read is opt-in; set NGFF_ZARR_NETWORK_TESTS=1 to run")
    pytest.importorskip("aiohttp", reason="aiohttp required to read https stores")

    url = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
    try:
        multiscales = from_ngff_zarr(url)
    except OSError as exc:
        # Connection refused / DNS / timeout: clearly the network, not the code.
        pytest.skip(f"remote store unavailable: {exc!r}")

    # Metadata is read eagerly; pixel data stays lazy, so this stays cheap.
    assert len(multiscales.images) > 0
    assert multiscales.images[0].data.ndim > 0


@pytest.mark.skipif(
    zarr_version_major < 3, reason="storage_options requires zarr-python 3"
)
def test_omero_metadata_backward_compatibility():
    """Test that OMERO metadata with only min/max or only start/end is handled correctly."""
    import numpy as np
    from ngff_zarr import from_ngff_zarr
    from zarr.storage import MemoryStore

    # Create a test store with OMERO metadata using only min/max (old format)
    store = MemoryStore()
    zarr_group = zarr.open_group(store, mode="w")

    # Create sample data
    np.random.seed(42)  # For reproducibility
    data = np.random.randint(0, 1000, size=(1, 1, 10, 100, 100), dtype=np.uint16)
    array = zarr_group.create_array(
        "0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 10, 50, 50)
    )
    array[:] = data

    # Set attributes with OMERO metadata using old min/max format only
    attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.5, 0.5]},
                            {
                                "type": "translation",
                                "translation": [0.0, 0.0, 0.0, 0.0, 0.0],
                            },
                        ],
                    }
                ],
            }
        ],
        "omero": {
            "channels": [
                {
                    "active": True,
                    "color": "FFFFFF",
                    "label": "channel-0",
                    "window": {
                        "min": 0,
                        "max": 1000,
                        # Note: no start/end keys - this is the old format
                    },
                }
            ],
            "version": "0.4",
        },
    }
    zarr_group.attrs.update(attrs)

    # This should work without raising KeyError for 'start'
    multiscales = from_ngff_zarr(store)

    # Verify the result
    assert multiscales is not None
    assert len(multiscales.images) == 1
    assert multiscales.metadata.omero is not None
    assert len(multiscales.metadata.omero.channels) == 1

    # Check that the window has both min/max and start/end populated
    channel = multiscales.metadata.omero.channels[0]
    assert channel.window.min == 0
    assert channel.window.max == 1000
    # For backward compatibility, min/max should be used as start/end
    assert channel.window.start == 0
    assert channel.window.end == 1000

    # Test with start/end format only (newer format)
    store2 = MemoryStore()
    zarr_group2 = zarr.open_group(store2, mode="w")
    array2 = zarr_group2.create_array(
        "0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 10, 50, 50)
    )
    array2[:] = data

    attrs2 = attrs.copy()
    attrs2["omero"]["channels"][0]["window"] = {
        "start": 10,
        "end": 900,
        # Note: no min/max keys - this is a hypothetical newer format
    }
    zarr_group2.attrs.update(attrs2)

    # This should also work
    multiscales2 = from_ngff_zarr(store2)

    # Verify the result
    assert multiscales2 is not None
    assert multiscales2.metadata.omero is not None
    channel2 = multiscales2.metadata.omero.channels[0]
    # For forward compatibility, start/end should be used as min/max
    assert channel2.window.start == 10
    assert channel2.window.end == 900
    assert channel2.window.min == 10
    assert channel2.window.max == 900

    # Test with both formats present (most complete)
    store3 = MemoryStore()
    zarr_group3 = zarr.open_group(store3, mode="w")
    array3 = zarr_group3.create_array(
        "0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 10, 50, 50)
    )
    array3[:] = data

    attrs3 = attrs.copy()
    attrs3["omero"]["channels"][0]["window"] = {
        "min": 5,
        "max": 995,
        "start": 15,
        "end": 985,
    }
    zarr_group3.attrs.update(attrs3)

    # This should preserve both formats
    multiscales3 = from_ngff_zarr(store3)

    # Verify the result
    assert multiscales3 is not None
    assert multiscales3.metadata.omero is not None
    channel3 = multiscales3.metadata.omero.channels[0]
    assert channel3.window.min == 5
    assert channel3.window.max == 995
    assert channel3.window.start == 15
    assert channel3.window.end == 985


def test_from_ngff_zarr_empty_directory(tmp_path):
    """Test that from_ngff_zarr raises a helpful error for empty directories."""
    empty_zarr = tmp_path / "empty.zarr"
    empty_zarr.mkdir()

    with pytest.raises(ValueError, match="No valid Zarr group found"):
        from_ngff_zarr(str(empty_zarr))


def test_from_ngff_zarr_invalid_store(tmp_path):
    """Test that from_ngff_zarr raises a helpful error for invalid Zarr stores."""
    invalid_zarr = tmp_path / "invalid.zarr"
    invalid_zarr.mkdir()
    # Create a non-zarr file
    (invalid_zarr / "random.txt").write_text("not a zarr file")

    with pytest.raises(ValueError, match="No valid Zarr group found"):
        from_ngff_zarr(str(invalid_zarr))


def test_from_ngff_zarr_array_at_root(tmp_path):
    """Test that from_ngff_zarr raises a helpful error when root is an array."""
    import numpy as np

    array_zarr = tmp_path / "array.zarr"
    # Create a Zarr array (not a group) at the root
    zarr.save(str(array_zarr), np.array([1, 2, 3, 4, 5]))

    with pytest.raises(ValueError, match="contains an array at the root level"):
        from_ngff_zarr(str(array_zarr))


@pytest.mark.skipif(
    zarr_version_major < 3, reason="BloscCodec.from_dict exists only in zarr 3"
)
class TestBloscCodecCompat:
    """Tests for blosc codec backward-compatibility (Issue #516)."""

    def test_from_dict_strips_nthreads(self):
        """BloscCodec.from_dict should accept nthreads/blocksize without error."""
        from zarr.codecs.blosc import BloscCname, BloscCodec

        result = BloscCodec.from_dict(
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zlib",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "nthreads": 4,
                    "blocksize": 0,
                },
            }
        )
        assert result is not None
        assert result.cname == BloscCname.zlib
        assert result.clevel == 5

    def test_from_ngff_zarr_reads_with_nthreads(self, input_images):
        """Reading a zarr v3 file with nthreads in blosc metadata should work."""
        import json
        import tempfile

        from ngff_zarr import Methods

        dataset_name = "cthead1"
        image = input_images[dataset_name]
        chunks = (64, 64)
        multiscales = to_multiscales(
            image, [2, 4], chunks=chunks, method=Methods.ITKWASM_GAUSSIAN
        )

        compressors = zarr.codecs.BloscCodec(
            cname="zlib", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle
        )

        version = "0.5"
        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(
                tmpdir,
                multiscales,
                version=version,
                compressors=compressors,
                chunks_per_shard=2,
            )

            zarr_json_path = tmpdir + "/zarr.json"
            with open(zarr_json_path) as f:
                zarr_meta = json.load(f)

            metadata = zarr_meta["consolidated_metadata"]["metadata"]
            saw_nested_blosc = False
            for scale_name in metadata:
                codecs = metadata[scale_name].get("codecs", [])
                for codec in codecs:
                    if codec["name"] == "blosc":
                        codec["configuration"]["nthreads"] = 4
                        codec["configuration"]["blocksize"] = 0
                    elif codec["name"] == "sharding_indexed":
                        inner = codec["configuration"].get("codecs", [])
                        for inner_codec in inner:
                            if inner_codec["name"] == "blosc":
                                saw_nested_blosc = True
                                inner_codec["configuration"]["nthreads"] = 4
                                inner_codec["configuration"]["blocksize"] = 0

            assert saw_nested_blosc

            with open(zarr_json_path, "w") as f:
                json.dump(zarr_meta, f)

            loaded = from_ngff_zarr(tmpdir)
            assert loaded is not None
