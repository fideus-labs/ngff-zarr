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


def test_from_ngff_zarr(input_images, tmp_path):
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
    test_store = str(tmp_path / f"{dataset_name}.zarr")
    version = "0.4"
    to_ngff_zarr(test_store, multiscales, version=version)

    multiscales_back = from_ngff_zarr(test_store, version=version)
    # store_new_multiscales(dataset_name, baseline_name, multiscales)
    verify_against_baseline(
        dataset_name, baseline_name, multiscales_back, version=version
    )


def test_omero_zarr_from_ngff_zarr_to_ngff_zarr(input_images, tmp_path):
    dataset_name = "13457537"
    store_path = test_data_dir / "input" / f"{dataset_name}.zarr"
    version = "0.4"
    multiscales = from_ngff_zarr(store_path, version=version)
    test_store = str(tmp_path / f"{dataset_name}.zarr")
    to_ngff_zarr(test_store, multiscales, version=version)


@pytest.mark.skipif(
    zarr_version_major < 3, reason="storage_options requires zarr-python 3"
)
def test_from_ngff_zarr_with_storage_options(input_images, tmp_path):
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

    # Test with a local directory store (should work as before)
    test_store = str(tmp_path / f"{dataset_name}.zarr")
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


def test_from_ngff_zarr_string_url_with_storage_options():
    """String URLs forward storage_options into the remote store handle.

    Remote URLs read through zarrista's async API over obstore;
    ``from_ngff_zarr`` wraps the URL in a ``RemoteZarrStore`` that carries
    the caller's fsspec-style ``storage_options`` (translated to obstore
    configuration inside the handle; the translation itself is pinned in
    ``test_remote_zarrista.py``). The legacy zarr-python/fsspec fallback
    this test used to pin was removed with zarr-python store support; an
    unavailable remote engine now raises ImportError up-front instead.
    """
    from unittest.mock import patch

    from ngff_zarr._remote_reader import remote_read_available

    if not remote_read_available():
        pytest.skip("zarrista/obstore remote engine unavailable")

    captured = {}

    def capture_open_remote_node(store, component=None):
        captured["store"] = store
        return  # no node here -> "No valid Zarr group found"

    url = "s3://test-bucket/test-dataset.zarr"
    storage_options = {"anon": True, "region_name": "us-west-2"}
    with patch("ngff_zarr.from_ngff_zarr.open_remote_node", capture_open_remote_node):
        with pytest.raises(ValueError, match="No valid Zarr group found"):
            from_ngff_zarr(url, storage_options=storage_options)

    store = captured["store"]
    assert store.url == url
    assert store.storage_options == storage_options


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
    """A missing remote engine for any remote scheme yields an actionable error.

    When the zarrista/obstore remote reader is unavailable (Python 3.10, or
    obstore not installed) there is no fallback engine anymore: the read
    fails up-front, before any network access, with an ImportError pointing
    at the ``[remote]`` extra.
    """
    from unittest.mock import patch

    with patch("ngff_zarr.from_ngff_zarr.remote_read_available", return_value=False):
        with pytest.raises(ImportError) as exc_info:
            from_ngff_zarr(url)

    message = str(exc_info.value)
    assert "ngff-zarr[remote]" in message
    assert url in message


@pytest.mark.skipif(
    zarr_version_major < 3,
    reason="zarr-python 2.x stores are MutableMappings, a supported input form",
)
def test_from_ngff_zarr_rejects_zarr_python_store_objects():
    """Passing a zarr-python store object raises a helpful TypeError.

    This used to pin the remote-only ImportError-rewrite guard by reading
    through a ``MemoryStore``. Store objects no longer reach any read
    engine (the remote rewrite applies only to URL strings, before the
    store is opened), so that scenario is unreachable; this now pins the
    breaking-change contract instead: reads accept local directory paths,
    remote URL strings, ``.ozx``/zip paths, and key-to-bytes mappings --
    never zarr-python store objects.
    """
    with pytest.raises(
        TypeError, match="Pass a path instead of a zarr-python store object"
    ):
        from_ngff_zarr(MemoryStore())


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

    from ngff_zarr._remote_reader import remote_read_available

    if not os.environ.get("NGFF_ZARR_NETWORK_TESTS"):
        pytest.skip("live remote read is opt-in; set NGFF_ZARR_NETWORK_TESTS=1 to run")
    if not remote_read_available():
        # Legacy zarr-python/fsspec engine (Python 3.10 or no obstore).
        pytest.importorskip("aiohttp", reason="aiohttp required to read https stores")

    url = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
    try:
        multiscales = from_ngff_zarr(url)
    except OSError as exc:
        # Connection refused / DNS / timeout: clearly the network, not the code.
        pytest.skip(f"remote store unavailable: {exc!r}")
    except Exception as exc:
        # zarrista folds transport failures into its own exception types;
        # store-absence instead surfaces as the group-not-found ValueError,
        # which must keep failing the test.
        if type(exc).__module__.startswith("zarrista"):
            pytest.skip(f"remote store unavailable: {exc!r}")
        raise

    # Metadata is read eagerly; pixel data stays lazy, so this stays cheap.
    assert len(multiscales.images) > 0
    assert multiscales.images[0].data.ndim > 0


@pytest.mark.skipif(
    zarr_version_major < 3, reason="storage_options requires zarr-python 3"
)
def test_omero_metadata_backward_compatibility(tmp_path):
    """Test that OMERO metadata with only min/max or only start/end is handled correctly."""
    import numpy as np
    from ngff_zarr import from_ngff_zarr

    # Create a test store with OMERO metadata using only min/max (old format)
    store = str(tmp_path / "min_max.zarr")
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
    store2 = str(tmp_path / "start_end.zarr")
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
    store3 = str(tmp_path / "both_formats.zarr")
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

    def test_import_does_not_patch_blosc_from_dict(self):
        """Importing ngff_zarr leaves zarr-python's BloscCodec unpatched.

        ngff-zarr used to monkey-patch ``BloscCodec.from_dict`` at import
        time to strip the legacy ``nthreads``/``blocksize`` keys before
        zarr-python parsed codec metadata (and this test pinned that shim).
        Reads now go through the zarrista compat layer, zarr-python is only
        a test dependency, and the shim is gone: importing ngff_zarr must
        not mutate zarr-python internals. The user-facing tolerance for the
        legacy keys is covered by the integration test below.
        """
        import ngff_zarr  # noqa: F401
        from zarr.codecs.blosc import BloscCodec

        defining_module = BloscCodec.from_dict.__func__.__module__
        assert defining_module.startswith("zarr."), (
            f"BloscCodec.from_dict is patched (defined in {defining_module})"
        )

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

            # Chunk decoding is likewise unaffected by the legacy keys.
            import numpy as np

            np.testing.assert_array_equal(
                loaded.images[0].data[:8, :8].compute(),
                multiscales.images[0].data[:8, :8].compute(),
            )
