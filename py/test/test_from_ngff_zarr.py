# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dask_image import imread
from ngff_zarr import (
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)
from zarr.storage import MemoryStore
import packaging.version
import zarr
import pytest

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


def test_omero_zarr_from_ngff_zarr_to_ngff_zarr(input_images):  # noqa: ARG001
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
    from unittest.mock import patch, MagicMock

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


@pytest.mark.skipif(
    zarr_version_major < 3, reason="storage_options requires zarr-python 3"
)
def test_omero_metadata_backward_compatibility():
    """Test that OMERO metadata with only min/max or only start/end is handled correctly."""
    import numpy as np
    from zarr.storage import MemoryStore
    from ngff_zarr import from_ngff_zarr

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
