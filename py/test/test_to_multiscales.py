# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from ngff_zarr import Methods
from ngff_zarr.to_multiscales import to_multiscales
from ngff_zarr.to_ngff_image import to_ngff_image

rng = np.random.default_rng(12345)


@pytest.mark.parametrize(
    "shape, chunk_shape",
    [
        (
            (1, 30, 1024, 1024),
            (1, 30, 65, 65),
        ),
        (
            (1, 125, 1024, 1024),
            (1, 50, 51, 50),
        ),
    ],
)
def test_to_multiscales_metadata_synced_with_data(shape, chunk_shape):
    array = rng.random(size=shape, dtype=np.float32) * 100.0
    input_image = to_ngff_image(array, dims=["t", "z", "y", "x"])
    multiscales = to_multiscales(
        input_image, scale_factors=max(chunk_shape), chunks=chunk_shape
    )
    for i, dataset in enumerate(multiscales.metadata.datasets):
        image = multiscales.images[i]
        toplevel_meta_scale = dataset.coordinateTransformations[0].scale

        image_scale_spatial_only = [image.scale[d] for d in ["z", "y", "x"]]
        assert image_scale_spatial_only == toplevel_meta_scale[1:]

        # Assert scale factors are applied to the correct dimensions
        assert image.data.shape[2] == image.data.shape[3]  # 512 != 1024


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "uint16", "int16", "int32"],
)
@pytest.mark.parametrize(
    "method",
    [Methods.DASK_IMAGE_GAUSSIAN, Methods.ITKWASM_GAUSSIAN],
)
def test_downsample_preserves_uniform_integer_data(dtype, method):
    """Downsampling uniform integer data should preserve the constant value."""
    data = np.ones((1, 2, 32, 32, 32), dtype=dtype)
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[2, 4], chunks=(1, 1, 16, 16, 16), method=method
    )
    for scale_idx, img in enumerate(multiscales.images):
        computed = img.data.compute()
        assert np.all(computed == 1), (
            f"Scale {scale_idx} with {method.value} on {dtype}: "
            f"expected all 1s, got min={computed.min()}, max={computed.max()}"
        )


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "uint16"],
)
@pytest.mark.parametrize(
    "method",
    [Methods.DASK_IMAGE_GAUSSIAN, Methods.ITKWASM_GAUSSIAN],
)
def test_downsample_preserves_zero_integer_data(dtype, method):
    """Downsampling all-zero integer data should produce all zeros."""
    data = np.zeros((1, 2, 32, 32, 32), dtype=dtype)
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[2, 4], chunks=(1, 1, 16, 16, 16), method=method
    )
    for scale_idx, img in enumerate(multiscales.images):
        computed = img.data.compute()
        assert np.all(computed == 0), (
            f"Scale {scale_idx} with {method.value} on {dtype}: "
            f"expected all 0s, got min={computed.min()}, max={computed.max()}"
        )


def test_downsample_preserves_dtype_uint16():
    """Downsampled output dtype should match input dtype for uint16."""
    data = np.ones((1, 2, 32, 32, 32), dtype="uint16")
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=(1, 1, 16, 16, 16))
    for img in multiscales.images:
        computed = img.data.compute()
        assert computed.dtype == np.uint16, f"Expected uint16, got {computed.dtype}"


def test_downsample_preserves_dtype_float64():
    """Downsampled output dtype should match input dtype for float64."""
    data = np.ones((1, 2, 32, 32, 32), dtype="float64")
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=(1, 1, 16, 16, 16))
    for img in multiscales.images:
        computed = img.data.compute()
        assert computed.dtype == np.float64, f"Expected float64, got {computed.dtype}"


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_to_ngff_zarr_roundtrip_preserves_integer_data(version):
    """Write/read round-trip should preserve integer data integrity."""
    import os
    import tempfile

    from ngff_zarr import from_ngff_zarr, to_ngff_zarr

    data = np.ones((1, 2, 50, 60, 70), dtype="uint16")
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=(1, 1, 16, 16, 16))

    td = os.path.join(tempfile.mkdtemp(), "test.zarr")
    to_ngff_zarr(td, multiscales, version=version)

    loaded = from_ngff_zarr(td)
    for scale_idx, img in enumerate(loaded.images):
        computed = np.array(img.data)
        assert np.all(computed == 1), (
            f"Scale {scale_idx} v{version}: expected all 1s, "
            f"got min={computed.min()}, max={computed.max()}"
        )
