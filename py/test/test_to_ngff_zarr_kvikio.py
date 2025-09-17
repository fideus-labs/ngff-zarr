# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import pytest
from ngff_zarr import Methods, to_multiscales, to_ngff_zarr
from ngff_zarr._zarr_kwargs import zarr_kwargs

pytest.importorskip("kvikio")
pytest.importorskip("itkwasm_downsample_cucim")


def test_bin_shrink_isotropic_scale_factors(input_images, tmp_path):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITKWASM_BIN_SHRINK.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_BIN_SHRINK)

    from kvikio.zarr import GDSStore

    store = GDSStore(tmp_path / baseline_name, **zarr_kwargs)
    from kvikio.nvcomp_codec import NvCompBatchCodec

    compressor = NvCompBatchCodec("lz4")
    to_ngff_zarr(store, multiscales, compressor=compressor)


def test_gaussian_isotropic_scale_factors(input_images, tmp_path):
    dataset_name = "cthead1"
    image = input_images[dataset_name]
    baseline_name = "2_4/ITKWASM_GAUSSIAN.zarr"
    multiscales = to_multiscales(image, [2, 4], method=Methods.ITKWASM_GAUSSIAN)

    from kvikio.zarr import GDSStore

    store = GDSStore(tmp_path / baseline_name, **zarr_kwargs)
    from kvikio.nvcomp_codec import NvCompBatchCodec

    compressor = NvCompBatchCodec("zstd")
    to_ngff_zarr(store, multiscales, compressor=compressor)
