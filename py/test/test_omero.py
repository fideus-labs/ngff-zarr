# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import pytest
import numpy as np
from zarr.storage import MemoryStore
from ngff_zarr import (
    Omero,
    OmeroChannel,
    OmeroWindow,
    from_ngff_zarr,
    to_ngff_image,
    to_multiscales,
    to_ngff_zarr,
)

from ._data import test_data_dir


def test_read_omero(input_images):  # noqa: ARG001
    dataset_name = "13457537"
    store_path = test_data_dir / "input" / f"{dataset_name}.zarr"
    multiscales = from_ngff_zarr(store_path, validate=True)

    omero = multiscales.metadata.omero
    assert omero is not None
    assert len(omero.channels) == 6

    # Channel 0
    assert omero.channels[0].color == "FFFFFF"
    assert omero.channels[0].window.min == 0.0
    assert omero.channels[0].window.max == 65535.0
    assert omero.channels[0].window.start == 0.0
    assert omero.channels[0].window.end == 1200.0
    assert omero.channels[0].label == "cy 1"

    # Channel 1
    assert omero.channels[1].color == "FFFFFF"
    assert omero.channels[1].window.min == 0.0
    assert omero.channels[1].window.max == 65535.0
    assert omero.channels[1].window.start == 0.0
    assert omero.channels[1].window.end == 1200.0
    assert omero.channels[1].label == "cy 2"

    # Channel 2
    assert omero.channels[2].color == "FFFFFF"
    assert omero.channels[2].window.min == 0.0
    assert omero.channels[2].window.max == 65535.0
    assert omero.channels[2].window.start == 0.0
    assert omero.channels[2].window.end == 1200.0
    assert omero.channels[2].label == "cy 3"

    # Channel 3
    assert omero.channels[3].color == "FFFFFF"
    assert omero.channels[3].window.min == 0.0
    assert omero.channels[3].window.max == 65535.0
    assert omero.channels[3].window.start == 0.0
    assert omero.channels[3].window.end == 1200.0
    assert omero.channels[3].label == "cy 4"

    # Channel 4
    assert omero.channels[4].color == "0000FF"
    assert omero.channels[4].window.min == 0.0
    assert omero.channels[4].window.max == 65535.0
    assert omero.channels[4].window.start == 0.0
    assert omero.channels[4].window.end == 5000.0
    assert omero.channels[4].label == "DAPI"

    # Channel 5
    assert omero.channels[5].color == "FF0000"
    assert omero.channels[5].window.min == 0.0
    assert omero.channels[5].window.max == 65535.0
    assert omero.channels[5].window.start == 0.0
    assert omero.channels[5].window.end == 100.0
    assert omero.channels[5].label == "Hyb probe"


def test_write_omero():
    data = np.random.randint(0, 256, 262144).reshape((2, 32, 64, 64)).astype(np.uint8)
    image = to_ngff_image(data, dims=["c", "z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2, 4], chunks=32)

    omero = Omero(
        channels=[
            OmeroChannel(
                color="008000",
                window=OmeroWindow(min=0.0, max=255.0, start=10.0, end=150.0),
                label="Phalloidin",
            ),
            OmeroChannel(
                color="0000FF",
                window=OmeroWindow(min=0.0, max=255.0, start=30.0, end=200.0),
                label="",
            ),
        ]
    )
    multiscales.metadata.omero = omero

    store = MemoryStore()
    version = "0.4"
    to_ngff_zarr(store, multiscales, version=version)

    multiscales_read = from_ngff_zarr(store, validate=True, version=version)
    read_omero = multiscales_read.metadata.omero

    assert read_omero is not None
    assert len(read_omero.channels) == 2
    assert read_omero.channels[0].color == "008000"
    assert read_omero.channels[0].window.start == 10.0
    assert read_omero.channels[0].window.end == 150.0
    assert read_omero.channels[0].label == "Phalloidin"
    assert read_omero.channels[1].color == "0000FF"
    assert read_omero.channels[1].window.start == 30.0
    assert read_omero.channels[1].window.end == 200.0
    assert read_omero.channels[1].label == ""


def test_validate_color():
    valid_channel = OmeroChannel(
        color="1A2B3C", window=OmeroWindow(min=0.0, max=255.0, start=0.0, end=100.0)
    )
    # This should not raise an error
    valid_channel.validate_color()

    invalid_channel = OmeroChannel(
        color="ZZZZZZ", window=OmeroWindow(min=0.0, max=255.0, start=0.0, end=100.0)
    )
    with pytest.raises(ValueError, match=r"Invalid color 'ZZZZZZ'"):
        invalid_channel.validate_color()
