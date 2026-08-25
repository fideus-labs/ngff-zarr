# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The dataclasses accept what the bundled schema of their version declares."""

import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import NgffImage, from_ome_zarr, to_multiscales, to_ome_zarr, validate
from ngff_zarr.parse_metadata import _parse_omero
from ngff_zarr.v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow


def _write_06(tmp_path):
    image = NgffImage(
        da.zeros((2, 8, 8), dtype=np.uint8, chunks=(2, 8, 8)),
        ["c", "y", "x"],
        {"c": 1.0, "y": 1.0, "x": 1.0},
        {"c": 0.0, "y": 0.0, "x": 0.0},
    )
    store = str(tmp_path / "s.ome.zarr")
    to_ome_zarr(
        store, to_multiscales(image, scale_factors=[], cache=False), version="0.6"
    )
    return store


def _patch_first_axis(store, mutate):
    doc_path = Path(store) / "zarr.json"
    doc = json.loads(doc_path.read_text())
    systems = doc["attributes"]["ome"]["multiscales"][0]["coordinateSystems"]
    mutate(systems[0]["axes"][0])
    doc_path.write_text(json.dumps(doc, indent=4))
    return doc["attributes"]


@pytest.mark.parametrize(
    "mutate, description",
    [
        (lambda axis: axis.update(longName="channel index"), "longName"),
        (lambda axis: axis.pop("type", None), "no type"),
    ],
)
def test_the_0_6_reader_accepts_what_its_schema_declares(tmp_path, mutate, description):
    """The 0.6 axes schema requires only ``name`` and declares ``longName``."""
    store = _write_06(tmp_path)
    attrs = _patch_first_axis(store, mutate)

    validate(attrs, version="0.6")
    image = from_ome_zarr(store).images[0]
    assert image.data.shape == (2, 8, 8), description


def test_an_unknown_axis_key_is_dropped_with_a_warning(tmp_path):
    """The 0.6 axes schema sets no additionalProperties, so a later release may
    add a property this version does not model."""
    store = _write_06(tmp_path)
    _patch_first_axis(store, lambda axis: axis.update(madeUp=1))

    with pytest.warns(UserWarning, match="madeUp"):
        assert len(from_ome_zarr(store).images) == 1


def test_every_omero_channel_keeps_its_place():
    """The channel list is positional, so dropping one renumbers the rest."""
    omero = _parse_omero(
        {
            "version": "0.4",
            "channels": [
                {
                    "color": "FF0000",
                    "window": {"min": 0, "max": 255, "start": 0, "end": 255},
                    "label": "first",
                    "active": True,
                },
                {"color": "00FF00", "label": "no window"},
                {"window": {"min": 0, "max": 1}, "label": "no color"},
            ],
        }
    )
    assert [channel.label for channel in omero.channels] == [
        "first",
        "no window",
        "no color",
    ]
    assert omero.version == "0.4"
    assert omero.channels[0].active is True
    assert omero.channels[1].window is None
    assert omero.channels[2].color is None
    # min/max standing in for start/end, as this library has always read them
    assert omero.channels[2].window == OmeroWindow(min=0.0, max=1.0, start=0.0, end=1.0)


def test_a_channel_without_a_color_passes_the_color_rule():
    """Whether the field may be absent is the schema's decision."""
    OmeroChannel().validate_color()
    with pytest.raises(ValueError, match="6 hex digits"):
        OmeroChannel(color="nope").validate_color()


def test_omero_defaults_are_writable():
    """An Omero built with no channel detail still serializes."""
    assert Omero(channels=[OmeroChannel()]).channels[0].window is None
