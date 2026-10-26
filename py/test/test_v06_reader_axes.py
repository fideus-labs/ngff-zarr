# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The v0.6 reader takes an image's axes from the system its datasets map to."""

import json

import numpy as np
import pytest
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr


def _edit_v06_store(tmp_path, edit):
    store = tmp_path / "image.ome.zarr"
    image = to_ngff_image(
        np.zeros((4, 8, 8), np.uint8),
        dims=("z", "y", "x"),
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
    )
    to_ome_zarr(store, to_multiscales(image, scale_factors=[]), version="0.6")
    doc_path = store / "zarr.json"
    doc = json.loads(doc_path.read_text())
    edit(doc["attributes"]["ome"]["multiscales"][0])
    doc_path.write_text(json.dumps(doc))
    return store


def test_intrinsic_system_listed_second(tmp_path):
    def world_first(multiscales):
        world = {"name": "world", "axes": [{"name": "y"}, {"name": "x"}]}
        multiscales["coordinateSystems"].insert(0, world)

    image = from_ome_zarr(_edit_v06_store(tmp_path, world_first)).images[0]
    assert tuple(image.dims) == ("z", "y", "x")
    assert image.scale == {"z": 2.0, "y": 1.0, "x": 1.0}


def test_axis_without_type_or_with_long_name(tmp_path):
    def loosen_axes(multiscales):
        axes = multiscales["coordinateSystems"][0]["axes"]
        del axes[0]["type"]
        axes[1]["longName"] = "rows"

    image = from_ome_zarr(_edit_v06_store(tmp_path, loosen_axes)).images[0]
    assert tuple(image.dims) == ("z", "y", "x")


def test_undeclared_output_system_names_it(tmp_path):
    def rename_output(multiscales):
        sequence = multiscales["datasets"][0]["coordinateTransformations"][0]
        sequence["output"]["name"] = "elsewhere"

    with pytest.raises(ValueError, match="elsewhere"):
        from_ome_zarr(_edit_v06_store(tmp_path, rename_output))
