# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Multiscale-level coordinateTransformations read as transform objects."""

import json

import numpy as np
import pytest
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_multiscale_level_scale_validates(tmp_path, version):
    store = tmp_path / "image.ome.zarr"
    image = to_ngff_image(np.zeros((8, 8), np.uint8), dims=("y", "x"))
    to_ome_zarr(store, to_multiscales(image, scale_factors=[]), version=version)

    doc_path = store / (".zattrs" if version == "0.4" else "zarr.json")
    doc = json.loads(doc_path.read_text())
    attrs = doc if version == "0.4" else doc["attributes"]["ome"]
    multiscales = attrs["multiscales"][0]
    multiscales["coordinateTransformations"] = [{"type": "scale", "scale": [2.0, 3.0]}]
    doc_path.write_text(json.dumps(doc))

    multiscales = from_ome_zarr(store, validate=True)
    assert multiscales.images[0].data.shape == (8, 8)
