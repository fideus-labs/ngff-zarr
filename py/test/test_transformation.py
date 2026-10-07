# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A transformation on its own, read and written with kind="transformation"."""

import json

import numpy as np
import pytest
from ngff_zarr import (
    AxisType,
    CoordinateSystemIdentifier,
    NgffImage,
    NgffMultiscales,
    from_ome_zarr,
    to_multiscales,
    to_ome_zarr,
)
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Displacements,
    Scale,
    TransformSequence,
    Translation,
)


def _affine() -> Affine:
    return Affine(
        affine=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        input=CoordinateSystemIdentifier(name="ji"),
        output=CoordinateSystemIdentifier(name="yx"),
        name="ji to yx",
    )


def _root_ome(store) -> dict:
    return json.loads((store / "zarr.json").read_text())["attributes"]["ome"]


def test_transformation_round_trip(tmp_path):
    store = tmp_path / "affine.ome.zarr"
    to_ome_zarr(store, _affine())

    ome = _root_ome(store)
    assert ome["version"] == "0.6"
    assert ome["coordinateTransformations"] == [
        {
            "type": "affine",
            "affine": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "input": {"name": "ji"},
            "output": {"name": "yx"},
            "name": "ji to yx",
        }
    ]
    assert from_ome_zarr(store, kind="transformation") == _affine()


def test_nested_transformation_round_trips(tmp_path):
    sequence = TransformSequence(
        transformations=[Translation(translation=[0.1, 0.9]), Scale(scale=[2.0, 3.0])],
        input=CoordinateSystemIdentifier(name="in"),
        output=CoordinateSystemIdentifier(name="out"),
        name="in to out",
    )
    store = tmp_path / "sequence.ome.zarr"
    to_ome_zarr(store, sequence, version="0.9.dev1")
    assert _root_ome(store)["version"] == "0.9.dev1"
    read = from_ome_zarr(store, kind="transformation", validate=True)
    assert read == sequence


def test_the_kind_option_selects_what_a_store_holds(tmp_path):
    store = tmp_path / "affine.ome.zarr"
    to_ome_zarr(store, _affine())
    with pytest.raises(ValueError, match="kind='transformation'"):
        from_ome_zarr(store)
    with pytest.raises(ValueError, match="kind='transformation'"):
        from_ome_zarr(store, kind="scene")

    image = to_multiscales(np.zeros((8, 8), dtype=np.uint8), scale_factors=[])
    image_store = tmp_path / "image.ome.zarr"
    to_ome_zarr(image_store, image, version="0.6")
    with pytest.raises(ValueError, match="holds no transformation"):
        from_ome_zarr(image_store, kind="transformation")
    with pytest.raises(ValueError, match="written to a directory path"):
        to_ome_zarr(tmp_path / "affine.ozx", _affine())
    with pytest.raises(ValueError, match="0.6"):
        to_ome_zarr(tmp_path / "affine05.ome.zarr", _affine(), version="0.5")


def test_transformation_with_a_displacement_field(tmp_path):
    import dask.array as da

    field = NgffImage(
        data=da.zeros((2, 8, 8), dtype=np.float32),
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": 0.5, "x": 0.5},
        translation={"c": 0.0, "y": 0.0, "x": 0.0},
        axes_types={"c": AxisType.Displacement},
    )
    warp = Displacements(
        path="coordinateTransformations/dfield",
        interpolation="linear",
        input=CoordinateSystemIdentifier(name="fixed"),
        output=CoordinateSystemIdentifier(name="moving"),
    )
    store = tmp_path / "warp.ome.zarr"
    with pytest.raises(ValueError, match="write those first"):
        to_ome_zarr(store, warp)
    with pytest.raises(ValueError, match="which the store does not hold"):
        to_ome_zarr(store, warp, overwrite=False)
    assert not store.exists()

    to_ome_zarr(
        store / warp.path, to_multiscales(field, scale_factors=[]), version="0.6"
    )
    to_ome_zarr(store, warp, overwrite=False)
    document = json.loads((store / "zarr.json").read_text())
    assert "coordinateTransformations" in document["consolidated_metadata"]["metadata"]
    assert f"{warp.path}/scale0/image" in document["consolidated_metadata"]["metadata"]

    read = from_ome_zarr(store, kind="transformation")
    assert read == warp
    field_read = from_ome_zarr(store / read.path)
    assert isinstance(field_read, NgffMultiscales)
    assert field_read.metadata.coordinateSystems[0].axes[0].type == "displacement"


@pytest.mark.parametrize("path", ["../dfield", "..\\dfield", "%2e%2e/dfield"])
def test_paths_outside_the_store_are_refused(tmp_path, path):
    warp = Displacements(
        path=path,
        input=CoordinateSystemIdentifier(name="fixed"),
        output=CoordinateSystemIdentifier(name="moving"),
    )
    with pytest.raises(ValueError, match="relative path below the scene group"):
        to_ome_zarr(tmp_path / "warp.ome.zarr", warp, overwrite=False)

    store = tmp_path / "affine.ome.zarr"
    to_ome_zarr(store, _affine())
    document = json.loads((store / "zarr.json").read_text())
    document["attributes"]["ome"]["coordinateTransformations"] = [
        {"type": "displacements", "path": path}
    ]
    (store / "zarr.json").write_text(json.dumps(document))
    assert from_ome_zarr(store, kind="transformation").path == path
    with pytest.raises(ValueError, match="relative path below the scene group"):
        from_ome_zarr(store, kind="transformation", validate=True)
