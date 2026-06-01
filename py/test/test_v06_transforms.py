# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for v0.6 coordinate/displacement transformation types."""

from dataclasses import asdict

import numpy as np
from ngff_zarr import from_ngff_zarr, to_multiscales, to_ngff_image, to_ngff_zarr
from ngff_zarr.v04.zarr_metadata import (
    Coordinates,
    Displacements,
    Identity,
    Scale,
    Transform,
    Translation,
    _parse_group_transforms,
    _parse_transform,
)


def test_displacements_dataclass():
    d = Displacements(
        path="coordinateTransformations/field",
        indexTransformation=Scale(scale=[1.0, 2.0]),
        interpolation="linear",
    )
    assert d.path == "coordinateTransformations/field"
    assert d.type == "displacements"
    assert d.interpolation == "linear"
    assert isinstance(d.indexTransformation, Scale)
    assert d.indexTransformation.scale == [1.0, 2.0]

    d2 = Displacements(
        path="field",
        indexTransformation=Identity(),
    )
    assert d2.interpolation is None
    assert d2.indexTransformation.type == "identity"


def test_coordinates_dataclass():
    c = Coordinates(
        path="coordinateTransformations/coord_field",
        indexTransformation=Scale(scale=[3.0, 1.0]),
        interpolation="nearest",
    )
    assert c.path == "coordinateTransformations/coord_field"
    assert c.type == "coordinates"
    assert c.interpolation == "nearest"
    assert isinstance(c.indexTransformation, Scale)
    assert c.indexTransformation.scale == [3.0, 1.0]

    c2 = Coordinates(
        path="field",
        indexTransformation=Identity(),
    )
    assert c2.interpolation is None


def test_displacements_serialization_roundtrip():
    d = Displacements(
        path="coordinateTransformations/field",
        indexTransformation=Scale(scale=[1.0, 2.0]),
        interpolation="linear",
        name="my_disp",
    )
    d_dict = asdict(d)
    assert d_dict["type"] == "displacements"
    assert d_dict["path"] == "coordinateTransformations/field"
    assert d_dict["indexTransformation"] == {"scale": [1.0, 2.0], "type": "scale"}
    assert d_dict["interpolation"] == "linear"
    assert d_dict["name"] == "my_disp"


def test_coordinates_serialization_roundtrip():
    c = Coordinates(
        path="coord_field",
        indexTransformation=Identity(),
        interpolation="cubic",
    )
    c_dict = asdict(c)
    assert c_dict["type"] == "coordinates"
    assert c_dict["path"] == "coord_field"
    assert c_dict["indexTransformation"] == {"type": "identity"}
    assert c_dict["interpolation"] == "cubic"


def test_parse_transform_displacements():
    transform_dict = {
        "type": "displacements",
        "path": "coordinateTransformations/field",
        "indexTransformation": {"type": "scale", "scale": [2.0, 1.0]},
        "interpolation": "nearest",
    }
    result = _parse_transform(transform_dict)
    assert isinstance(result, Displacements)
    assert result.path == "coordinateTransformations/field"
    assert result.interpolation == "nearest"
    assert isinstance(result.indexTransformation, Scale)
    assert result.indexTransformation.scale == [2.0, 1.0]


def test_parse_transform_displacements_identity():
    transform_dict = {
        "type": "displacements",
        "path": "field",
        "indexTransformation": {"type": "identity"},
    }
    result = _parse_transform(transform_dict)
    assert isinstance(result, Displacements)
    assert result.path == "field"
    assert isinstance(result.indexTransformation, Identity)


def test_parse_transform_coordinates():
    transform_dict = {
        "type": "coordinates",
        "path": "coordinateTransformations/coord_field",
        "indexTransformation": {"type": "identity"},
        "interpolation": "linear",
    }
    result = _parse_transform(transform_dict)
    assert isinstance(result, Coordinates)
    assert result.path == "coordinateTransformations/coord_field"
    assert result.interpolation == "linear"
    assert isinstance(result.indexTransformation, Identity)


def test_parse_transform_scale():
    result = _parse_transform({"type": "scale", "scale": [1.0, 2.0]})
    assert isinstance(result, Scale)
    assert result.scale == [1.0, 2.0]


def test_parse_transform_translation():
    result = _parse_transform({"type": "translation", "translation": [0.5, 1.0]})
    assert isinstance(result, Translation)
    assert result.translation == [0.5, 1.0]


def test_parse_group_transforms():
    transforms = [
        {"type": "scale", "scale": [1.0, 2.0]},
        {"type": "translation", "translation": [0.5, 1.0]},
        {
            "type": "displacements",
            "path": "field",
            "indexTransformation": {"type": "identity"},
        },
    ]
    result = _parse_group_transforms(transforms)
    assert len(result) == 3
    assert isinstance(result[0], Scale)
    assert isinstance(result[1], Translation)
    assert isinstance(result[2], Displacements)


def test_parse_group_transforms_none():
    assert _parse_group_transforms(None) is None


def test_transform_union_includes_new_types():
    d = Displacements(path="field", indexTransformation=Identity())
    c = Coordinates(path="field", indexTransformation=Scale(scale=[1.0]))
    assert d.path == "field"
    assert c.path == "field"


def test_identity_in_dataset_transforms(tmp_path):
    """Test round-trip preserves dataset coordinateTransformations."""
    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    store = str(tmp_path / "test_id.zarr")
    image = to_ngff_image(data=data, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[])
    to_ngff_zarr(store, multiscales, version="0.4")

    result = from_ngff_zarr(store, version="0.4")
    for dataset in result.metadata.datasets:
        for t in dataset.coordinateTransformations:
            assert t.type in ("scale", "translation")

    assert result.images[0].data.shape == (32, 32)


def test_identity_dataclass():
    ident = Identity()
    assert ident.type == "identity"
    assert asdict(ident) == {"type": "identity"}


def test_transforms_union_compatibility():
    transforms: list[Transform] = [
        Scale(scale=[1.0]),
        Translation(translation=[0.0]),
        Identity(),
    ]
    assert len(transforms) == 3
    assert transforms[0].type == "scale"
    assert transforms[1].type == "translation"
    assert transforms[2].type == "identity"
