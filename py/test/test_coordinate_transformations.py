# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import tempfile

import ngff_zarr as nz
import numpy as np
import pytest
import zarr
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Axis,
    Bijection,
    ByDimension,
    ByDimensionItem,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Displacements,
    Identity,
    MapAxis,
    Rotation,
    Scale,
    TransformSequence,
    Translation,
    validate_transform,
)
from packaging import version

rng = np.random.default_rng(12345)

# OME-Zarr v0.6 requires a Zarr v3 store, which zarr-python only writes at
# >= 3.0.0b1. The CI matrix exercises legs pinned to zarr 2.x (e.g. Python
# 3.10), where ``to_ngff_zarr(..., version="0.6")`` raises; skip there.
requires_zarr_v3 = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.6 requires zarr-python >= 3.0.0b1",
)


def identity_transform() -> Identity:
    return Identity()


def scale_transform() -> Scale:
    return Scale(scale=[2.0, 2.0, 2.0])


def translation_transform() -> Translation:
    return Translation(translation=[10.0, 20.0, 30.0])


def rotation_transform() -> Rotation:
    return Rotation(
        rotation=[
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def affine_transform() -> Affine:
    return Affine(
        affine=[
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 15.0],
        ]
    )


def transform_sequence() -> TransformSequence:
    return TransformSequence(
        transformations=[
            scale_transform(),
            translation_transform(),
            rotation_transform(),
            affine_transform(),
        ]
    )


def map_axis_transform() -> MapAxis:
    return MapAxis(mapAxis=[2, 0, 1])


def by_dimension_transform() -> ByDimension:
    return ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                input_axes=[0, 1],
                output_axes=[0, 1],
            ),
            ByDimensionItem(
                transformation=Translation(translation=[5.0]),
                input_axes=[2],
                output_axes=[2],
            ),
        ]
    )


def bijection_transform() -> Bijection:
    return Bijection(
        forward=Displacements(path="forward_field"),
        inverse=Displacements(path="inverse_field"),
    )


@requires_zarr_v3
@pytest.mark.parametrize(
    "transform",
    [
        identity_transform(),
        scale_transform(),
        translation_transform(),
        rotation_transform(),
        affine_transform(),
        transform_sequence(),
        map_axis_transform(),
        by_dimension_transform(),
        bijection_transform(),
    ],
)
def test_transform_serialization(transform):

    array = rng.random(size=(128, 128, 128), dtype=np.float32)
    input_image = nz.to_ngff_image(
        array,
        dims=["z", "y", "x"],
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )
    multiscales = nz.to_multiscales(input_image, scale_factors=[2], chunks=(64, 64, 64))
    assert "intrinsic" in [cs.name for cs in multiscales.metadata.coordinateSystems]

    # add output coordinate system to multiscales
    output_cs = CoordinateSystem(
        name="output",
        axes=[
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
    )
    input_cs = multiscales.metadata.intrinsic_coordinate_system
    transform.input = CoordinateSystemIdentifier(name=input_cs.name)
    transform.output = CoordinateSystemIdentifier(name=output_cs.name)
    transform.name = f"additional_{transform.type}"

    multiscales.metadata.coordinateSystems.append(output_cs)
    multiscales.metadata.coordinateTransformations = [transform]

    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ngff_zarr(tmpdir, multiscales, version="0.6")

        imported = nz.from_ngff_zarr(tmpdir)
        imported_transforms = imported.metadata.coordinateTransformations
        assert len(imported_transforms) == 1
        assert len(imported.metadata.coordinateSystems) == 2

        assert imported_transforms[0].type == transform.type
        assert imported_transforms[0].name == transform.name
        assert imported_transforms[0].input == CoordinateSystemIdentifier(
            name=input_cs.name
        )
        assert imported_transforms[0].output == CoordinateSystemIdentifier(
            name=output_cs.name
        )


@requires_zarr_v3
def test_affine_image_single_store_roundtrip():
    """An image and its affine transform live in one store; values round-trip.

    The affine parameters are inline metadata, so a single ``to_ome_zarr``
    call writes both the image pixel data and the transformation.
    """
    array = rng.random(size=(8, 8, 8), dtype=np.float32)
    input_image = nz.to_ngff_image(
        array,
        dims=["z", "y", "x"],
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )
    multiscales = nz.to_multiscales(input_image, scale_factors=[])

    output_cs = CoordinateSystem(
        name="output",
        axes=[
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
    )
    transform = affine_transform()
    input_cs = multiscales.metadata.intrinsic_coordinate_system
    transform.input = CoordinateSystemIdentifier(name=input_cs.name)
    transform.output = CoordinateSystemIdentifier(name=output_cs.name)
    transform.name = "to_output"

    multiscales.metadata.coordinateSystems.append(output_cs)
    multiscales.metadata.coordinateTransformations = [transform]

    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ome_zarr(tmpdir, multiscales, version="0.6")

        imported = nz.from_ome_zarr(tmpdir)
        np.testing.assert_array_equal(np.asarray(imported.images[0].data), array)

        imported_transforms = imported.metadata.coordinateTransformations
        assert len(imported_transforms) == 1
        assert imported_transforms[0].type == "affine"
        assert imported_transforms[0].affine == affine_transform().affine


@requires_zarr_v3
def test_new_transform_payloads_roundtrip():
    """mapAxis, byDimension and bijection survive the round-trip by value."""
    array = rng.random(size=(8, 8, 8), dtype=np.float32)
    input_image = nz.to_ngff_image(
        array,
        dims=["z", "y", "x"],
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )
    multiscales = nz.to_multiscales(input_image, scale_factors=[])
    intrinsic = multiscales.metadata.intrinsic_coordinate_system

    transforms = [
        map_axis_transform(),
        by_dimension_transform(),
        bijection_transform(),
    ]
    for transform in transforms:
        transform.input = CoordinateSystemIdentifier(name=intrinsic.name)
        transform.output = CoordinateSystemIdentifier(name=intrinsic.name)
    multiscales.metadata.coordinateTransformations = transforms

    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ngff_zarr(tmpdir, multiscales, version="0.6")
        imported = nz.from_ngff_zarr(tmpdir)

    imported_map, imported_by_dim, imported_bijection = (
        imported.metadata.coordinateTransformations
    )
    assert imported_map.mapAxis == [2, 0, 1]

    assert len(imported_by_dim.transformations) == 2
    first, second = imported_by_dim.transformations
    assert first.transformation.scale == [2.0, 3.0]
    assert first.input_axes == [0, 1]
    assert first.output_axes == [0, 1]
    assert second.transformation.translation == [5.0]
    assert second.input_axes == [2]
    assert second.output_axes == [2]

    assert imported_bijection.forward.path == "forward_field"
    assert imported_bijection.inverse.path == "inverse_field"


def _three_axis_system(name: str = "system") -> CoordinateSystem:
    return CoordinateSystem(
        name=name,
        axes=[
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
    )


def test_map_axis_must_be_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        validate_transform(MapAxis(mapAxis=[2, 0, 2]))
    with pytest.raises(ValueError, match="permutation"):
        validate_transform(MapAxis(mapAxis=[0, 1, 3]))


def test_map_axis_length_must_match_coordinate_system():
    transform = MapAxis(
        mapAxis=[1, 0],
        input=CoordinateSystemIdentifier(name="system"),
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_transform(transform, [_three_axis_system()])


def test_by_dimension_rejects_duplicate_output_axes():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                input_axes=[0, 1],
                output_axes=[0, 1],
            ),
            ByDimensionItem(
                transformation=Translation(translation=[5.0]),
                input_axes=[2],
                output_axes=[1],
            ),
        ]
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_transform(transform)


def test_by_dimension_rejects_item_dimension_mismatch():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                input_axes=[0],
                output_axes=[0],
            ),
        ]
    )
    with pytest.raises(ValueError, match="dimensional"):
        validate_transform(transform)


def test_by_dimension_must_cover_every_output_axis():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                input_axes=[0, 1],
                output_axes=[0, 1],
            ),
        ],
        output=CoordinateSystemIdentifier(name="system"),
    )
    with pytest.raises(ValueError, match="every output axis"):
        validate_transform(transform, [_three_axis_system()])


def test_bijection_dimensions_must_match():
    transform = Bijection(
        forward=Displacements(path="forward_field"),
        inverse=Displacements(path="inverse_field"),
        input=CoordinateSystemIdentifier(name="two"),
        output=CoordinateSystemIdentifier(name="three"),
    )
    two_axis = CoordinateSystem(
        name="two",
        axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
    )
    with pytest.raises(ValueError, match="dimensionality"):
        validate_transform(transform, [two_axis, _three_axis_system("three")])


def test_invalid_map_axis_is_rejected_at_parse():
    """The reader enforces the constraints on untrusted on-disk metadata."""
    from ngff_zarr.v06.zarr_metadata import Metadata

    with pytest.raises(ValueError, match="permutation"):
        Metadata._parse_transforms([{"type": "mapAxis", "mapAxis": [0, 0, 1]}], [])


def test_axis_indices_must_be_integers():
    with pytest.raises(ValueError, match="integers"):
        validate_transform(MapAxis(mapAxis=[0.0, 1.0, 2.0]))
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                input_axes=[0.0, 1.0],
                output_axes=[0, 1],
            ),
        ]
    )
    with pytest.raises(ValueError, match="integers"):
        validate_transform(transform)


def test_by_dimension_rejects_out_of_range_input_axes():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0, 4.0]),
                input_axes=[0, 1, 3],
                output_axes=[0, 1, 2],
            ),
        ],
        input=CoordinateSystemIdentifier(name="system"),
        output=CoordinateSystemIdentifier(name="system"),
    )
    with pytest.raises(ValueError, match="exceed"):
        validate_transform(transform, [_three_axis_system()])
