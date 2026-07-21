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
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Identity,
    Rotation,
    Scale,
    TransformSequence,
    Translation,
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
