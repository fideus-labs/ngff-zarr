# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import pytest
import tempfile
import numpy as np
import ngff_zarr as nz
from ngff_zarr.v06.zarr_metadata import Scale, Translation, Rotation, Affine, Identity, TransformSequence, CoordinateSystem, Axis

rng = np.random.default_rng(12345)

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
        ])

def affine_transform() -> Affine:
    return Affine(
        affine=[
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 15.0],
        ])

def transform_sequence() -> TransformSequence:
    return TransformSequence(
        transformations=[
            scale_transform(),
            translation_transform(),
            rotation_transform(),
            affine_transform(),
        ]
    )

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
    assert 'intrinsic' in [cs.name for cs in multiscales.metadata.coordinateSystems]

    # add output coordinate system to multiscales
    output_cs = CoordinateSystem(
        name="output",
        axes=[
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ]
    )
    input_cs = multiscales.metadata.coordinateSystems[0]
    transform.input = input_cs.name
    transform.output = output_cs.name
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
        assert imported_transforms[0].input == input_cs.name
        assert imported_transforms[0].output == output_cs.name

if __name__ == "__main__":
    pytest.main([__file__])