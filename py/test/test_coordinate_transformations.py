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
    ProjectAxis,
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
                inputAxes=[0, 1],
                outputAxes=[0, 1],
            ),
            ByDimensionItem(
                transformation=Translation(translation=[5.0]),
                inputAxes=[2],
                outputAxes=[2],
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
    assert first.inputAxes == [0, 1]
    assert first.outputAxes == [0, 1]
    assert second.transformation.translation == [5.0]
    assert second.inputAxes == [2]
    assert second.outputAxes == [2]

    assert imported_bijection.forward.path == "forward_field"
    assert imported_bijection.inverse.path == "inverse_field"


@requires_zarr_v3
def test_project_axis_roundtrips():
    """A projectAxis transform survives a 0.6 write and read by value.

    Before ``projectAxis`` was modelled, a document using it passed the
    bundled schema and the reader raised ``Unsupported transform type``.
    """
    array = rng.random(size=(8, 8, 8), dtype=np.float32)
    input_image = nz.to_ngff_image(
        array,
        dims=["z", "y", "x"],
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
    )
    multiscales = nz.to_multiscales(input_image, scale_factors=[])
    intrinsic = multiscales.metadata.intrinsic_coordinate_system

    plane = CoordinateSystem(
        name="plane",
        axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
    )
    multiscales.metadata.coordinateSystems.append(plane)
    multiscales.metadata.coordinateTransformations = [
        ProjectAxis(
            droppedInputs=[0],
            input=CoordinateSystemIdentifier(name=intrinsic.name),
            output=CoordinateSystemIdentifier(name="plane"),
            name="drop_z",
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ome_zarr(tmpdir, multiscales, version="0.6")
        imported = nz.from_ome_zarr(tmpdir)

    (projection,) = imported.metadata.coordinateTransformations
    assert projection.type == "projectAxis"
    assert projection.droppedInputs == [0]
    assert projection.createdOutputs is None
    assert projection.output.name == "plane"


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
    """Intrinsic constraints hold by construction; the instance never exists."""
    with pytest.raises(ValueError, match="permutation"):
        MapAxis(mapAxis=[2, 0, 2])
    with pytest.raises(ValueError, match="permutation"):
        MapAxis(mapAxis=[0, 1, 3])


def test_map_axis_arity_is_bounded_below_0_9_dev1():
    """Through 0.6 a coordinate system holds 2 to 5 axes; mapAxis matches.

    The bound is checked at validation rather than at construction: it mirrors
    the minItems and maxItems those schemas set, and ``__post_init__`` has no
    version to consult.
    """
    for indices in ([0], [], [5, 4, 3, 2, 1, 0]):
        with pytest.raises(ValueError, match="between 2 and 5"):
            MapAxis(mapAxis=indices).validate()


def test_map_axis_arity_is_unbounded_at_0_9_dev1():
    """RFC-3 lifts the five-axis cap, and the 0.9.dev1 mapAxis sets no bound.

    Its schema declares neither minItems nor maxItems, so a permutation over
    six axes is a valid 0.9.dev1 document and the model must read it.
    """
    MapAxis(mapAxis=[5, 4, 3, 2, 1, 0]).validate(version="0.9.dev1")
    MapAxis(mapAxis=[0]).validate(version="0.9.dev1")


def test_a_nested_map_axis_follows_the_document_version():
    """A wrapper carries the version to what it holds.

    Without that, a six-axis permutation inside a sequence or a bijection
    still met the v0.6 arity even in a 0.9.dev1 document.
    """
    six = MapAxis(mapAxis=[5, 4, 3, 2, 1, 0])

    TransformSequence(transformations=[six]).validate(version="0.9.dev1")
    Bijection(forward=six, inverse=six).validate(version="0.9.dev1")

    with pytest.raises(ValueError, match="between 2 and 5"):
        TransformSequence(transformations=[six]).validate(version="0.6")
    with pytest.raises(ValueError, match="between 2 and 5"):
        Bijection(forward=six, inverse=six).validate(version="0.6")


def test_map_axis_is_a_permutation_at_every_version():
    """The permutation rule holds regardless of the version."""
    for ngff_version in (None, "0.6", "0.9.dev1"):
        with pytest.raises(ValueError, match="permutation"):
            MapAxis(mapAxis=[0, 0, 1]).validate(version=ngff_version)


def test_axis_indices_must_be_integers():
    with pytest.raises(ValueError, match="integers"):
        MapAxis(mapAxis=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="integers"):
        ByDimensionItem(
            transformation=Scale(scale=[2.0, 3.0]),
            inputAxes=[0.0, 1.0],
            outputAxes=[0, 1],
        )


def test_by_dimension_item_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensional"):
        ByDimensionItem(
            transformation=Scale(scale=[2.0, 3.0]),
            inputAxes=[0],
            outputAxes=[0],
        )


def test_by_dimension_rejects_duplicate_outputAxes():
    """Duplicates within an item and across items are both rejected."""
    with pytest.raises(ValueError, match="exactly one"):
        ByDimensionItem(
            transformation=Scale(scale=[2.0, 3.0]),
            inputAxes=[0, 1],
            outputAxes=[1, 1],
        )
    with pytest.raises(ValueError, match="exactly one"):
        ByDimension(
            transformations=[
                ByDimensionItem(
                    transformation=Scale(scale=[2.0, 3.0]),
                    inputAxes=[0, 1],
                    outputAxes=[0, 1],
                ),
                ByDimensionItem(
                    transformation=Translation(translation=[5.0]),
                    inputAxes=[2],
                    outputAxes=[1],
                ),
            ]
        )


def test_map_axis_length_must_match_coordinate_system():
    """Contextual constraints need the coordinate systems: validate_transform."""
    transform = MapAxis(
        mapAxis=[1, 0],
        input=CoordinateSystemIdentifier(name="system"),
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_transform(transform, [_three_axis_system()])


def test_by_dimension_must_cover_every_output_axis():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                inputAxes=[0, 1],
                outputAxes=[0, 1],
            ),
        ],
        output=CoordinateSystemIdentifier(name="system"),
    )
    with pytest.raises(ValueError, match="every output axis"):
        validate_transform(transform, [_three_axis_system()])


def test_by_dimension_rejects_out_of_range_inputAxes():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0, 4.0]),
                inputAxes=[0, 1, 3],
                outputAxes=[0, 1, 2],
            ),
        ],
        input=CoordinateSystemIdentifier(name="system"),
        output=CoordinateSystemIdentifier(name="system"),
    )
    with pytest.raises(ValueError, match="exceed"):
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


def test_contextual_checks_are_skipped_without_coordinate_systems():
    """A valid transform whose systems do not resolve passes on its own."""
    transform = MapAxis(
        mapAxis=[1, 0],
        input=CoordinateSystemIdentifier(name="unknown"),
    )
    validate_transform(transform)
    validate_transform(transform, [_three_axis_system("other")])


def test_axis_count_resolves_named_systems():
    systems = [_three_axis_system("system")]
    assert CoordinateSystemIdentifier(name="system").axis_count(systems) == 3
    assert CoordinateSystemIdentifier(name="unknown").axis_count(systems) is None
    assert CoordinateSystemIdentifier(path="scale0").axis_count(systems) is None
    assert CoordinateSystemIdentifier(name="system").axis_count(None) is None


def test_invalid_map_axis_is_rejected_at_parse():
    """The reader enforces the constraints on untrusted on-disk metadata."""
    from ngff_zarr.v06.zarr_metadata import Metadata

    with pytest.raises(ValueError, match="permutation"):
        Metadata._parse_transforms([{"type": "mapAxis", "mapAxis": [0, 0, 1]}], [])


def test_wrapper_validation_reaches_nested_transforms():
    """Mutating a nested transform after construction is caught by validate."""
    nested = MapAxis(mapAxis=[1, 0])
    wrappers = [
        Bijection(forward=nested, inverse=Identity()),
        TransformSequence(transformations=[nested]),
        ByDimension(
            transformations=[ByDimensionItem(nested, [0, 1], [0, 1])],
        ),
    ]
    for wrapper in wrappers:
        validate_transform(wrapper)
    nested.mapAxis = [0, 0]
    for wrapper in wrappers:
        with pytest.raises(ValueError, match="permutation"):
            validate_transform(wrapper)


def _shared_cases():
    import json
    from pathlib import Path

    spec = json.loads((Path(__file__).parent / "rfc5_transform_cases.json").read_text())
    systems = [
        CoordinateSystem(
            name=system["name"],
            axes=[
                Axis(name=axis["name"], type=axis["type"]) for axis in system["axes"]
            ],
        )
        for system in spec["coordinateSystems"]
    ]
    return systems, spec["cases"]


_SHARED_SYSTEMS, _SHARED_CASES = _shared_cases()


@pytest.mark.parametrize(
    "case", _SHARED_CASES, ids=[case["name"] for case in _SHARED_CASES]
)
def test_shared_rfc5_cases_match_expected_verdict(case):
    """The Python and TypeScript readers give the same verdict on each case.

    ``rfc5_transform_cases.json`` is also exercised by the TypeScript suite,
    so a rule enforced in one port and not the other fails here.
    """
    from ngff_zarr.v06.zarr_metadata import Metadata

    if case["ok"]:
        Metadata._parse_transforms([case["transformation"]], _SHARED_SYSTEMS)
    else:
        with pytest.raises(ValueError):
            Metadata._parse_transforms([case["transformation"]], _SHARED_SYSTEMS)


def test_by_dimension_item_reads_the_snake_case_spelling():
    # ngff-zarr 0.43.0 wrote `input_axes` and `output_axes`; the spec spells
    # them `inputAxes` and `outputAxes`, which is what is written now. A store
    # from that release must still read.
    from ngff_zarr.v06.zarr_metadata import ByDimensionItem, Scale

    item = ByDimensionItem.from_dict(
        {
            "transformation": {"type": "scale", "scale": [2.0]},
            "input_axes": [0],
            "output_axes": [0],
        }
    )

    assert item.inputAxes == [0]
    assert item.outputAxes == [0]
    assert isinstance(item.transformation, Scale)
