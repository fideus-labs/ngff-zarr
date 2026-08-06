# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for converting ITK transforms back to RFC-5 transformations.

The strongest check available here is the round trip: converting an RFC-5
transformation to ITK and back must preserve the mapping exactly. Anything
wrong with the axis reversal or the center-of-rotation algebra breaks it.
"""

import numpy as np
import pytest
from ngff_zarr import (
    itk_transform_to_ngff_matrix,
    itk_transform_to_ngff_transform,
    ngff_transform_to_itk_matrix,
    ngff_transform_to_itk_transform,
)
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Identity,
    Scale,
    TransformSequence,
    Translation,
)

ROUND_TRIP_CASES = [
    ("identity", Identity(), ("y", "x")),
    ("translation", Translation(translation=[5.0, 10.0]), ("y", "x")),
    ("scale", Scale(scale=[2.0, 3.0, 4.0]), ("z", "y", "x")),
    ("affine_2d", Affine(affine=[[0.8, -0.6, 10.0], [0.6, 0.8, -4.0]]), ("y", "x")),
    (
        "sheared_affine_3d",
        Affine(
            affine=[
                [1.0, 0.2, 0.0, 4.0],
                [0.0, 2.0, 0.3, -6.0],
                [0.5, 0.0, 1.0, 11.0],
            ]
        ),
        ("z", "y", "x"),
    ),
    (
        "sequence",
        TransformSequence(
            transformations=[
                Translation(translation=[0.0, 10.0]),
                Scale(scale=[1.0, 2.0]),
            ]
        ),
        ("y", "x"),
    ),
    (
        "affine_with_non_spatial_axes",
        Affine(
            affine=[
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.1, 3.0],
                [0.0, 0.0, 0.2, 1.1, -2.0],
            ]
        ),
        ("t", "c", "y", "x"),
    ),
]


@pytest.mark.parametrize(
    "transform, dims",
    [(case[1], case[2]) for case in ROUND_TRIP_CASES],
    ids=[case[0] for case in ROUND_TRIP_CASES],
)
def test_round_trip_preserves_the_mapping(transform, dims):
    """RFC-5 -> ITK -> RFC-5 must describe the same function."""
    expected_matrix, expected_offset = ngff_transform_to_itk_matrix(transform, dims)

    converted = itk_transform_to_ngff_transform(
        ngff_transform_to_itk_transform(transform, dims), dims
    )

    matrix, offset = ngff_transform_to_itk_matrix(converted, dims)
    assert np.allclose(matrix, expected_matrix)
    assert np.allclose(offset, expected_offset)


@pytest.mark.parametrize(
    "transform, expected_type",
    [
        (Identity(), Identity),
        (Translation(translation=[5.0, 10.0]), Translation),
        (Scale(scale=[2.0, 3.0]), Scale),
        (
            TransformSequence(
                transformations=[
                    Scale(scale=[2.0, 3.0]),
                    Translation(translation=[1.0, 2.0]),
                ]
            ),
            TransformSequence,
        ),
        (Affine(affine=[[0.8, -0.6, 0.0], [0.6, 0.8, 0.0]]), Affine),
    ],
)
def test_simplify_returns_the_least_expressive_form(transform, expected_type):
    """RFC-5 recommends the simplest form, and datasets only accept those."""
    dims = ("y", "x")
    converted = itk_transform_to_ngff_transform(
        ngff_transform_to_itk_transform(transform, dims), dims
    )
    assert isinstance(converted, expected_type)


def test_simplify_can_be_disabled():
    dims = ("y", "x")
    converted = itk_transform_to_ngff_transform(
        ngff_transform_to_itk_transform(Identity(), dims), dims, simplify=False
    )
    assert isinstance(converted, Affine)
    assert converted.affine == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_itk_transform_with_angle_parameters_is_converted():
    """Euler stores an angle, not a matrix, so decoding parameters would fail.

    The conversion probes the mapping instead, which is independent of how the
    transform packs its parameters.
    """
    itk = pytest.importorskip("itk")

    euler = itk.Euler2DTransform[itk.D].New()
    euler.SetAngle(0.3)
    euler.SetTranslation([5.0, -2.0])

    matrix, offset = itk_transform_to_ngff_matrix(euler, ("y", "x"))

    cosine, sine = np.cos(0.3), np.sin(0.3)
    reversal = np.eye(2)[::-1]
    expected = reversal @ np.array([[cosine, -sine], [sine, cosine]]) @ reversal
    assert np.allclose(matrix, expected)
    # ITK order (x, y) reverses to NGFF order (y, x).
    assert np.allclose(offset, [-2.0, 5.0])


def test_itk_composite_transform_agrees_with_transform_point():
    """The Elastix path: a composite must convert to the same mapping."""
    itk = pytest.importorskip("itk")

    translation = itk.TranslationTransform[itk.D, 2].New()
    translation.SetOffset([10.0, 0.0])
    scaling = itk.AffineTransform[itk.D, 2].New()
    scaling.SetMatrix(itk.matrix_from_array(np.diag([2.0, 2.0])))
    scaling.SetTranslation([0.0, 0.0])
    scaling.SetCenter([0.0, 0.0])
    composite = itk.CompositeTransform[itk.D, 2].New()
    composite.AddTransform(translation)
    composite.AddTransform(scaling)

    matrix, offset = itk_transform_to_ngff_matrix(composite, ("y", "x"))

    rng = np.random.default_rng(12345)
    for _ in range(5):
        point = rng.normal(size=2)  # (y, x)
        expected = np.asarray(composite.TransformPoint(point[::-1].tolist()))[::-1]
        assert np.allclose(matrix @ point + offset, expected)


def test_itk_center_of_rotation_is_folded_into_the_offset():
    """An RFC-5 affine has no center, so ITK's must be absorbed."""
    itk = pytest.importorskip("itk")

    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    center = np.array([9.5, 4.5])
    translation = np.array([10.0, 10.0])
    affine = itk.AffineTransform[itk.D, 2].New()
    affine.SetMatrix(itk.matrix_from_array(rotation))
    affine.SetTranslation(translation.tolist())
    affine.SetCenter(center.tolist())

    _, offset = itk_transform_to_ngff_matrix(affine, ("y", "x"))

    expected = translation + center - rotation @ center
    assert np.allclose(offset, expected[::-1])


def test_non_linear_itk_transform_is_rejected():
    itk = pytest.importorskip("itk")

    bspline = itk.BSplineTransform[itk.D, 2, 3].New()
    with pytest.raises(NotImplementedError, match="not linear"):
        itk_transform_to_ngff_transform(bspline, ("y", "x"))


def _itkwasm_affine(matrix_row_major, translation, center, parameterization=None):
    """An ITK-Wasm affine, by default built with the enum member."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    dimension = len(translation)
    transform_type = TransformType(
        transformParameterization=(
            TransformParameterizations.Affine
            if parameterization is None
            else parameterization
        ),
        parametersValueType=FloatTypes.Float64,
        inputDimension=dimension,
        outputDimension=dimension,
    )
    parameters = np.asarray(
        list(matrix_row_major) + list(translation), dtype=np.float64
    )
    return Transform(
        transformType=transform_type,
        numberOfFixedParameters=len(center),
        numberOfParameters=len(parameters),
        fixedParameters=np.asarray(center, dtype=np.float64),
        parameters=parameters,
    )


def test_parameterization_is_read_by_value_not_by_str():
    """``str()`` on an itkwasm parameterization is not its name.

    ``TransformParameterizations`` mixes in ``str``, so since Python 3.11
    ``str(member)`` gives ``"TransformParameterizations.Affine"``. Matching on
    that would silently miss every enum-built transform.
    """
    from itkwasm import TransformParameterizations
    from ngff_zarr.itk_transform_to_ngff_transform import _parameterization_name

    member = TransformParameterizations.Affine
    assert str(member) != "Affine"
    assert (
        _parameterization_name(type("T", (), {"transformParameterization": member}))
        == "Affine"
    )
    # A plain string must work identically.
    assert (
        _parameterization_name(type("T", (), {"transformParameterization": "Affine"}))
        == "Affine"
    )


def test_affine_decoding_is_exact_for_an_enum_built_transform():
    """Decoding must read the parameters, not re-derive them.

    Falling back to probing the mapping would land within a few ULP rather than
    reproducing the input exactly, so this asserts bit equality.
    """
    matrix = [0.90, 0.12, -0.30, -0.22, 1.05, 0.17, 0.08, -0.41, 0.95]
    translation = [3.5, -1.25, 2.0]
    transform = _itkwasm_affine(matrix, translation, [0.0, 0.0, 0.0])

    got_matrix, got_offset = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))

    reversal = np.eye(3)[::-1]
    expected_matrix = reversal @ np.asarray(matrix).reshape(3, 3) @ reversal
    expected_offset = reversal @ np.asarray(translation)
    assert np.array_equal(got_matrix, expected_matrix)
    assert np.array_equal(got_offset, expected_offset)


def test_matrix_parameterizations_decode_without_itk(monkeypatch):
    """The matrix-carrying types must not need ``itk`` to decode.

    They were reaching the ``itk`` fallback, which both required an optional
    dependency and reported a misleading reason for an affine.
    """
    import builtins

    real_import = builtins.__import__

    def no_itk(name, *args, **kwargs):
        if name == "itk":
            raise ImportError("itk is unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_itk)

    transform = _itkwasm_affine(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]
    )
    _, offset = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))
    assert np.array_equal(offset, [3.0, 2.0, 1.0])


def test_this_ports_own_output_decodes_without_itk(monkeypatch):
    """``ngff_transform_to_itk_transform`` emits the enum, so the pair must agree."""
    import builtins

    real_import = builtins.__import__

    def no_itk(name, *args, **kwargs):
        if name == "itk":
            raise ImportError("itk is unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_itk)

    dims = ("y", "x")
    emitted = ngff_transform_to_itk_transform(
        Affine(affine=[[0.8, -0.6, 10.0], [0.6, 0.8, -4.0]]), dims
    )
    converted = itk_transform_to_ngff_transform(emitted, dims)
    assert isinstance(converted, Affine)
    assert np.allclose(converted.affine, [[0.8, -0.6, 10.0], [0.6, 0.8, -4.0]])


def test_itkwasm_transform_list_composes_last_entry_first():
    """An ITK transform list applies its last entry first."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    def entry(parameterization, parameters, fixed=()):
        transform_type = TransformType(
            transformParameterization=parameterization,
            parametersValueType=FloatTypes.Float64,
            inputDimension=2,
            outputDimension=2,
        )
        return Transform(
            transformType=transform_type,
            numberOfFixedParameters=len(fixed),
            numberOfParameters=len(parameters),
            fixedParameters=np.asarray(fixed, dtype=np.float64),
            parameters=np.asarray(parameters, dtype=np.float64),
        )

    # ITK order (x, y): translate x by 10, and scale by 2.
    shift = entry(TransformParameterizations.Translation, [10.0, 0.0])
    scaling = entry(
        TransformParameterizations.Affine, [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0]
    )

    # [shift, scaling] means shift(scaling(p)) = 2p + 10.
    matrix, offset = itk_transform_to_ngff_matrix([shift, scaling], ("y", "x"))
    assert np.allclose(matrix, np.diag([2.0, 2.0]))
    assert np.allclose(offset, [0.0, 10.0])  # reversed to (y, x)

    # The other order means scaling(shift(p)) = 2(p + 10) = 2p + 20.
    _, reversed_offset = itk_transform_to_ngff_matrix([scaling, shift], ("y", "x"))
    assert np.allclose(reversed_offset, [0.0, 20.0])


def test_registration_result_can_be_attached_to_multiscales():
    """The point of this direction: persist a registration into the store."""
    itk = pytest.importorskip("itk")
    import dask.array as da
    import ngff_zarr as nz

    image = nz.to_ngff_image(da.zeros((32, 32), dtype=np.uint8), dims=["y", "x"])
    multiscales = nz.to_multiscales(image, scale_factors=[2], chunks=16)

    euler = itk.Euler2DTransform[itk.D].New()
    euler.SetAngle(0.2)
    euler.SetTranslation([3.0, -1.0])

    transform = itk_transform_to_ngff_transform(euler, ["y", "x"])

    assert isinstance(transform, Affine)
    # An RFC-5 affine is M rows of N+1 columns, translation last.
    assert len(transform.affine) == 2
    assert all(len(row) == 3 for row in transform.affine)
    assert transform.type == "affine"
    # It is shaped to go straight onto the multiscales metadata.
    assert multiscales.metadata.coordinateTransformations is None
