# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-5 transformations that are not an affine, converted to ITK.

``mapAxis``, ``byDimension`` and ``bijection`` describe a linear mapping the
way the RFC lets a writer describe it -- as a permutation, as one
transformation per group of axes, or alongside its inverse -- rather than as a
matrix. Each is folded into the single affine ITK gets, so the anchor here is
``itk``'s own ``TransformPoint``: the rebuilt transform must map a point the
way the transformation says it does, evaluated by an oracle that shares no
code with the conversion.
"""

from dataclasses import asdict

import numpy as np
import pytest
from ngff_zarr import ngff_transform_to_itk_transform
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Bijection,
    ByDimension,
    ByDimensionItem,
    Identity,
    MapAxis,
    Rotation,
    Scale,
    TransformSequence,
    Translation,
)

itk = pytest.importorskip("itk")

_ITK_ORDER = ("x", "y", "z")


def _apply(transform, point):
    """Evaluate an RFC-5 transformation, independently of the module.

    Shares no code with the conversion under test: every convention it
    encodes -- the direction a ``mapAxis`` permutation runs in, the order a
    ``sequence`` applies its entries in, which side of a ``bijection`` is the
    mapping -- is read straight from the RFC rather than from the matrix
    builder.
    """
    point = np.asarray(point, dtype=float)
    if isinstance(transform, Identity):
        return point
    if isinstance(transform, Scale):
        return point * np.asarray(transform.scale, dtype=float)
    if isinstance(transform, Translation):
        return point + np.asarray(transform.translation, dtype=float)
    if isinstance(transform, Rotation):
        return np.asarray(transform.rotation, dtype=float) @ point
    if isinstance(transform, Affine):
        affine = np.asarray(transform.affine, dtype=float)
        return affine[:, :-1] @ point + affine[:, -1]
    if isinstance(transform, MapAxis):
        # The value at position i names the input axis that becomes output i.
        return np.array([point[axis] for axis in transform.mapAxis])
    if isinstance(transform, ByDimension):
        result = np.zeros(point.shape)
        for item in transform.transformations:
            sub = _apply(item.transformation, point[list(item.input_axes)])
            for position, axis in enumerate(item.output_axes):
                result[axis] = sub[position]
        return result
    if isinstance(transform, Bijection):
        return _apply(transform.forward, point)
    if isinstance(transform, TransformSequence):
        for sub_transform in transform.transformations:
            point = _apply(sub_transform, point)
        return point
    raise AssertionError(f"the oracle does not handle {type(transform).__name__}")


def _itk_order(values, dims):
    """``values``, given in ``dims`` order, reordered fastest-axis-first."""
    values = np.asarray(values, dtype=float)
    return np.array([values[dims.index(dim)] for dim in _ITK_ORDER if dim in dims])


def _native(itkwasm_list):
    """The native ITK transform an ITK-Wasm list rebuilds to."""
    assert len(itkwasm_list) == 1
    rebuilt = itk.transform_from_dict(asdict(itkwasm_list[0]))
    if hasattr(rebuilt, "GetNthTransform"):
        rebuilt = rebuilt.GetNthTransform(0)
    return rebuilt


def _sample_points(ndim):
    """Points that pin every column of the matrix and the offset."""
    rng = np.random.default_rng(20260825)
    return [np.zeros(ndim), *rng.normal(scale=10.0, size=(5, ndim))]


def _assert_maps_like_the_oracle(transform, dims):
    rebuilt = _native(ngff_transform_to_itk_transform(transform, dims))
    for point in _sample_points(len(dims)):
        mapped = rebuilt.TransformPoint(list(_itk_order(point, dims)))
        np.testing.assert_allclose(
            np.array(mapped), _itk_order(_apply(transform, point), dims), atol=1e-9
        )


MAPPING_CASES = [
    ("mapAxis_reversal_3d", MapAxis(mapAxis=[2, 1, 0]), ("z", "y", "x")),
    ("mapAxis_swap_2d", MapAxis(mapAxis=[1, 0]), ("y", "x")),
    (
        "mapAxis_non_canonical_dims",
        MapAxis(mapAxis=[1, 2, 0]),
        ("x", "z", "y"),
    ),
    (
        "by_dimension_scale_and_translation",
        ByDimension(
            transformations=[
                ByDimensionItem(
                    transformation=Translation(translation=[7.0]),
                    input_axes=[0],
                    output_axes=[0],
                ),
                ByDimensionItem(
                    transformation=Scale(scale=[2.0, 3.0]),
                    input_axes=[1, 2],
                    output_axes=[1, 2],
                ),
            ]
        ),
        ("z", "y", "x"),
    ),
    (
        "by_dimension_that_permutes",
        ByDimension(
            transformations=[
                ByDimensionItem(
                    transformation=Scale(scale=[4.0]),
                    input_axes=[0],
                    output_axes=[2],
                ),
                ByDimensionItem(
                    transformation=Affine(affine=[[1.0, 0.5, 3.0], [0.0, 2.0, -1.0]]),
                    input_axes=[1, 2],
                    output_axes=[0, 1],
                ),
            ]
        ),
        ("z", "y", "x"),
    ),
    (
        "by_dimension_holding_a_map_axis",
        ByDimension(
            transformations=[
                ByDimensionItem(
                    transformation=MapAxis(mapAxis=[1, 0]),
                    input_axes=[0, 1],
                    output_axes=[0, 1],
                ),
                ByDimensionItem(
                    transformation=Translation(translation=[-5.0]),
                    input_axes=[2],
                    output_axes=[2],
                ),
            ]
        ),
        ("z", "y", "x"),
    ),
    (
        "bijection_uses_its_forward_direction",
        Bijection(
            forward=Scale(scale=[2.0, 4.0]),
            inverse=Scale(scale=[0.5, 0.25]),
        ),
        ("y", "x"),
    ),
    (
        "sequence_of_the_new_types",
        TransformSequence(
            transformations=[
                MapAxis(mapAxis=[1, 0]),
                Translation(translation=[10.0, -20.0]),
                Bijection(
                    forward=Affine(affine=[[0.8, -0.6, 1.0], [0.6, 0.8, 2.0]]),
                    inverse=Identity(),
                ),
            ]
        ),
        ("y", "x"),
    ),
]


@pytest.mark.parametrize(
    "transform, dims",
    [(case[1], case[2]) for case in MAPPING_CASES],
    ids=[case[0] for case in MAPPING_CASES],
)
def test_conversion_matches_transform_point(transform, dims):
    _assert_maps_like_the_oracle(transform, dims)


def test_map_axis_is_not_its_own_inverse_when_it_is_a_rotation():
    # [1, 2, 0] and [2, 0, 1] are inverse permutations, so a conversion that
    # reads the transpose vector backwards passes every symmetric case and
    # fails this one.
    dims = ("z", "y", "x")
    rebuilt = _native(ngff_transform_to_itk_transform(MapAxis(mapAxis=[1, 2, 0]), dims))
    point = np.array([1.0, 2.0, 3.0])
    mapped = np.array(rebuilt.TransformPoint(list(_itk_order(point, dims))))
    np.testing.assert_allclose(mapped, _itk_order([2.0, 3.0, 1.0], dims))


def test_bijection_ignores_an_inverse_that_disagrees():
    # RFC-5 does not require the two directions to be consistent, and the
    # forward one is the mapping. A conversion that read `inverse` instead
    # would come back with a scale of 9.
    dims = ("y", "x")
    transform = Bijection(
        forward=Scale(scale=[2.0, 2.0]), inverse=Scale(scale=[9.0, 9.0])
    )
    rebuilt = _native(ngff_transform_to_itk_transform(transform, dims))
    np.testing.assert_allclose(np.array(rebuilt.TransformPoint([1.0, 1.0])), [2.0, 2.0])


def test_map_axis_length_must_match_the_coordinate_system():
    with pytest.raises(ValueError, match="permutes 2 axes"):
        ngff_transform_to_itk_transform(MapAxis(mapAxis=[1, 0]), ("z", "y", "x"))


def test_map_axis_that_couples_a_channel_axis_is_refused():
    # Swapping c and x is a permutation ITK has no room for: dropping it
    # would leave the image in the wrong place rather than merely untimed.
    with pytest.raises(ValueError, match="couples spatial and non-spatial"):
        ngff_transform_to_itk_transform(MapAxis(mapAxis=[2, 1, 0]), ("c", "y", "x"))


def test_by_dimension_may_leave_a_non_spatial_axis_to_itself():
    # A time scale is projected away, exactly as it is for an affine: it is
    # not part of the spatial mapping an ITK transform describes.
    dims = ("t", "y", "x")
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[0.25]), input_axes=[0], output_axes=[0]
            ),
            ByDimensionItem(
                transformation=Translation(translation=[3.0, -4.0]),
                input_axes=[1, 2],
                output_axes=[1, 2],
            ),
        ]
    )
    rebuilt = _native(ngff_transform_to_itk_transform(transform, dims))
    np.testing.assert_allclose(
        np.array(rebuilt.TransformPoint([1.0, 2.0])), [1.0 - 4.0, 2.0 + 3.0]
    )


def test_by_dimension_that_leaves_an_output_axis_unset_is_refused():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0]), input_axes=[0], output_axes=[0]
            )
        ]
    )
    with pytest.raises(ValueError, match="leaving \\[1, 2\\]"):
        ngff_transform_to_itk_transform(transform, ("z", "y", "x"))


def test_by_dimension_item_of_unequal_arity_is_refused():
    item = ByDimensionItem(
        transformation=Affine(affine=[[1.0, 0.0, 2.0, 0.0]]),
        input_axes=[0, 1, 2],
        output_axes=[0],
    )
    transform = ByDimension(
        transformations=[
            item,
            ByDimensionItem(
                transformation=Identity(), input_axes=[1, 2], output_axes=[1, 2]
            ),
        ]
    )
    with pytest.raises(ValueError, match="only a square mapping"):
        ngff_transform_to_itk_transform(transform, ("z", "y", "x"))


def test_by_dimension_axis_index_beyond_the_coordinate_system_is_refused():
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 2.0]),
                input_axes=[0, 5],
                output_axes=[0, 1],
            )
        ]
    )
    with pytest.raises(ValueError, match="exceed the 2 axes"):
        ngff_transform_to_itk_transform(transform, ("y", "x"))


def test_by_dimension_matches_the_affine_it_is_equivalent_to():
    dims = ("z", "y", "x")
    by_dimension = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Translation(translation=[7.0]),
                input_axes=[0],
                output_axes=[0],
            ),
            ByDimensionItem(
                transformation=Scale(scale=[2.0, 3.0]),
                input_axes=[1, 2],
                output_axes=[1, 2],
            ),
        ]
    )
    affine = Affine(
        affine=[
            [1.0, 0.0, 0.0, 7.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    (from_by_dimension,) = ngff_transform_to_itk_transform(by_dimension, dims)
    (from_affine,) = ngff_transform_to_itk_transform(affine, dims)
    np.testing.assert_allclose(from_by_dimension.parameters, from_affine.parameters)


def test_an_unsupported_type_names_the_ones_that_convert():
    class Unsupported:
        type = "someFutureType"

    with pytest.raises(NotImplementedError, match="mapAxis, byDimension, bijection"):
        ngff_transform_to_itk_transform(Unsupported(), ("y", "x"))
