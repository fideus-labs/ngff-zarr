# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the RFC-5 to ITK transform bridge and the resample bounding box.

The expected regions here are not taken from the pipeline itself. They come
either from the worked examples in the ITK-Wasm ``resample-bounding-box``
documentation, or from ``_oracle_region`` below, which recomputes the region
from first principles in NGFF axis order. That keeps the tests honest about
the two conventions that differ between RFC-5 and ITK: axis order and
sequence composition order.
"""

import itertools

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import (
    RAS,
    NgffImage,
    itk_transform_resample_bounding_box,
    ngff_image_to_itk_image,
)
from ngff_zarr.itk_transform_resample_bounding_box import (
    _itk_direction,
    _metadata_only_itk_image,
)
from ngff_zarr.ngff_transform_to_itk_transform import _ngff_transform_to_itk_matrix
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Displacements,
    Identity,
    Rotation,
    Scale,
    TransformSequence,
    Translation,
)


def _translation(offset):
    """An ITK-Wasm translation, in ITK (fastest-axis-first) order."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    dimension = len(offset)
    return [
        Transform(
            transformType=TransformType(
                transformParameterization=TransformParameterizations.Translation,
                parametersValueType=FloatTypes.Float64,
                inputDimension=dimension,
                outputDimension=dimension,
            ),
            numberOfFixedParameters=0,
            numberOfParameters=dimension,
            fixedParameters=np.empty((0,), dtype=np.float64),
            parameters=np.asarray(offset, dtype=np.float64),
        )
    ]


def _identity(dimension):
    return _translation([0.0] * dimension)


def _image(dims, shape, scale, translation, orientations=None):
    """A geometry-only NgffImage; the data is never meant to be computed."""
    return NgffImage(
        data=da.zeros(tuple(shape[d] for d in dims), dtype=np.uint8, chunks=8),
        dims=list(dims),
        scale={d: float(scale[d]) for d in dims},
        translation={d: float(translation[d]) for d in dims},
        axes_orientations=orientations,
    )


def _affine(matrix, offset):
    """An ITK-Wasm affine: row-major matrix then translation, centre at origin."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    dimension = len(offset)
    parameters = np.concatenate([np.asarray(matrix, dtype=float).ravel(), offset])
    return [
        Transform(
            transformType=TransformType(
                transformParameterization=TransformParameterizations.Affine,
                parametersValueType=FloatTypes.Float64,
                inputDimension=dimension,
                outputDimension=dimension,
            ),
            numberOfFixedParameters=dimension,
            numberOfParameters=len(parameters),
            fixedParameters=np.zeros(dimension, dtype=np.float64),
            parameters=parameters,
        )
    ]


def _oracle_region(matrix, offset, fixed, moving, spatial, padding):
    """Recompute the region in NGFF order, independently of the pipeline.

    A linear map sends the fixed rectangle to a convex region, so sampling the
    corners is exact.
    """
    shape = [fixed.data.shape[list(fixed.dims).index(d)] for d in spatial]
    fixed_scale = np.array([fixed.scale[d] for d in spatial])
    fixed_translation = np.array([fixed.translation[d] for d in spatial])
    moving_scale = np.array([moving.scale[d] for d in spatial])
    moving_translation = np.array([moving.translation[d] for d in spatial])

    index_min = np.full(len(spatial), np.inf)
    index_max = np.full(len(spatial), -np.inf)
    for corner in itertools.product(*[(0, n - 1) for n in shape]):
        point = fixed_translation + fixed_scale * np.array(corner, dtype=float)
        moved = matrix @ point + offset
        continuous_index = (moved - moving_translation) / moving_scale
        index_min = np.minimum(index_min, continuous_index)
        index_max = np.maximum(index_max, continuous_index)

    start = np.floor(index_min).astype(np.int64) - padding
    end = np.ceil(index_max).astype(np.int64) + padding
    return start, np.maximum(end - start + 1, 0)


def test_ngff_translation_matches_the_documented_worked_example():
    """Reproduces the 2D translation example from the ITK-Wasm documentation.

    That example is stated in ITK order: fixed 16x16 spacing (2,2) origin
    (10,20), moving 64x64 spacing (1,1) origin 0, translation (10,5),
    padding 1. Written as RFC-5 the axes reverse, so the translation is
    ``[5, 10]`` over ``("y", "x")``.
    """
    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 2, "x": 2}, {"y": 20, "x": 10})
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        Translation(translation=[5.0, 10.0]), fixed, moving, padding=1
    )

    assert bounding_box.start_index == {"y": 24, "x": 19}
    assert bounding_box.size == {"y": 33, "x": 33}
    assert bounding_box.corners_min == {"y": 25.0, "x": 20.0}
    assert bounding_box.corners_max == {"y": 55.0, "x": 50.0}
    assert bounding_box.padded_corners_min == {"y": 24.0, "x": 19.0}
    assert bounding_box.padded_corners_max == {"y": 56.0, "x": 51.0}


def test_padding_is_symmetric():
    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 2, "x": 2}, {"y": 20, "x": 10})
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    transform = Translation(translation=[5.0, 10.0])

    padded = itk_transform_resample_bounding_box(transform, fixed, moving, padding=1)
    tight = itk_transform_resample_bounding_box(transform, fixed, moving, padding=0)

    for dim in ("y", "x"):
        assert tight.start_index[dim] == padded.start_index[dim] + 1
        assert tight.size[dim] == padded.size[dim] - 2
    # The tight corners are padding independent.
    assert tight.corners_min == padded.corners_min
    assert tight.corners_max == padded.corners_max


def test_asymmetric_three_dimensional_ngff_affine_matches_oracle():
    """An asymmetric sheared affine makes any axis-order slip visible."""
    spatial = ("z", "y", "x")
    fixed = _image(
        spatial,
        {"z": 4, "y": 8, "x": 16},
        {"z": 3.0, "y": 2.0, "x": 1.0},
        {"z": 30.0, "y": 20.0, "x": 10.0},
    )
    moving = _image(
        spatial,
        {"z": 64, "y": 128, "x": 256},
        {"z": 1.5, "y": 0.5, "x": 0.25},
        {"z": -5.0, "y": 7.0, "x": 3.0},
    )
    matrix = np.array([[1.0, 0.2, 0.0], [0.0, 2.0, 0.3], [0.5, 0.0, 1.0]])
    offset = np.array([4.0, -6.0, 11.0])
    affine = np.hstack([matrix, offset.reshape(-1, 1)]).tolist()

    bounding_box = itk_transform_resample_bounding_box(
        Affine(affine=affine), fixed, moving, padding=2
    )

    expected_start, expected_size = _oracle_region(
        matrix, offset, fixed, moving, spatial, 2
    )
    assert [bounding_box.start_index[d] for d in spatial] == expected_start.tolist()
    assert [bounding_box.size[d] for d in spatial] == expected_size.tolist()


def test_affine_translation_is_the_last_column():
    """RFC-5 stores the translation as the last column of the affine matrix."""
    matrix, offset = _ngff_transform_to_itk_matrix(
        Affine(affine=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ("y", "x")
    )
    # In NGFF terms: y = 1*y + 2*x + 3 and x = 4*y + 5*x + 6. ITK reverses the
    # axes, so both the matrix and the offset come back reversed.
    assert np.allclose(offset, [6.0, 3.0])
    assert np.allclose(matrix, [[5.0, 4.0], [2.0, 1.0]])


def test_affine_with_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="translation is the last column"):
        _ngff_transform_to_itk_matrix(
            Affine(affine=[[1.0, 0.0], [0.0, 1.0]]), ("y", "x")
        )


def test_sequence_applies_its_first_entry_first():
    """RFC-5 composes ``[f0, f1]`` as ``f1(f0(x))``.

    ITK transform lists compose the other way round, so a sequence that is
    passed through unreversed would silently give ``2x + 10`` here instead of
    ``2(x + 10)``.
    """
    sequence = TransformSequence(
        transformations=[
            Translation(translation=[0.0, 10.0]),
            Scale(scale=[1.0, 2.0]),
        ]
    )
    matrix, offset = _ngff_transform_to_itk_matrix(sequence, ("y", "x"))

    # ITK order is (x, y): translate by 10 then scale by 2 gives an offset of 20.
    assert np.isclose(offset[0], 20.0)
    assert np.isclose(matrix[0, 0], 2.0)

    reversed_sequence = TransformSequence(
        transformations=[
            Scale(scale=[1.0, 2.0]),
            Translation(translation=[0.0, 10.0]),
        ]
    )
    _, reversed_offset = _ngff_transform_to_itk_matrix(reversed_sequence, ("y", "x"))
    assert np.isclose(reversed_offset[0], 10.0)


def test_sequence_order_survives_the_whole_pipeline():
    """The composition order has to hold at the public entry point too.

    Every other sequence test here reads the matrix out of
    ``_ngff_transform_to_itk_matrix``. If the order inverted, the region the
    caller actually gets would move, and nothing below that helper would
    notice. Applying ``[translate 10, scale 2]`` in the wrong order gives
    ``2x + 10`` instead of ``2(x + 10)``, so the x start moves by 10.
    """
    spatial = ("y", "x")
    fixed = _image(spatial, {"y": 16, "x": 16}, {"y": 2, "x": 2}, {"y": 20, "x": 10})
    moving = _image(spatial, {"y": 512, "x": 512}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    sequence = TransformSequence(
        transformations=[
            Translation(translation=[0.0, 10.0]),
            Scale(scale=[1.0, 2.0]),
        ]
    )

    bounding_box = itk_transform_resample_bounding_box(
        sequence, fixed, moving, padding=0
    )

    # First entry first: y = x, x = 2(x + 10) = 2x + 20.
    expected_start, expected_size = _oracle_region(
        np.diag([1.0, 2.0]), np.array([0.0, 20.0]), fixed, moving, spatial, 0
    )
    assert [bounding_box.start_index[d] for d in spatial] == expected_start.tolist()
    assert [bounding_box.size[d] for d in spatial] == expected_size.tolist()
    assert bounding_box.start_index["x"] == 40  # 30 if the order inverted


def test_nested_sequences_compose():
    inner = TransformSequence(
        transformations=[Scale(scale=[1.0, 2.0]), Translation(translation=[0.0, 1.0])]
    )
    outer = TransformSequence(
        transformations=[inner, Translation(translation=[0.0, 100.0])]
    )
    _, offset = _ngff_transform_to_itk_matrix(outer, ("y", "x"))
    assert np.isclose(offset[0], 101.0)


def test_identity_scale_rotation_round_trip():
    matrix, offset = _ngff_transform_to_itk_matrix(Identity(), ("y", "x"))
    assert np.allclose(matrix, np.eye(2))
    assert np.allclose(offset, np.zeros(2))

    matrix, _ = _ngff_transform_to_itk_matrix(Scale(scale=[2.0, 3.0]), ("y", "x"))
    # Reversed to ITK order (x, y).
    assert np.allclose(matrix, np.diag([3.0, 2.0]))

    rotation = [[0.0, -1.0], [1.0, 0.0]]
    matrix, _ = _ngff_transform_to_itk_matrix(Rotation(rotation=rotation), ("y", "x"))
    reversal = np.eye(2)[::-1]
    assert np.allclose(matrix, reversal @ np.array(rotation) @ reversal)


def test_non_linear_transform_is_rejected():
    with pytest.raises(NotImplementedError, match="cannot be converted"):
        _ngff_transform_to_itk_matrix(Displacements(path="field"), ("y", "x"))


def test_transform_coupling_spatial_and_non_spatial_axes_is_rejected():
    affine = np.eye(3, 4)
    affine[1, 0] = 0.5  # y would depend on c
    with pytest.raises(ValueError, match="couples spatial and non-spatial"):
        _ngff_transform_to_itk_matrix(Affine(affine=affine.tolist()), ("c", "y", "x"))


def test_rfc5_branch_ignores_anatomical_orientation():
    """An RFC-5 transformation acts on the intrinsic coordinate system,
    where a point is ``translation + scale * index`` and no direction matrix
    applies, so the region it selects cannot depend on RFC-4 anatomical
    orientation: the oriented region must equal the unoriented one.
    """
    from ngff_zarr.rfc4 import RAS

    spatial = ("z", "y", "x")
    shape = {"z": 8, "y": 16, "x": 24}
    scale = dict.fromkeys(spatial, 1.0)
    translation = dict.fromkeys(spatial, 0.0)
    transform = Translation(translation=[2.0, 3.0, 4.0])

    plain = itk_transform_resample_bounding_box(
        transform,
        _image(spatial, shape, scale, translation),
        _image(spatial, {"z": 32, "y": 64, "x": 96}, scale, translation),
        padding=0,
    )
    oriented = itk_transform_resample_bounding_box(
        transform,
        _image(spatial, shape, scale, translation, RAS),
        _image(spatial, {"z": 32, "y": 64, "x": 96}, scale, translation, RAS),
        padding=0,
    )

    assert oriented.start_index == plain.start_index == {"z": 2, "y": 3, "x": 4}
    assert oriented.size == plain.size


def test_non_canonical_spatial_order_binds_itk_axes_by_name():
    """ITK's first component is x by *name*, not whichever axis comes last.

    Reversing the dims is only equivalent for the canonical ("z", "y", "x"):
    with dims ("z", "x", "y") it would bind ITK x to y and ITK y to x,
    silently swapping the two axes' regions. Both paths must agree, and
    agree with the axis names.
    """
    dims = ("z", "x", "y")
    fixed = _image(
        dims,
        {"z": 4, "x": 8, "y": 16},
        dict.fromkeys(dims, 1.0),
        dict.fromkeys(dims, 0.0),
    )
    moving = _image(
        dims,
        {"z": 32, "x": 64, "y": 64},
        dict.fromkeys(dims, 1.0),
        dict.fromkeys(dims, 0.0),
    )
    expected = {"z": 1, "x": 7, "y": 5}

    # RFC-5 parameters are in dims order: (z, x, y).
    via_rfc5 = itk_transform_resample_bounding_box(
        Translation(translation=[1.0, 7.0, 5.0]), fixed, moving, padding=0
    )
    # ITK parameters are fastest-axis-first by name: (x, y, z).
    via_itk = itk_transform_resample_bounding_box(
        _translation([7.0, 5.0, 1.0]), fixed, moving, padding=0
    )

    assert via_rfc5.start_index == expected
    assert via_itk.start_index == expected


def test_a_component_on_a_non_spatial_axis_alone_is_projected_away():
    """ITK has no non-spatial axis, so a frame interval on ``t`` has nowhere
    to go.

    Dropping it leaves the spatial mapping exact, which is all an ITK transform
    describes. This pins that as a decision rather than an accident; the
    coupled case above is refused instead, because dropping *that* would move
    the image.
    """
    sequence = TransformSequence(
        transformations=[
            Scale(scale=[0.5, 1.0, 2.0, 2.0]),
            Translation(translation=[7.0, 0.0, 3.0, -4.0]),
        ]
    )

    matrix, offset = _ngff_transform_to_itk_matrix(sequence, ("t", "c", "y", "x"))

    # ITK order (x, y): the t and c entries are gone, the spatial ones exact.
    assert np.allclose(matrix, np.diag([2.0, 2.0]))
    assert np.allclose(offset, [-4.0, 3.0])


def test_v04_transform_dataclasses_are_accepted():
    """``ngff_zarr.Scale`` and friends are the v0.4 spellings, not the v0.6 ones."""
    import ngff_zarr as nz

    assert not isinstance(nz.Translation(translation=[0.0, 0.0]), Translation)

    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 2, "x": 2}, {"y": 20, "x": 10})
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        nz.Translation(translation=[5.0, 10.0]), fixed, moving, padding=1
    )

    assert bounding_box.start_index == {"y": 24, "x": 19}


def test_mismatched_spatial_dims_are_rejected():
    fixed = _image(
        "zyx",
        {"z": 4, "y": 4, "x": 4},
        {"z": 1, "y": 1, "x": 1},
        {"z": 0, "y": 0, "x": 0},
    )
    moving = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="they must match"):
        itk_transform_resample_bounding_box(_identity(2), fixed, moving)


def test_pixel_buffers_are_never_computed():
    """The whole point: geometry in, region out, no pixels touched."""
    computed = []

    def explode(block):
        computed.append(True)
        raise AssertionError("pixel data was materialized")

    poison = da.zeros((32, 32), dtype=np.uint8, chunks=8).map_blocks(
        explode, dtype=np.uint8
    )
    fixed = NgffImage(
        data=poison,
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )

    # dask probes the block function once while building the graph above, so
    # only what happens from here on counts.
    computed.clear()
    bounding_box = itk_transform_resample_bounding_box(
        _identity(2), fixed, fixed, padding=1
    )

    assert not computed
    assert bounding_box.start_index == {"y": -1, "x": -1}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_non_spatial_axes_are_passed_through():
    shape = {"t": 3, "c": 2, "z": 4, "y": 8, "x": 8}
    unit = dict.fromkeys(shape, 1)
    zero = dict.fromkeys(shape, 0)
    fixed = _image("tczyx", shape, unit, zero)
    moving = _image("tczyx", {"t": 3, "c": 2, "z": 16, "y": 32, "x": 32}, unit, zero)

    bounding_box = itk_transform_resample_bounding_box(
        _translation([3.0, 2.0, 1.0]), fixed, moving, padding=0
    )

    assert bounding_box.dims == ("z", "y", "x")
    assert bounding_box.start_index == {"z": 1, "y": 2, "x": 3}
    slices = bounding_box.slices(moving.dims)
    assert slices[0] == slice(None)  # t
    assert slices[1] == slice(None)  # c
    assert slices[2:] == (slice(1, 5), slice(2, 10), slice(3, 11))


def test_region_outside_the_moving_image_is_empty():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        _translation([1000.0, 1000.0]), fixed, moving, padding=1
    )

    assert bounding_box.is_empty
    assert bounding_box.crop(moving) is None


def test_negative_start_index_is_clamped_not_wrapped():
    """A negative slice bound would wrap silently instead of raising."""
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        _identity(2), fixed, moving, padding=2
    )

    assert bounding_box.start_index == {"y": -2, "x": -2}
    assert bounding_box.clamped() == {"y": (0, 6), "x": (0, 6)}
    assert bounding_box.slices() == (slice(0, 6), slice(0, 6))
    assert not bounding_box.is_empty


def test_crop_is_lazy_and_shifts_the_translation():
    fixed = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 512, "x": 1024})
    moving = _image("yx", {"y": 4096, "x": 4096}, {"y": 2, "x": 2}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        _translation([0.0, 0.0]), fixed, moving, padding=1
    )
    cropped = bounding_box.crop(moving)

    assert isinstance(cropped.data, da.Array)
    bounds = bounding_box.clamped()
    assert cropped.data.shape == tuple(
        stop - start for start, stop in (bounds["y"], bounds["x"])
    )
    for dim in ("y", "x"):
        expected = moving.translation[dim] + bounds[dim][0] * moving.scale[dim]
        assert np.isclose(cropped.translation[dim], expected)
    # Reads a small corner rather than the whole moving image.
    assert np.prod(cropped.data.shape) < 0.01 * np.prod(moving.data.shape)


def test_crop_preserves_orientation_and_scale():
    moving = _image(
        "zyx",
        {"z": 32, "y": 32, "x": 32},
        {"z": 2.0, "y": 1.0, "x": 0.5},
        {"z": 1.0, "y": 2.0, "x": 3.0},
        RAS,
    )
    fixed = _image(
        "zyx",
        {"z": 4, "y": 4, "x": 4},
        {"z": 2.0, "y": 1.0, "x": 0.5},
        {"z": 1.0, "y": 2.0, "x": 3.0},
        RAS,
    )
    bounding_box = itk_transform_resample_bounding_box(_identity(3), fixed, moving)
    cropped = bounding_box.crop(moving)

    assert cropped.scale == moving.scale
    assert cropped.axes_orientations == RAS
    assert list(cropped.dims) == list(moving.dims)


@pytest.mark.parametrize("orientations", [None, RAS])
def test_metadata_only_geometry_matches_ngff_image_to_itk_image(orientations):
    """The ITK path must land in the same physical space Elastix registered in."""
    image = _image(
        "zyx",
        {"z": 4, "y": 8, "x": 16},
        {"z": 3.0, "y": 2.0, "x": 1.0},
        {"z": 30.0, "y": 20.0, "x": 10.0},
        orientations,
    )
    reference = ngff_image_to_itk_image(image, wasm=True)

    itk_dims = ["x", "y", "z"]
    geometry = _metadata_only_itk_image(
        image, itk_dims, _itk_direction(image, itk_dims)
    )

    assert list(geometry.origin) == list(reference.origin)
    assert list(geometry.spacing) == list(reference.spacing)
    assert list(geometry.size) == list(reference.size)
    assert np.allclose(np.asarray(geometry.direction), np.asarray(reference.direction))
    assert geometry.data.size == 0


def test_ras_orientation_yields_a_non_identity_direction():
    image = _image(
        "zyx",
        {"z": 4, "y": 8, "x": 16},
        {"z": 1, "y": 1, "x": 1},
        {"z": 0, "y": 0, "x": 0},
        RAS,
    )
    direction = _itk_direction(image, ["x", "y", "z"])
    assert np.allclose(direction, np.diag([-1.0, -1.0, 1.0]))


def test_itk_translation_transform_is_accepted():
    itk = pytest.importorskip("itk")

    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 2, "x": 2}, {"y": 20, "x": 10})
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    transform = itk.TranslationTransform[itk.D, 2].New()
    transform.SetOffset([10.0, 5.0])  # ITK order (x, y)

    bounding_box = itk_transform_resample_bounding_box(
        transform, fixed, moving, padding=1
    )

    assert bounding_box.start_index == {"y": 24, "x": 19}
    assert bounding_box.size == {"y": 33, "x": 33}


def test_itk_composite_transform_is_accepted():
    """An ITK composite applies its last-added transform first."""
    itk = pytest.importorskip("itk")

    translation = itk.TranslationTransform[itk.D, 2].New()
    translation.SetOffset([10.0, 0.0])
    scaling = itk.AffineTransform[itk.D, 2].New()
    scaling.SetMatrix(itk.matrix_from_array(np.array([[2.0, 0.0], [0.0, 2.0]])))
    scaling.SetTranslation([0.0, 0.0])
    scaling.SetCenter([0.0, 0.0])
    composite = itk.CompositeTransform[itk.D, 2].New()
    composite.AddTransform(translation)
    composite.AddTransform(scaling)

    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 2, "x": 2}, {"y": 20, "x": 10})
    moving = _image("yx", {"y": 512, "x": 512}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        composite, fixed, moving, padding=0
    )

    # Cross-checked against the composite's own point mapping.
    low = composite.TransformPoint([10.0, 20.0])
    high = composite.TransformPoint([40.0, 50.0])
    assert np.isclose(bounding_box.corners_min["x"], low[0])
    assert np.isclose(bounding_box.corners_min["y"], low[1])
    assert np.isclose(bounding_box.corners_max["x"], high[0])
    assert np.isclose(bounding_box.corners_max["y"], high[1])


def test_index_range_overflow_is_reported_not_silently_empty():
    """A grid too large for the pipeline's index space must not read as empty.

    The region is computed in 32-bit index space while the corners come back as
    doubles, so a fixed/moving scale mismatch of this size wraps the integer
    region. Reporting "no overlap" for a grid that covers the whole moving
    image would be a silent, wrong answer.
    """
    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 1e9, "x": 1e9}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    with pytest.raises(ValueError, match="does not contain the transformed grid"):
        itk_transform_resample_bounding_box(_identity(2), fixed, moving, padding=1)


@pytest.mark.parametrize("bad", [-1, 1.5, float("nan"), float("inf")])
def test_padding_that_is_not_a_non_negative_integer_is_rejected(bad):
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="padding must be a non-negative integer"):
        itk_transform_resample_bounding_box(_identity(2), fixed, fixed, padding=bad)


def test_unsupported_spatial_dimensionality_is_rejected():
    fixed = _image("x", {"x": 8}, {"x": 1}, {"x": 0})
    with pytest.raises(ValueError, match="only 2 and 3 are supported"):
        itk_transform_resample_bounding_box(_identity(2), fixed, fixed)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_geometry_is_rejected(bad):
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed.scale["x"] = bad
    with pytest.raises(ValueError, match="must be finite"):
        itk_transform_resample_bounding_box(_identity(2), fixed, fixed)


def test_missing_scale_entry_is_rejected():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    del fixed.scale["x"]
    with pytest.raises(ValueError, match="no entry for dimension 'x'"):
        itk_transform_resample_bounding_box(_identity(2), fixed, fixed)


def test_zero_scale_is_rejected():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 0}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="scale for dimension 'x' is zero"):
        itk_transform_resample_bounding_box(_identity(2), fixed, fixed)


def test_degenerate_fixed_grid_yields_an_empty_region():
    fixed = _image("yx", {"y": 0, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(_identity(2), fixed, moving)

    assert bounding_box.is_empty
    assert bounding_box.crop(moving) is None


def _constant_displacement_field(itk, shift, size=8, spacing=8.0, ctype=None):
    """A displacement field that shifts every point by ``shift`` (ITK order)."""
    if ctype is None:
        ctype = itk.D
    field = itk.Image[itk.Vector[ctype, 2], 2].New()
    region = itk.ImageRegion[2]()
    extent = itk.Size[2]()
    extent[0], extent[1] = size, size
    region.SetSize(extent)
    field.SetRegions(region)
    field.SetSpacing([spacing, spacing])
    field.SetOrigin([0.0, 0.0])
    field.Allocate()
    offset = itk.Vector[ctype, 2]()
    offset[0], offset[1] = shift
    field.FillBuffer(offset)

    transform = itk.DisplacementFieldTransform[ctype, 2].New()
    transform.SetDisplacementField(field)
    return transform


def test_non_linear_displacement_field_is_supported():
    """Non-linear transforms are handled: the pipeline walks the grid boundary.

    This is what deformable registration needs, so it must not be refused
    along with the transforms that genuinely cannot work.
    """
    itk = pytest.importorskip("itk")

    transform = _constant_displacement_field(itk, (5.0, 3.0))
    assert not transform.IsLinear()

    fixed = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 256, "x": 256}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        transform, fixed, moving, padding=1
    )

    # A constant field shifts ITK (x, y) by (5, 3), so NGFF (y, x) by (3, 5).
    # The fixed grid spans 0..31 on both axes.
    assert bounding_box.corners_min == {"y": 3.0, "x": 5.0}
    assert bounding_box.corners_max == {"y": 34.0, "x": 36.0}
    assert bounding_box.start_index == {"y": 2, "x": 4}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_float_displacement_field_matches_double():
    """A float32-parameterized transform yields the double-precision region.

    itk.dict_from_transform materializes the parameters as float64 while
    declaring the transform's own value type, so a float32 declaration over
    a float64 buffer must be corrected, not read as raw float32.
    """
    itk = pytest.importorskip("itk")

    fixed = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 256, "x": 256}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    reference = itk_transform_resample_bounding_box(
        _constant_displacement_field(itk, (5.0, 3.0)), fixed, moving, padding=1
    )
    float_transform = _constant_displacement_field(itk, (5.0, 3.0), ctype=itk.F)
    bounding_box = itk_transform_resample_bounding_box(
        float_transform, fixed, moving, padding=1
    )

    assert bounding_box.start_index == reference.start_index == {"y": 2, "x": 4}
    assert bounding_box.size == reference.size == {"y": 34, "x": 34}


def _identity_bspline(itk, extent=32.0):
    """An identity B-spline transform whose domain spans ``extent`` per axis."""
    bspline = itk.BSplineTransform[itk.D, 2, 3].New()
    bspline.SetTransformDomainOrigin([0.0, 0.0])
    bspline.SetTransformDomainPhysicalDimensions([extent, extent])
    bspline.SetTransformDomainMeshSize([4, 4])
    parameters = itk.OptimizerParameters[itk.D](bspline.GetNumberOfParameters())
    parameters.Fill(0.0)
    bspline.SetParameters(parameters)
    return bspline


def test_bspline_transform_is_supported():
    """A B-spline transform passes through the pipeline directly.

    itkwasm-downsample releases before 2.0.1 aborted while reconstructing
    this parameterization, so it is pinned out and covered here.
    """
    itk = pytest.importorskip("itk")

    fixed = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        _identity_bspline(itk), fixed, moving, padding=1
    )

    # An identity B-spline maps the grid onto itself; padding adds one pixel.
    assert bounding_box.start_index == {"y": -1, "x": -1}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_composite_with_bspline_stage_is_supported():
    """The affine + B-spline composite a registration returns works directly."""
    itk = pytest.importorskip("itk")

    affine = itk.AffineTransform[itk.D, 2].New()
    affine.Translate([5.0, 3.0])
    composite = itk.CompositeTransform[itk.D, 2].New()
    composite.AddTransform(affine)
    composite.AddTransform(_identity_bspline(itk))

    fixed = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 256, "x": 256}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = itk_transform_resample_bounding_box(
        composite, fixed, moving, padding=1
    )

    # The identity B-spline stage leaves the affine translation of ITK
    # (x, y) = (5, 3), so NGFF (y, x) = (3, 5), matching the displacement
    # field test above.
    assert bounding_box.start_index == {"y": 2, "x": 4}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_unsupported_transform_type_is_rejected():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(TypeError, match="unsupported transform type"):
        itk_transform_resample_bounding_box("not a transform", fixed, fixed)


def test_asymmetric_three_dimensional_affine_matches_oracle():
    """An asymmetric sheared affine makes any axis-order slip visible."""
    spatial = ("z", "y", "x")
    fixed = _image(
        spatial,
        {"z": 4, "y": 8, "x": 16},
        {"z": 3.0, "y": 2.0, "x": 1.0},
        {"z": 30.0, "y": 20.0, "x": 10.0},
    )
    moving = _image(
        spatial,
        {"z": 64, "y": 128, "x": 256},
        {"z": 1.5, "y": 0.5, "x": 0.25},
        {"z": -5.0, "y": 7.0, "x": 3.0},
    )
    # Stated in NGFF order for the oracle; the transform is built in ITK order,
    # so both the rows and the columns reverse.
    matrix = np.array([[1.0, 0.2, 0.0], [0.0, 2.0, 0.3], [0.5, 0.0, 1.0]])
    offset = np.array([4.0, -6.0, 11.0])
    reversal = np.eye(3)[::-1]

    bounding_box = itk_transform_resample_bounding_box(
        _affine(reversal @ matrix @ reversal, reversal @ offset),
        fixed,
        moving,
        padding=2,
    )

    expected_start, expected_size = _oracle_region(
        matrix, offset, fixed, moving, spatial, 2
    )
    assert [bounding_box.start_index[d] for d in spatial] == expected_start.tolist()
    assert [bounding_box.size[d] for d in spatial] == expected_size.tolist()


def _stub_result(padded_start_index, padded_size):
    corners = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    return {
        "paddedStartIndex": padded_start_index,
        "paddedSize": padded_size,
        "corners": corners,
        "paddedCorners": {"min": [0.0, 0.0], "max": [1.0, 1.0]},
    }


@pytest.mark.parametrize(
    ("padded_start_index", "padded_size", "match"),
    [
        ([0], [4, 4], "1 paddedStartIndex values for 2 dimensions"),
        ([0, 0], [4], "1 paddedSize values for 2 dimensions"),
        ([0, float("nan")], [4, 4], "paddedStartIndex for dimension 'y'"),
        ([0, 0], [4, float("inf")], "paddedSize for dimension 'y'"),
    ],
)
def test_unusable_pipeline_index_arrays_are_rejected(
    monkeypatch, padded_start_index, padded_size, match
):
    import itkwasm_downsample

    monkeypatch.setattr(
        itkwasm_downsample,
        "resample_bounding_box",
        lambda *args, **kwargs: _stub_result(padded_start_index, padded_size),
    )

    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match=match):
        itk_transform_resample_bounding_box(_identity(2), fixed, fixed)
