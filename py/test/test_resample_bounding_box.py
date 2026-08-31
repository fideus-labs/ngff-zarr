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
    itk_displacement_field_to_ngff_transform,
    ngff_image_to_itk_image,
    resample_bounding_box,
)
from ngff_zarr.ngff_transform_to_itk_transform import _ngff_transform_to_itk_matrix
from ngff_zarr.resample_bounding_box import (
    _as_itk_transform_list,
    _itk_direction,
    _metadata_only_itk_image,
    _shifted_translation,
)
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Bijection,
    ByDimension,
    ByDimensionItem,
    Coordinates,
    Displacements,
    Identity,
    MapAxis,
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

    bounding_box = resample_bounding_box(
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

    padded = resample_bounding_box(transform, fixed, moving, padding=1)
    tight = resample_bounding_box(transform, fixed, moving, padding=0)

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

    bounding_box = resample_bounding_box(
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

    bounding_box = resample_bounding_box(sequence, fixed, moving, padding=0)

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

    plain = resample_bounding_box(
        transform,
        _image(spatial, shape, scale, translation),
        _image(spatial, {"z": 32, "y": 64, "x": 96}, scale, translation),
        padding=0,
    )
    oriented = resample_bounding_box(
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
    via_rfc5 = resample_bounding_box(
        Translation(translation=[1.0, 7.0, 5.0]), fixed, moving, padding=0
    )
    # ITK parameters are fastest-axis-first by name: (x, y, z).
    via_itk = resample_bounding_box(
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

    bounding_box = resample_bounding_box(
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
        resample_bounding_box(_identity(2), fixed, moving)


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
    bounding_box = resample_bounding_box(_identity(2), fixed, fixed, padding=1)

    assert not computed
    assert bounding_box.start_index == {"y": -1, "x": -1}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_non_spatial_axes_are_passed_through():
    shape = {"t": 3, "c": 2, "z": 4, "y": 8, "x": 8}
    unit = dict.fromkeys(shape, 1)
    zero = dict.fromkeys(shape, 0)
    fixed = _image("tczyx", shape, unit, zero)
    moving = _image("tczyx", {"t": 3, "c": 2, "z": 16, "y": 32, "x": 32}, unit, zero)

    bounding_box = resample_bounding_box(
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

    bounding_box = resample_bounding_box(
        _translation([1000.0, 1000.0]), fixed, moving, padding=1
    )

    assert bounding_box.is_empty
    assert bounding_box.crop(moving) is None


def test_negative_start_index_is_clamped_not_wrapped():
    """A negative slice bound would wrap silently instead of raising."""
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = resample_bounding_box(_identity(2), fixed, moving, padding=2)

    assert bounding_box.start_index == {"y": -2, "x": -2}
    assert bounding_box.clamped() == {"y": (0, 6), "x": (0, 6)}
    assert bounding_box.slices() == (slice(0, 6), slice(0, 6))
    assert not bounding_box.is_empty


def test_crop_is_lazy_and_shifts_the_translation():
    fixed = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 512, "x": 1024})
    moving = _image("yx", {"y": 4096, "x": 4096}, {"y": 2, "x": 2}, {"y": 0, "x": 0})

    bounding_box = resample_bounding_box(
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


def test_crop_binds_itk_axes_by_name_in_a_non_canonical_order():
    """A sub-grid's origin moves along the *oriented* axes.

    ``translation`` is keyed by name while the direction matrix is in ITK
    component order, so the shift has to be read back by name too. Reversing
    ``dims`` gives that order for ``zyx`` and ``yx``, and for nothing else.
    """
    moving = _image(
        "xyz",
        {"x": 32, "y": 32, "z": 32},
        {"x": 0.5, "y": 1.0, "z": 2.0},
        {"x": 3.0, "y": 2.0, "z": 1.0},
        RAS,
    )

    shifted = _shifted_translation(moving, {"x": 4, "y": 6, "z": 8})

    # RAS is diag(-1, -1, 1) in ITK order, so x and y count backwards.
    assert shifted == pytest.approx({"x": 1.0, "y": -4.0, "z": 17.0})


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
    bounding_box = resample_bounding_box(_identity(3), fixed, moving)
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

    bounding_box = resample_bounding_box(transform, fixed, moving, padding=1)

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

    bounding_box = resample_bounding_box(composite, fixed, moving, padding=0)

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
        resample_bounding_box(_identity(2), fixed, moving, padding=1)


@pytest.mark.parametrize("bad", [-1, 1.5, float("nan"), float("inf")])
def test_padding_that_is_not_a_non_negative_integer_is_rejected(bad):
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="padding must be a non-negative integer"):
        resample_bounding_box(_identity(2), fixed, fixed, padding=bad)


def test_unsupported_spatial_dimensionality_is_rejected():
    fixed = _image("x", {"x": 8}, {"x": 1}, {"x": 0})
    with pytest.raises(ValueError, match="only 2 and 3 are supported"):
        resample_bounding_box(_identity(2), fixed, fixed)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_geometry_is_rejected(bad):
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed.scale["x"] = bad
    with pytest.raises(ValueError, match="must be finite"):
        resample_bounding_box(_identity(2), fixed, fixed)


def test_missing_scale_entry_is_rejected():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    del fixed.scale["x"]
    with pytest.raises(ValueError, match="no entry for dimension 'x'"):
        resample_bounding_box(_identity(2), fixed, fixed)


def test_zero_scale_is_rejected():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 0}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="scale for dimension 'x' is zero"):
        resample_bounding_box(_identity(2), fixed, fixed)


def test_degenerate_fixed_grid_yields_an_empty_region():
    fixed = _image("yx", {"y": 0, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    bounding_box = resample_bounding_box(_identity(2), fixed, moving)

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

    bounding_box = resample_bounding_box(transform, fixed, moving, padding=1)

    # A constant field shifts ITK (x, y) by (5, 3), so NGFF (y, x) by (3, 5).
    # The fixed grid spans 0..31 on both axes.
    assert bounding_box.corners_min == {"y": 3.0, "x": 5.0}
    assert bounding_box.corners_max == {"y": 34.0, "x": 36.0}
    assert bounding_box.start_index == {"y": 2, "x": 4}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_rfc5_displacements_matches_the_itk_field_it_came_from():
    """The RFC-5 branch has to reach the same region for a field as for an
    affine, which means the field it points at has to reach the pipeline."""
    itk = pytest.importorskip("itk")

    fixed = _image("yx", {"y": 8, "x": 8}, {"y": 8.0, "x": 8.0}, {"y": 0.0, "x": 0.0})
    moving = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0}
    )
    warp = _constant_displacement_field(itk, [5.0, -3.0], size=8, spacing=8.0)

    via_itk = resample_bounding_box(warp, fixed, moving)
    transform, field = itk_displacement_field_to_ngff_transform(
        warp, ("y", "x"), path="warp"
    )
    via_rfc5 = resample_bounding_box(transform, fixed, moving, fields={"warp": field})

    assert via_rfc5.start_index == via_itk.start_index
    assert via_rfc5.size == via_itk.size

    # Without the field there is nothing to convert, and the message says so.
    with pytest.raises(ValueError, match="no field was passed"):
        resample_bounding_box(transform, fixed, moving)


def test_float_displacement_field_matches_double():
    """A float32-parameterized transform yields the double-precision region.

    itk.dict_from_transform materializes the parameters as float64 while
    declaring the transform's own value type, so a float32 declaration over
    a float64 buffer must be corrected, not read as raw float32.
    """
    itk = pytest.importorskip("itk")

    fixed = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image("yx", {"y": 256, "x": 256}, {"y": 1, "x": 1}, {"y": 0, "x": 0})

    reference = resample_bounding_box(
        _constant_displacement_field(itk, (5.0, 3.0)), fixed, moving, padding=1
    )
    float_transform = _constant_displacement_field(itk, (5.0, 3.0), ctype=itk.F)
    bounding_box = resample_bounding_box(float_transform, fixed, moving, padding=1)

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

    bounding_box = resample_bounding_box(
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

    bounding_box = resample_bounding_box(composite, fixed, moving, padding=1)

    # The identity B-spline stage leaves the affine translation of ITK
    # (x, y) = (5, 3), so NGFF (y, x) = (3, 5), matching the displacement
    # field test above.
    assert bounding_box.start_index == {"y": 2, "x": 4}
    assert bounding_box.size == {"y": 34, "x": 34}


def test_unsupported_transform_type_is_rejected():
    fixed = _image("yx", {"y": 4, "x": 4}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(TypeError, match="unsupported transform type"):
        resample_bounding_box("not a transform", fixed, fixed)


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

    bounding_box = resample_bounding_box(
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
        resample_bounding_box(_identity(2), fixed, fixed)


def test_map_axis_region_matches_the_oracle():
    """A permutation reaches the region the matrix it stands for reaches."""
    spatial = ("z", "y", "x")
    fixed = _image(
        spatial,
        {"z": 12, "y": 20, "x": 16},
        {"z": 2.0, "y": 1.0, "x": 0.5},
        {"z": 3.0, "y": -4.0, "x": 6.0},
    )
    moving = _image(
        spatial,
        {"z": 64, "y": 64, "x": 64},
        {"z": 1.0, "y": 1.0, "x": 1.0},
        {"z": 0.0, "y": 0.0, "x": 0.0},
    )
    # Output axis i takes input axis mapAxis[i], so row i selects that column.
    matrix = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    bounding_box = resample_bounding_box(
        MapAxis(mapAxis=[1, 2, 0]), fixed, moving, padding=1
    )

    expected_start, expected_size = _oracle_region(
        matrix, np.zeros(3), fixed, moving, spatial, 1
    )
    assert [bounding_box.start_index[d] for d in spatial] == expected_start.tolist()
    assert [bounding_box.size[d] for d in spatial] == expected_size.tolist()


def test_by_dimension_region_matches_the_oracle():
    spatial = ("z", "y", "x")
    fixed = _image(
        spatial,
        {"z": 10, "y": 24, "x": 24},
        {"z": 1.0, "y": 0.5, "x": 0.5},
        {"z": -2.0, "y": 1.0, "x": 4.0},
    )
    moving = _image(
        spatial,
        {"z": 64, "y": 128, "x": 128},
        {"z": 1.0, "y": 0.25, "x": 0.25},
        {"z": 0.0, "y": 0.0, "x": 0.0},
    )
    transform = ByDimension(
        transformations=[
            ByDimensionItem(
                transformation=Translation(translation=[5.0]),
                inputAxes=[0],
                outputAxes=[0],
            ),
            ByDimensionItem(
                transformation=Affine(affine=[[0.8, -0.6, 2.0], [0.6, 0.8, -3.0]]),
                inputAxes=[1, 2],
                outputAxes=[1, 2],
            ),
        ]
    )
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]])
    offset = np.array([5.0, 2.0, -3.0])

    bounding_box = resample_bounding_box(transform, fixed, moving, padding=2)

    expected_start, expected_size = _oracle_region(
        matrix, offset, fixed, moving, spatial, 2
    )
    assert [bounding_box.start_index[d] for d in spatial] == expected_start.tolist()
    assert [bounding_box.size[d] for d in spatial] == expected_size.tolist()


def test_bijection_region_follows_its_forward_direction():
    spatial = ("y", "x")
    fixed = _image(spatial, {"y": 16, "x": 16}, {"y": 1.0, "x": 1.0}, {"y": 0, "x": 0})
    moving = _image(
        spatial, {"y": 128, "x": 128}, {"y": 1.0, "x": 1.0}, {"y": 0, "x": 0}
    )
    forward = Scale(scale=[2.0, 4.0])
    transform = Bijection(forward=forward, inverse=Scale(scale=[0.5, 0.25]))

    bounding_box = resample_bounding_box(transform, fixed, moving, padding=1)
    directly = resample_bounding_box(forward, fixed, moving, padding=1)

    assert bounding_box.start_index == directly.start_index
    assert bounding_box.size == directly.size


def test_rfc5_coordinates_matches_the_displacements_it_equals():
    """A coordinates field names the same points a displacements field does."""
    itk = pytest.importorskip("itk")

    fixed = _image("yx", {"y": 8, "x": 8}, {"y": 8.0, "x": 8.0}, {"y": 0.0, "x": 0.0})
    moving = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0}
    )
    warp = _constant_displacement_field(itk, [5.0, -3.0], size=8, spacing=8.0)
    displacements, field = itk_displacement_field_to_ngff_transform(
        warp, ("y", "x"), path="warp"
    )

    grid = np.meshgrid(
        *[
            field.translation[dim] + field.scale[dim] * np.arange(extent)
            for dim, extent in zip(("y", "x"), field.data.shape[1:])
        ],
        indexing="ij",
    )
    absolute = NgffImage(
        data=da.from_array(np.asarray(field.data) + np.stack(grid)),
        dims=field.dims,
        scale=dict(field.scale),
        translation=dict(field.translation),
        axes_types={"c": "coordinate"},
    )

    via_displacements = resample_bounding_box(
        displacements, fixed, moving, fields={"warp": field}
    )
    via_coordinates = resample_bounding_box(
        Coordinates(path="warp"), fixed, moving, fields={"warp": absolute}
    )

    assert via_coordinates.start_index == via_displacements.start_index
    assert via_coordinates.size == via_displacements.size


def _interior_bump(extent, radius, peaks):
    """A displacement that is zero on the grid boundary and peaks in the middle.

    The boundary walk the pipeline runs sees only the zero, which is what
    makes this the case a region has to be widened for.
    """
    axes = np.mgrid[tuple(slice(0, extent) for _ in peaks)].astype(np.float64)
    distance = np.sqrt(sum((axis - extent / 2) ** 2 for axis in axes))
    profile = np.where(
        distance < radius, 0.5 * (1 + np.cos(np.pi * distance / radius)), 0.0
    )
    return np.stack([peak * profile for peak in peaks]).astype(np.float32)


def test_a_bump_inside_the_grid_widens_the_region():
    """The reported region has to contain what the field displaces onto.

    The pipeline sizes a region by walking the boundary of the transformed
    grid, so a displacement that is zero there and large inside left the
    region covering the grid alone. The peak here carries the grid past its
    own extent, so that is not enough.
    """
    fixed = _image("yx", {"y": 64, "x": 64}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0})
    moving = _image(
        "yx", {"y": 160, "x": 160}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0}
    )
    values = _interior_bump(64, 20, (60.0, -60.0))
    assert float(np.abs(values[:, 0, :]).max()) == 0.0
    assert float(np.abs(values[:, -1, :]).max()) == 0.0
    field = NgffImage(
        data=da.from_array(values, chunks=(2, 16, 16)),
        dims=("c", "y", "x"),
        scale=dict.fromkeys(("c", "y", "x"), 1.0),
        translation=dict.fromkeys(("c", "y", "x"), 0.0),
        axes_types={"c": "displacement"},
    )

    region = resample_bounding_box(
        Displacements(path="warp"), fixed, moving, fields={"warp": field}
    )

    index = np.mgrid[0:64, 0:64].astype(np.float64)
    for axis, dim in enumerate(("y", "x")):
        reached = index[axis] + values[axis]
        assert region.start_index[dim] <= float(reached.min())
        assert region.start_index[dim] + region.size[dim] >= float(reached.max())


def test_a_fields_chunking_does_not_change_the_region_it_reports():
    """The range is read a chunk at a time; over the whole grid it is one range."""
    fixed = _image("yx", {"y": 64, "x": 64}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0})
    moving = _image(
        "yx", {"y": 160, "x": 160}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0}
    )
    values = _interior_bump(64, 20, (60.0, -60.0))

    regions = []
    for chunks in ((2, 16, 16), (2, 64, 64)):
        field = NgffImage(
            data=da.from_array(values, chunks=chunks),
            dims=("c", "y", "x"),
            scale=dict.fromkeys(("c", "y", "x"), 1.0),
            translation=dict.fromkeys(("c", "y", "x"), 0.0),
            axes_types={"c": "displacement"},
        )
        regions.append(
            resample_bounding_box(
                Displacements(path="warp"), fixed, moving, fields={"warp": field}
            )
        )

    assert regions[0].start_index == regions[1].start_index
    assert regions[0].size == regions[1].size


def test_the_identity_region_is_the_pipelines():
    """The arithmetic region and the pipeline's agree, field for field.

    The field path derives its region from `_identity_region` rather than a
    per-block pipeline call, so the two must not drift: randomized geometry,
    fractional and integer scales and translations, paddings 0 to 3.
    """
    from ngff_zarr.ngff_transform_to_itk_transform import (
        ngff_transform_to_itk_transform,
    )
    from ngff_zarr.resample_bounding_box import _identity_region

    rng = np.random.default_rng(0)
    for _ in range(25):
        ndim = int(rng.integers(2, 4))
        dims = ("z", "y", "x")[-ndim:]

        def geometry():
            kind = int(rng.integers(0, 3))
            if kind == 0:
                return 1.0
            if kind == 1:
                return float(rng.integers(1, 4))
            return float(np.round(rng.random() * 3 + 0.25, 3))

        def offset():
            if rng.random() < 0.7:
                return float(np.round(rng.random() * 20 - 10, 3))
            return float(rng.integers(-10, 10))

        grid = NgffImage(
            data=da.zeros(tuple(int(v) for v in rng.integers(1, 40, ndim))),
            dims=dims,
            scale={dim: geometry() for dim in dims},
            translation={dim: offset() for dim in dims},
        )
        moving = NgffImage(
            data=da.zeros(tuple(int(v) for v in rng.integers(8, 200, ndim))),
            dims=dims,
            scale={dim: geometry() for dim in dims},
            translation={dim: offset() for dim in dims},
        )
        padding = int(rng.integers(0, 4))
        identity = ngff_transform_to_itk_transform(Identity(), dims)

        pipeline = resample_bounding_box(identity, grid, moving, padding=padding)
        direct = _identity_region(grid, moving, padding)

        assert direct.dims == pipeline.dims
        assert direct.start_index == pipeline.start_index
        assert direct.size == pipeline.size
        assert direct.moving_shape == pipeline.moving_shape
        for mine, theirs in (
            (direct.corners_min, pipeline.corners_min),
            (direct.corners_max, pipeline.corners_max),
            (direct.padded_corners_min, pipeline.padded_corners_min),
            (direct.padded_corners_max, pipeline.padded_corners_max),
        ):
            for dim in dims:
                np.testing.assert_allclose(mine[dim], theirs[dim], rtol=1e-12)


def _bump_displacement_transform(itk, extent=64, radius=20, peaks=(60.0, -60.0)):
    """A field that is zero on the grid boundary and peaks in the middle.

    The boundary walk sees only the zero, which is what the interval has to
    make up for. ``peaks`` are the (y, x) amplitudes; the field's grid is
    unit-spaced at the origin.
    """
    grid_y, grid_x = np.mgrid[0:extent, 0:extent].astype(np.float64)
    distance = np.sqrt((grid_y - extent / 2) ** 2 + (grid_x - extent / 2) ** 2)
    profile = np.where(
        distance < radius, 0.5 * (1 + np.cos(np.pi * distance / radius)), 0.0
    )
    field = itk.Image[itk.Vector[itk.D, 2], 2].New()
    field.SetRegions(itk.ImageRegion[2]([extent, extent]))
    field.Allocate()
    view = itk.array_view_from_image(field)
    view[..., 0] = peaks[1] * profile
    view[..., 1] = peaks[0] * profile
    transform = itk.DisplacementFieldTransform[itk.D, 2].New()
    transform.SetDisplacementField(field)
    return transform, profile


def test_the_stage_interval_reads_the_layouts_itk_writes():
    """A field interleaves components per point; a B-spline blocks them.

    Pinned against ITK itself: one distinctive value planted per component
    must come back on its own component, not on a neighbour's. Misreading the
    layout would produce plausible wrong bounds silently.
    """
    itk = pytest.importorskip("itk")
    from ngff_zarr.resample_bounding_box import _stage_interval

    field = itk.Image[itk.Vector[itk.D, 2], 2].New()
    field.SetRegions(itk.ImageRegion[2]([4, 3]))
    field.Allocate()
    field.FillBuffer(itk.Vector[itk.D, 2]())
    vector = itk.Vector[itk.D, 2]()
    vector[0], vector[1] = 7.0, -9.0
    field.SetPixel([2, 1], vector)
    warp = itk.DisplacementFieldTransform[itk.D, 2].New()
    warp.SetDisplacementField(field)
    (entry,) = _as_itk_transform_list(warp)
    low, high = _stage_interval(entry, 2)
    assert (low.tolist(), high.tolist()) == ([0.0, -9.0], [7.0, 0.0])

    spline = itk.BSplineTransform[itk.D, 2, 3].New()
    mesh = itk.Size[2]()
    mesh.Fill(3)
    spline.SetTransformDomainMeshSize(mesh)
    count = spline.GetNumberOfParameters()
    parameters = itk.OptimizerParameters[itk.D](count)
    for index in range(count):
        parameters.SetElement(index, 0.0)
    parameters.SetElement(5, 11.0)
    parameters.SetElement(count // 2 + 5, -13.0)
    spline.SetParameters(parameters)
    (entry,) = _as_itk_transform_list(spline)
    low, high = _stage_interval(entry, 2)
    assert (low.tolist(), high.tolist()) == ([0.0, -13.0], [11.0, 0.0])


def test_an_interior_bump_in_an_itk_field_widens_the_region():
    """A displacement the grid's boundary does not show still gets covered.

    The walk reported the identity region for this field and resample
    returned the default value for 64 of 4096 pixels, by up to 93.
    """
    itk = pytest.importorskip("itk")
    warp, profile = _bump_displacement_transform(itk)

    fixed = _image("yx", {"y": 64, "x": 64}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0})
    moving = _image(
        "yx", {"y": 256, "x": 256}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0}
    )

    region = resample_bounding_box(warp, fixed, moving, padding=1)

    grid_y, grid_x = np.mgrid[0:64, 0:64].astype(np.float64)
    for dim, reached in (
        ("y", grid_y + 60.0 * profile),
        ("x", grid_x - 60.0 * profile),
    ):
        assert region.start_index[dim] <= float(reached.min())
        assert region.start_index[dim] + region.size[dim] >= float(reached.max())


def test_an_interior_bump_resamples_exactly_in_one_block():
    """One output block over the whole grid, against a whole-image call.

    Small blocks hide the miss by accident, their boundaries crossing the
    bump; one block is the case the walk got wrong.
    """
    itk = pytest.importorskip("itk")
    from ngff_zarr import resample

    warp, _profile = _bump_displacement_transform(itk)
    # One block over the whole grid, said explicitly: left to itself dask
    # cuts this small array into blocks whose boundaries cross the bump, and
    # the walk then finds by accident what it misses on the block at stake.
    fixed = NgffImage(
        data=da.zeros((64, 64), chunks=(64, 64), dtype=np.float32),
        dims=("y", "x"),
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    # The images this file builds carry no pixels, which the bounding box
    # never reads; this test resamples, so the moving image has to carry
    # something a missed region would lose.
    moving = NgffImage(
        data=da.from_array(
            (np.random.default_rng(0).random((256, 256)) * 100).astype(np.float32)
        ),
        dims=("y", "x"),
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )

    result = np.asarray(resample(warp, fixed, moving).data)

    from itkwasm_downsample import resample_to_reference
    from ngff_zarr.ngff_image_to_itk_image import ngff_image_to_itk_image
    from ngff_zarr.resample import _component_type
    from ngff_zarr.resample_bounding_box import _spatial_dims

    itk_dims = list(reversed(_spatial_dims(fixed)))
    reference = _metadata_only_itk_image(
        fixed, itk_dims, _itk_direction(fixed, itk_dims)
    )
    reference.imageType.componentType = _component_type(moving.data.dtype)
    expected = np.asarray(
        resample_to_reference(
            ngff_image_to_itk_image(moving, wasm=True),
            reference,
            transform=_as_itk_transform_list(warp),
            interpolator="linear",
        ).data
    ).reshape((64, 64))
    np.testing.assert_array_equal(result, expected)


def test_an_affine_around_the_field_folds_its_interval():
    """A composite of an affine and a field bounds through the affine's signs."""
    itk = pytest.importorskip("itk")
    from ngff_zarr.resample_bounding_box import _nonlinear_interval

    warp, _profile = _bump_displacement_transform(itk, peaks=(60.0, -60.0))
    scaling = itk.AffineTransform[itk.D, 2].New()
    scaling.Scale(2.0)
    composite = itk.CompositeTransform[itk.D, 2].New()
    composite.AddTransform(warp)
    composite.AddTransform(scaling)  # applied first: entries = [warp? or scaling?]

    entries = _as_itk_transform_list(composite)
    walked, interval = _nonlinear_interval(entries, 2)

    assert interval is not None
    low, high = interval
    # The field is the outermost stage here, so its range passes through no
    # affine and keeps its own numbers, zero included.
    assert low.tolist() == [-60.0, 0.0]
    assert high.tolist() == [0.0, 60.0]

    reordered = _as_itk_transform_list([entries[1], entries[0]])
    _walked, folded = _nonlinear_interval(reordered, 2)
    # The affine is outermost now, so the field's range doubles through it.
    assert folded[0].tolist() == [-120.0, 0.0]
    assert folded[1].tolist() == [0.0, 120.0]


def test_a_lone_bspline_keeps_zero_in_its_range():
    """A B-spline's lattice cannot prove the grid stays on its domain.

    The control lattice reaches spline-order points beyond the domain of
    support, so a grid inside the lattice can still land where ITK displaces
    it by nothing: coefficients all one way must not carry the region off
    the undisplaced points.
    """
    itk = pytest.importorskip("itk")

    spline = itk.BSplineTransform[itk.D, 2, 3].New()
    mesh = itk.Size[2]()
    mesh.Fill(3)
    spline.SetTransformDomainMeshSize(mesh)
    physical = itk.Vector[itk.D, 2]()
    physical[0], physical[1] = 32.0, 32.0
    spline.SetTransformDomainPhysicalDimensions(physical)
    count = spline.GetNumberOfParameters()
    parameters = itk.OptimizerParameters[itk.D](count)
    for index in range(count):
        parameters.SetElement(index, 10.0)
    spline.SetParameters(parameters)

    fixed = _image("yx", {"y": 16, "x": 16}, {"y": 1.0, "x": 1.0}, {"y": 8.0, "x": 8.0})
    moving = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0}
    )

    region = resample_bounding_box(spline, fixed, moving, padding=0)

    for dim in ("y", "x"):
        # Zero in the range keeps the grid's own undisplaced extent covered.
        assert region.start_index[dim] <= 8
        assert region.start_index[dim] + region.size[dim] >= 8 + 15 + 10
