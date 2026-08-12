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
