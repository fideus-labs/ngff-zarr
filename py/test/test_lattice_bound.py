# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The stage-by-stage region holds every point of a block, and reads little."""

import math

import numpy as np
import pytest
from ngff_zarr import to_ngff_image
from ngff_zarr._lattice_bound import bspline_displacement, transform_box
from ngff_zarr.resample_bounding_box import resample_bounding_box

itk = pytest.importorskip("itk")


def _image(shape, scale, translation):
    return to_ngff_image(
        np.zeros(shape, dtype=np.uint8),
        dims=["z", "y", "x"][-len(shape) :],
        scale=scale,
        translation=translation,
    )


def _bspline(dimension, extent, mesh, coefficients):
    """An ITK cubic B-spline over ``[0, extent]^d`` with ``mesh`` cells per axis."""
    spline = itk.BSplineTransform[itk.D, dimension, 3].New()
    spline.SetTransformDomainOrigin([0.0] * dimension)
    spline.SetTransformDomainPhysicalDimensions([float(extent)] * dimension)
    spline.SetTransformDomainMeshSize([int(mesh)] * dimension)
    spline.SetTransformDomainDirection(itk.matrix_from_array(np.eye(dimension)))
    values = coefficients(spline.GetNumberOfParameters())
    spline.SetParameters(itk.OptimizerParameters[itk.D](values.tolist()))
    return spline


def _exact_region(transform, fixed, moving, padding=1):
    """Every pixel of ``fixed`` mapped: the region any bound must contain."""
    dims = [d for d in fixed.dims]
    itk_dims = [d for d in ("x", "y", "z") if d in dims]
    dimension = len(itk_dims)
    reference = itk.Image[itk.UC, dimension].New()
    region = itk.ImageRegion[dimension]()
    region.SetSize([int(fixed.data.shape[dims.index(d)]) for d in itk_dims])
    reference.SetRegions(region)
    reference.SetSpacing([float(fixed.scale[d]) for d in itk_dims])
    reference.SetOrigin([float(fixed.translation[d]) for d in itk_dims])
    reference.Allocate()
    field = itk.array_view_from_image(
        itk.transform_to_displacement_field_filter(
            transform, use_reference_image=True, reference_image=reference
        )
    )
    bounds = {}
    for component, dim in enumerate(itk_dims):
        axis = dims.index(dim)
        positions = (
            fixed.translation[dim]
            + np.arange(fixed.data.shape[axis]) * fixed.scale[dim]
        )
        shape = [1] * dimension
        shape[axis] = fixed.data.shape[axis]
        landing = field[..., component].astype(float) + positions.reshape(shape)
        low = (landing.min() - moving.translation[dim]) / moving.scale[dim]
        high = (landing.max() - moving.translation[dim]) / moving.scale[dim]
        bounds[dim] = (math.floor(low) - padding, math.ceil(high) + padding + 1)
    return bounds


def _assert_contains(bounding_box, exact):
    for dim, (start, stop) in exact.items():
        assert bounding_box.start_index[dim] <= start, dim
        assert bounding_box.start_index[dim] + bounding_box.size[dim] >= stop, dim


def _reads(bounding_box):
    return math.prod(stop - start for start, stop in bounding_box.clamped().values())


def _exact_reads(exact, moving_shape):
    return math.prod(
        max(0, min(stop, extent) - max(start, 0))
        for (start, stop), extent in zip(exact.values(), moving_shape)
    )


def test_the_spline_evaluation_is_itks():
    rng = np.random.default_rng(0)
    spline = _bspline(2, 64.0, 6, lambda n: rng.normal(0, 3, n))
    fixed = np.asarray(spline.GetFixedParameters(), dtype=float)
    size = fixed[:2].astype(int)
    coefficients = np.asarray(spline.GetParameters()).reshape(2, *size[::-1])
    points = rng.uniform(-10, 74, size=(2000, 2))
    index = (points - fixed[2:4]) / fixed[4:6]

    displacement = bspline_displacement(coefficients, size, index)

    expected = np.array(
        [np.subtract(spline.TransformPoint(p.tolist()), p) for p in points]
    )
    assert np.allclose(displacement, expected, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_random_spline_is_contained_on_every_block(seed):
    rng = np.random.default_rng(seed)
    spline = _bspline(2, 128.0, 8, lambda n: rng.normal(0, 6, n))
    moving = _image((160, 160), {"y": 1.0, "x": 1.0}, {"y": -16.0, "x": -16.0})
    total = 0
    exact_total = 0
    for y in range(0, 128, 32):
        for x in range(0, 128, 32):
            fixed = _image(
                (32, 32), {"y": 1.0, "x": 1.0}, {"y": float(y), "x": float(x)}
            )
            bounding_box = resample_bounding_box(spline, fixed, moving)
            exact = _exact_region(spline, fixed, moving)
            _assert_contains(bounding_box, exact)
            total += _reads(bounding_box)
            exact_total += _exact_reads(exact, moving.data.shape)
    assert total <= 1.6 * exact_total


def test_a_bump_strictly_inside_a_block_is_contained():
    def coefficients(n):
        values = np.zeros(n)
        grid = int(math.sqrt(n // 2))
        values[(grid // 2) * grid + grid // 2] = 40.0  # one control point, x component
        return values

    spline = _bspline(2, 128.0, 8, coefficients)
    fixed = _image((128, 128), {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0})
    moving = _image((256, 256), {"y": 1.0, "x": 1.0}, {"y": -64.0, "x": -64.0})

    bounding_box = resample_bounding_box(spline, fixed, moving)

    _assert_contains(bounding_box, _exact_region(spline, fixed, moving))


def test_a_block_past_the_spline_domain_is_contained():
    rng = np.random.default_rng(3)
    spline = _bspline(2, 64.0, 4, lambda n: rng.normal(0, 8, n))
    fixed = _image((48, 48), {"y": 1.0, "x": 1.0}, {"y": 40.0, "x": 40.0})
    moving = _image((160, 160), {"y": 1.0, "x": 1.0}, {"y": -16.0, "x": -16.0})

    bounding_box = resample_bounding_box(spline, fixed, moving)

    _assert_contains(bounding_box, _exact_region(spline, fixed, moving))


def test_a_rigid_then_spline_composite_is_contained():
    rng = np.random.default_rng(4)
    spline = _bspline(2, 160.0, 8, lambda n: rng.normal(0, 5, n))
    rigid = itk.Euler2DTransform[itk.D].New()
    rigid.SetAngle(0.2)
    rigid.SetTranslation([7.0, -3.0])
    composite = itk.CompositeTransform[itk.D, 2].New()
    composite.AddTransform(spline)
    composite.AddTransform(rigid)
    moving = _image((256, 256), {"y": 1.0, "x": 1.0}, {"y": -48.0, "x": -48.0})
    for y in (16, 64):
        for x in (16, 64):
            fixed = _image(
                (48, 48), {"y": 1.0, "x": 1.0}, {"y": float(y), "x": float(x)}
            )
            bounding_box = resample_bounding_box(composite, fixed, moving)
            _assert_contains(bounding_box, _exact_region(composite, fixed, moving))


def test_a_displacement_field_at_the_grids_resolution_is_exact():
    rng = np.random.default_rng(5)
    field = itk.image_from_array(
        rng.normal(0, 2, size=(64, 64, 2)).astype(np.float64), is_vector=True
    )
    transform = itk.DisplacementFieldTransform[itk.D, 2].New()
    transform.SetDisplacementField(field)
    moving = _image((96, 96), {"y": 1.0, "x": 1.0}, {"y": -16.0, "x": -16.0})
    for y in (0, 32):
        for x in (0, 32):
            fixed = _image(
                (32, 32), {"y": 1.0, "x": 1.0}, {"y": float(y), "x": float(x)}
            )
            bounding_box = resample_bounding_box(transform, fixed, moving)
            exact = _exact_region(transform, fixed, moving)
            _assert_contains(bounding_box, exact)
            assert _reads(bounding_box) == _exact_reads(exact, moving.data.shape)


def test_a_linear_stage_matches_the_pipeline(monkeypatch):
    import sys

    affine = itk.AffineTransform[itk.D, 2].New()
    affine.SetMatrix(itk.matrix_from_array(np.array([[1.1, 0.2], [-0.1, 0.9]])))
    affine.SetTranslation([3.0, -5.0])
    fixed = _image((40, 24), {"y": 1.0, "x": 1.0}, {"y": 10.0, "x": 20.0})
    moving = _image((128, 128), {"y": 1.0, "x": 1.0}, {"y": 0.0, "x": 0.0})

    staged = resample_bounding_box(affine, fixed, moving)
    module = sys.modules["ngff_zarr.resample_bounding_box"]
    monkeypatch.setattr(module, "_stage_region", lambda *args, **kwargs: None)
    walked = resample_bounding_box(affine, fixed, moving)

    assert staged.start_index == walked.start_index
    assert staged.size == walked.size


def test_a_bspline_stage_reports_the_same_region_as_typescript():
    """ts/test/resample_bounding_box_test.ts checks these numbers too."""
    from itkwasm import FloatTypes, Transform, TransformParameterizations, TransformType

    size = [7, 6]
    ix, iy = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing="xy")
    cx = 6.0 * np.sin(ix * 1.1 + iy * 0.7) + 2.0 * np.cos(iy * 0.5)
    cy = -5.0 * np.cos(ix * 0.9 - iy * 0.6) + 3.0 * np.sin(ix * 0.4)
    parameters = np.round(np.concatenate([cx.ravel(), cy.ravel()]), 6)
    entry = Transform(
        transformType=TransformType(
            transformParameterization=TransformParameterizations.BSpline,
            parametersValueType=FloatTypes.Float64,
            inputDimension=2,
            outputDimension=2,
        ),
        numberOfFixedParameters=10,
        numberOfParameters=parameters.size,
        name="BSplineTransform",
        inputSpaceName="",
        outputSpaceName="",
        fixedParameters=np.array([*size, -16.0, -16.0, 16.0, 16.0, 1, 0, 0, 1]),
        parameters=parameters,
    )
    moving = _image((128, 128), {"y": 1.0, "x": 1.0}, {"y": -32.0, "x": -32.0})
    cases = [
        (0, 0, 32, 32, {"y": 28, "x": 30}, {"y": 38, "x": 34}),
        (20, 24, 24, 40, {"y": 50, "x": 50}, {"y": 33, "x": 52}),
        (40, 8, 16, 48, {"y": 67, "x": 31}, {"y": 30, "x": 66}),
    ]
    for y, x, height, width, start_index, size_ in cases:
        fixed = _image(
            (height, width), {"y": 1.0, "x": 1.0}, {"y": float(y), "x": float(x)}
        )
        region = resample_bounding_box([entry], fixed, moving)
        assert region.start_index == start_index
        assert region.size == size_


def test_a_composite_entry_is_left_to_the_pipeline():
    """A nested composite's children can be missing from the list."""
    from itkwasm import FloatTypes, Transform, TransformParameterizations, TransformType

    header = Transform(
        transformType=TransformType(
            transformParameterization=TransformParameterizations.Composite,
            parametersValueType=FloatTypes.Float64,
            inputDimension=2,
            outputDimension=2,
        ),
        numberOfFixedParameters=0,
        numberOfParameters=0,
        name="CompositeTransform",
        inputSpaceName="",
        outputSpaceName="",
        fixedParameters=np.zeros(0),
        parameters=np.zeros(0),
    )
    assert transform_box([header], 2, [0.0, 0.0], [8.0, 8.0], [1.0, 1.0]) is None
