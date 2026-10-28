# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Resampling through an RFC-5 ``sequence`` of an affine and a displacement field.

The shape of a rigid alignment refined by a deformable stage. The anchor is
``itk`` itself: the same affine and field, built independently as an
``itk.CompositeTransform``, resampled whole by ``itk.ResampleImageFilter``.
"""

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import NgffImage, resample
from ngff_zarr.v06.zarr_metadata import Affine, Displacements, TransformSequence

from .test_resample_bounding_box import _interior_bump

itk = pytest.importorskip("itk")


def _image(values, dims, spacing, origin, chunks):
    return NgffImage(
        data=da.from_array(values, chunks=chunks),
        dims=dims,
        scale=dict(zip(dims, spacing)),
        translation=dict(zip(dims, origin)),
    )


def _sequence(affine):
    return TransformSequence(
        transformations=[
            Affine(affine=affine.tolist()),
            Displacements(path="warp", interpolation="linear"),
        ]
    )


def _itk_reference(moving, fixed, field_values, affine):
    """``affine`` first, then the field, as one ITK composite; resampled whole.

    Every array is in Zarr order (rows of ``affine`` and field components
    too), reversed here into ITK's x, y, z.
    """
    dimension = len(fixed.dims)

    def itk_image(values, image, is_vector=False):
        out = itk.image_from_array(values, is_vector=is_vector)
        out.SetSpacing([image.scale[d] for d in reversed(image.dims)])
        out.SetOrigin([image.translation[d] for d in reversed(image.dims)])
        return out

    linear = itk.AffineTransform[itk.D, dimension].New()
    linear.SetMatrix(itk.matrix_from_array(affine[::-1, :-1][:, ::-1].copy()))
    linear.SetTranslation(affine[::-1, -1].tolist())
    field = itk_image(
        np.moveaxis(field_values[::-1], 0, -1).astype(np.float64).copy(),
        fixed,
        is_vector=True,
    )
    displacement = itk.DisplacementFieldTransform[itk.D, dimension].New()
    displacement.SetDisplacementField(field)

    # A CompositeTransform applies the transform added last first.
    composite = itk.CompositeTransform[itk.D, dimension].New()
    composite.AddTransform(displacement)
    composite.AddTransform(linear)

    moving_itk = itk_image(np.asarray(moving.data), moving)
    reference = itk.ResampleImageFilter.New(
        Input=moving_itk,
        Transform=composite,
        Interpolator=itk.LinearInterpolateImageFunction.New(moving_itk),
        ReferenceImage=itk_image(np.asarray(fixed.data), fixed),
        UseReferenceImage=True,
        DefaultPixelValue=0,
    )
    reference.Update()
    return itk.array_from_image(reference.GetOutput())


def _field(values, dims, spacing, origin):
    return NgffImage(
        data=da.from_array(values),
        dims=("c", *dims),
        scale={"c": 1.0, **dict(zip(dims, spacing))},
        translation={"c": 0.0, **dict(zip(dims, origin))},
        axes_types={"c": "displacement"},
    )


def test_affine_then_displacements_resamples_like_an_itk_composite():
    dims, spacing, origin = ("z", "y", "x"), (2.0, 0.5, 0.5), (-4.0, 1.0, 3.0)
    shape = (12, 40, 36)
    # Rows z, y, x and the translation last: a small shear and a shift of
    # several voxels, so a block reads well outside its own region.
    affine = np.array(
        [[1.0, 0.03, 0.0, 1.5], [-0.02, 1.0, 0.04, -2.0], [0.0, -0.03, 1.0, 2.5]]
    )
    grid = np.meshgrid(*[np.linspace(0, np.pi, n) for n in shape], indexing="ij")
    field_values = np.stack(
        [
            1.5 * np.sin(grid[1]) * np.cos(grid[2]),
            -2.0 * np.sin(grid[0]) * np.sin(grid[2]),
            3.0 * np.cos(grid[0]) * np.sin(grid[1]),
        ]
    )
    moving_values = np.random.default_rng(7).random(shape).astype(np.float32) * 100
    fixed = _image(np.zeros(shape, np.float32), dims, spacing, origin, (6, 16, 16))
    moving = _image(moving_values, dims, spacing, origin, (6, 16, 16))

    result = resample(
        _sequence(affine),
        fixed,
        moving,
        fields={"warp": _field(field_values, dims, spacing, origin)},
    )

    expected = _itk_reference(moving, fixed, field_values, affine)
    np.testing.assert_allclose(np.asarray(result.data), expected, atol=1e-3)
    # The comparison is not satisfied by the affine alone.
    affine_only = resample(Affine(affine=affine.tolist()), fixed, moving)
    assert np.abs(np.asarray(affine_only.data) - expected).max() > 10.0


def test_a_field_stage_reaching_past_the_block_boundary_is_read():
    """The field is zero on the grid's boundary and peaks inside it.

    A block sized by walking its boundary through the sequence would see the
    affine alone and miss what the field displaces onto.
    """
    dims, spacing, origin = ("y", "x"), (1.0, 1.0), (0.0, 0.0)
    field_values = _interior_bump(64, 20, (60.0, -60.0))
    affine = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 50.0]])
    moving_values = np.random.default_rng(3).random((160, 160)).astype(np.float32)
    fixed = _image(np.zeros((64, 64), np.float32), dims, spacing, origin, (64, 64))
    moving = _image(moving_values * 100, dims, spacing, origin, (32, 32))

    result = resample(
        _sequence(affine),
        fixed,
        moving,
        fields={"warp": _field(field_values, dims, spacing, origin)},
    )

    expected = _itk_reference(moving, fixed, field_values, affine)
    np.testing.assert_allclose(np.asarray(result.data), expected, atol=1e-3)


def test_the_bounding_box_of_a_sequence_covers_its_field_stage():
    """``resample_bounding_box`` on the RFC-5 sequence itself, not on ITK's list.

    The boundary walk sees the affine alone where the field is zero; the
    region has to be widened by what the field stage can add, as it is for
    the same stages handed over as an ITK transform list.
    """
    from ngff_zarr import ngff_transform_to_itk_transform, resample_bounding_box

    dims, spacing, origin = ("y", "x"), (1.0, 1.0), (0.0, 0.0)
    field = _field(_interior_bump(64, 20, (60.0, -60.0)), dims, spacing, origin)
    affine = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 50.0]])
    fixed = _image(np.zeros((64, 64), np.float32), dims, spacing, origin, (64, 64))
    moving = _image(np.zeros((160, 160), np.float32), dims, spacing, origin, (32, 32))
    sequence = _sequence(affine)

    region = resample_bounding_box(sequence, fixed, moving, fields={"warp": field})

    itk_list = ngff_transform_to_itk_transform(sequence, dims, fields={"warp": field})
    expected = resample_bounding_box(itk_list, fixed, moving)
    assert region.start_index == expected.start_index
    assert region.size == expected.size
    # The field stage reaches past what the affine alone covers.
    affine_only = resample_bounding_box(Affine(affine=affine.tolist()), fixed, moving)
    assert region.size["y"] > affine_only.size["y"]
