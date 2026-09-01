# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import asdict

import numpy as np
import pytest
from ngff_zarr import convert_field_block, to_ngff_image

itk = pytest.importorskip("itk")

from ngff_zarr import (  # noqa: E402
    itk_displacement_field_to_ngff_transform,
    ngff_displacement_field_to_itk_transform,
)

CANONICAL = {2: ("y", "x"), 3: ("z", "y", "x")}


def _field_transform(size, spacing, origin, seed=0):
    """A random deformation on a grid of ``size`` (ITK order), as a transform."""
    ndim = len(size)
    field = itk.Image[itk.Vector[itk.D, ndim], ndim].New()
    field.SetRegions(list(size))
    field.Allocate()
    field.SetSpacing(list(spacing))
    field.SetOrigin(list(origin))
    view = itk.array_view_from_image(field)
    view[:] = np.random.default_rng(seed).normal(scale=3.0, size=view.shape)
    transform = itk.DisplacementFieldTransform[itk.D, ndim].New()
    transform.SetDisplacementField(field)
    return transform


def _frame_image(dims, orientations=None):
    image = to_ngff_image(
        np.zeros((4,) * len(dims), dtype=np.float32),
        dims=list(dims),
        scale=dict.fromkeys(dims, 1.0),
        translation=dict.fromkeys(dims, 0.0),
    )
    image.axes_orientations = orientations
    return image


def _blocks(shape, rows):
    for start in range(0, shape[0], rows):
        yield start, min(start + rows, shape[0])


@pytest.mark.parametrize("ndim", [2, 3])
def test_blockwise_equals_the_whole_field_converter(ndim):
    # The contract: converting a block with the BLOCK's own origin equals
    # cutting that block from the converted whole. The reference is the
    # whole-field converter itself, so the two can never drift apart.
    dims = CANONICAL[ndim]
    size = (6, 5, 4)[:ndim][::-1]
    spacing_itk = (1.0, 1.5, 2.0)[:ndim]
    origin_itk = (7.0, -3.0, 5.0)[:ndim]
    transform = _field_transform(size, spacing_itk, origin_itk, seed=3)
    _, whole = itk_displacement_field_to_ngff_transform(transform, dims, path="DVF")
    reference = np.asarray(whole.data)

    field = itk.array_view_from_image(transform.GetDisplacementField())
    itk_block = np.ascontiguousarray(np.moveaxis(field, -1, 0))
    spacing = [float(whole.scale[dim]) for dim in dims]
    translation = [float(whole.translation[dim]) for dim in dims]

    assembled = np.empty_like(reference)
    for start, stop in _blocks(itk_block.shape[1:], rows=2):
        origin = list(translation)
        origin[0] += start * spacing[0]
        assembled[:, start:stop] = convert_field_block(
            itk_block[:, start:stop], dims, translation=origin, spacing=spacing
        )
    np.testing.assert_allclose(assembled, reference, rtol=0, atol=1e-12)


def test_blockwise_inverse_equals_the_reader(tmp_path):
    # The other direction, against the reader converter: a store's values
    # converted block by block are the vectors of the ITK transform the
    # whole-field reader rebuilds.
    from ngff_zarr.v06.zarr_metadata import Displacements

    dims = CANONICAL[3]
    transform = _field_transform((4, 5, 6), (1.0, 1.5, 2.0), (7.0, -3.0, 5.0), seed=5)
    entry, whole = itk_displacement_field_to_ngff_transform(transform, dims, path="DVF")
    stored = np.asarray(whole.data)

    wasm = ngff_displacement_field_to_itk_transform(
        Displacements(path="DVF", interpolation="linear"), whole, dims
    )
    rebuilt = itk.transform_from_dict(asdict(wasm[0]))
    if hasattr(rebuilt, "GetNthTransform"):
        rebuilt = rebuilt.GetNthTransform(0)
    reference = np.moveaxis(
        itk.array_view_from_image(rebuilt.GetDisplacementField()), -1, 0
    )

    spacing = [float(whole.scale[dim]) for dim in dims]
    translation = [float(whole.translation[dim]) for dim in dims]
    assembled = np.empty_like(reference)
    for start, stop in _blocks(stored.shape[1:], rows=2):
        origin = list(translation)
        origin[0] += start * spacing[0]
        assembled[:, start:stop] = convert_field_block(
            stored[:, start:stop],
            dims,
            translation=origin,
            spacing=spacing,
            inverse=True,
        )
    np.testing.assert_allclose(assembled, reference, rtol=0, atol=1e-12)


def test_forward_then_inverse_is_the_identity():
    dims = CANONICAL[3]
    rng = np.random.default_rng(7)
    block = rng.normal(scale=3.0, size=(3, 4, 5, 6))
    geometry = {"translation": [5.0, -3.0, 7.0], "spacing": [2.0, 1.5, 1.0]}
    fixed = _frame_image(dims)
    moving = _frame_image(dims)
    stored = convert_field_block(block, dims, fixed=fixed, moving=moving, **geometry)
    back = convert_field_block(
        stored, dims, fixed=fixed, moving=moving, inverse=True, **geometry
    )
    np.testing.assert_allclose(back, block, rtol=0, atol=1e-12)


def test_without_frames_the_conversion_is_the_component_permutation():
    # No frames, no positional term: the ITK (x, y, z) components land as the
    # (z, y, x) the store's axes name, values untouched. This is the whole of
    # the conversion for a producer whose field and images share one world
    # frame, which is the common registration output.
    dims = CANONICAL[3]
    rng = np.random.default_rng(9)
    block = rng.normal(size=(3, 2, 3, 4))
    stored = convert_field_block(
        block, dims, translation=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(stored, block[::-1])


@pytest.mark.parametrize("dims", [("z", "y", "x"), ("x", "y", "z")])
def test_noncanonical_dims_keep_the_positional_term_on_its_own_axes(dims):
    # The positional frame term is indexed on the canonical grid, so a block
    # whose spatial axes follow a caller's own order must be taken there and
    # put back. Without frames the term is zero and any order passes, which is
    # why this pins the framed case: the axes were off by a permutation, and
    # the values stayed plausible.
    from ngff_zarr import orientation_from_name

    transform = _field_transform((6, 5, 4), (1.0, 1.5, 2.0), (7.0, -3.0, 5.0), seed=3)
    fixed, moving = (
        _frame_image(dims, orientation_from_name(name)) for name in ("LPS", "RAS")
    )
    _, whole = itk_displacement_field_to_ngff_transform(
        transform, dims, path="DVF", fixed=fixed, moving=moving
    )
    reference = np.asarray(whole.data)

    field = itk.array_view_from_image(transform.GetDisplacementField())
    canonical = ("z", "y", "x")
    block = np.moveaxis(field, -1, 0)
    block = np.ascontiguousarray(
        np.transpose(block, [0] + [1 + canonical.index(dim) for dim in dims])
    )
    converted = convert_field_block(
        block,
        dims,
        translation=[float(whole.translation[dim]) for dim in dims],
        spacing=[float(whole.scale[dim]) for dim in dims],
        fixed=fixed,
        moving=moving,
    )
    np.testing.assert_allclose(converted, reference, rtol=0, atol=1e-12)


def test_a_block_shape_that_is_not_the_grid_is_refused():
    # Only the component axis was checked, so a (3, 4) block passed and a
    # coordinates conversion broadcast its one grid index over three axes.
    with pytest.raises(ValueError, match="one component axis"):
        convert_field_block(
            np.zeros((3, 4)),
            CANONICAL[3],
            translation=[0.0, 0.0, 0.0],
            spacing=[1.0, 1.0, 1.0],
        )


def test_geometry_that_does_not_cover_the_axes_is_refused():
    with pytest.raises(ValueError, match="translation and spacing"):
        convert_field_block(
            np.zeros((3, 4, 5, 6)),
            CANONICAL[3],
            translation=[0.0, 0.0],
            spacing=[1.0, 1.0, 1.0],
        )


def test_a_component_count_that_misses_an_axis_is_refused():
    with pytest.raises(ValueError, match="2 components per point"):
        convert_field_block(
            np.zeros((2, 4, 5, 6)),
            CANONICAL[3],
            translation=[0.0, 0.0, 0.0],
            spacing=[1.0, 1.0, 1.0],
        )


def test_an_unknown_transform_type_is_refused():
    with pytest.raises(ValueError, match="transform_type"):
        convert_field_block(
            np.zeros((3, 4, 5, 6)),
            CANONICAL[3],
            translation=[0.0, 0.0, 0.0],
            spacing=[1.0, 1.0, 1.0],
            transform_type="warp",
        )
