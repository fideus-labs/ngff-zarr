# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""ITK displacement fields to and from RFC-5 ``displacements`` transformations.

The anchor for every case is ``itk``'s own ``TransformPoint``: a converted
field is right when the ITK transform rebuilt from it maps the same points to
the same places as the original, on and off the grid. Layout, component order
and grid geometry are pinned separately with a field holding one known vector.
"""

from dataclasses import asdict

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import (
    NgffImage,
    from_ome_zarr,
    itk_displacement_field_to_ngff_transform,
    itk_transform_to_ngff_transform,
    ngff_transform_to_itk_transform,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
)
from ngff_zarr.rfc4 import RAS
from ngff_zarr.v06.zarr_metadata import CoordinateSystemIdentifier, Displacements

itk = pytest.importorskip("itk")

CANONICAL = {2: ("y", "x"), 3: ("z", "y", "x")}


def _field_transform(size, spacing, origin, direction=None, seed=0, value_type=None):
    """A random deformation on a grid of ``size`` (ITK order), as a transform."""
    value_type = itk.D if value_type is None else value_type
    ndim = len(size)
    field = itk.Image[itk.Vector[value_type, ndim], ndim].New()
    field.SetRegions(list(size))
    field.Allocate()
    field.SetSpacing(list(spacing))
    field.SetOrigin(list(origin))
    if direction is not None:
        field.SetDirection(itk.matrix_from_array(np.asarray(direction, dtype=float)))
    view = itk.array_view_from_image(field)
    view[:] = np.random.default_rng(seed).normal(scale=3.0, size=view.shape)
    transform = itk.DisplacementFieldTransform[value_type, ndim].New()
    transform.SetDisplacementField(field)
    return transform


def _native(itkwasm_list):
    """The native ITK transform an ITK-Wasm list rebuilds to."""
    assert len(itkwasm_list) == 1
    rebuilt = itk.transform_from_dict(asdict(itkwasm_list[0]))
    if hasattr(rebuilt, "GetNthTransform"):
        rebuilt = rebuilt.GetNthTransform(0)
    return rebuilt


def _points_inside(transform, count=6, seed=1):
    """Physical points inside the field's extent, mostly off the grid."""
    field = transform.GetDisplacementField()
    size = np.array(itk.size(field))
    origin = np.array(field.GetOrigin())
    spacing = np.array(field.GetSpacing())
    rng = np.random.default_rng(seed)
    fractions = rng.uniform(0.05, 0.95, size=(count, len(size)))
    points = origin + fractions * spacing * (size - 1)
    return [origin, *points]


def _transform_point(transform, point):
    return np.array(transform.TransformPoint([float(value) for value in point]))


@pytest.mark.parametrize(
    "dims",
    # The canonical orders, and one of each dimensionality that reversing
    # would bind to the wrong ITK axis.
    [("y", "x"), ("x", "y"), ("z", "y", "x"), ("x", "z", "y")],
)
def test_round_trip_matches_transform_point(dims):
    ndim = len(dims)
    size = (5, 4, 3)[:ndim]
    spacing = (0.5, 2.0, 1.5)[:ndim]
    origin = (10.0, 20.0, -3.0)[:ndim]
    original = _field_transform(size, spacing, origin)

    transform, field = itk_displacement_field_to_ngff_transform(
        original, dims, path="warp"
    )

    assert isinstance(transform, Displacements)
    assert transform.path == "warp"
    assert transform.interpolation == "linear"
    assert tuple(field.dims) == ("c", *dims)
    assert field.axes_types == {"c": "displacement"}
    itk_order = [dim for dim in ("x", "y", "z") if dim in dims]
    assert field.data.shape == (ndim, *(size[itk_order.index(d)] for d in dims))
    for dim in dims:
        assert field.scale[dim] == spacing[itk_order.index(dim)]
        assert field.translation[dim] == origin[itk_order.index(dim)]

    rebuilt = _native(
        ngff_transform_to_itk_transform(transform, dims, fields={"warp": field})
    )
    for point in _points_inside(original):
        np.testing.assert_allclose(
            _transform_point(rebuilt, point), _transform_point(original, point)
        )


def test_components_follow_dims_order():
    # One vector (dx=7, dy=9) at ITK index x=2, y=1. RFC-5 components follow
    # the axes of dims, so a yx field reads (9, 7) at [y=1, x=2] and an xy
    # field reads (7, 9) at [x=2, y=1].
    original = _field_transform((3, 2), (0.5, 2.0), (10.0, 20.0))
    view = itk.array_view_from_image(original.GetDisplacementField())
    view[:] = 0.0
    view[1, 2, :] = [7.0, 9.0]

    _, yx = itk_displacement_field_to_ngff_transform(original, ("y", "x"), path="w")
    _, xy = itk_displacement_field_to_ngff_transform(original, ("x", "y"), path="w")

    assert yx.data.shape == (2, 2, 3)
    np.testing.assert_array_equal(np.asarray(yx.data)[:, 1, 2], [9.0, 7.0])
    assert yx.scale == {"c": 1.0, "y": 2.0, "x": 0.5}
    assert yx.translation == {"c": 0.0, "y": 20.0, "x": 10.0}
    assert xy.data.shape == (2, 3, 2)
    np.testing.assert_array_equal(np.asarray(xy.data)[:, 2, 1], [7.0, 9.0])
    assert xy.scale == {"c": 1.0, "x": 0.5, "y": 2.0}
    assert np.count_nonzero(np.asarray(yx.data)) == 2


def test_every_input_form_gives_the_same_field():
    import itkwasm

    original = _field_transform((4, 3), (1.0, 1.5), (2.0, -1.0), seed=3)
    image = original.GetDisplacementField()
    image_dict = itk.dict_from_image(image)
    wasm_image = itkwasm.Image(
        **{
            key: image_dict[key]
            for key in (
                "imageType",
                "name",
                "origin",
                "spacing",
                "direction",
                "size",
                "data",
            )
        }
    )
    transform_dict = itk.dict_from_transform(original)
    wasm_transform = itkwasm.Transform(
        transformType=itkwasm.TransformType(
            transformParameterization=itkwasm.TransformParameterizations.DisplacementField,
            parametersValueType=itkwasm.FloatTypes.Float64,
            inputDimension=2,
            outputDimension=2,
        ),
        numberOfFixedParameters=len(transform_dict["fixedParameters"]),
        numberOfParameters=len(transform_dict["parameters"]),
        fixedParameters=np.asarray(transform_dict["fixedParameters"]),
        parameters=np.asarray(transform_dict["parameters"]),
    )

    reference = itk_displacement_field_to_ngff_transform(original, ("y", "x"), path="w")
    for candidate in (image, wasm_image, wasm_transform, [wasm_transform]):
        transform, field = itk_displacement_field_to_ngff_transform(
            candidate, ("y", "x"), path="w"
        )
        assert transform == reference[0]
        assert tuple(field.dims) == tuple(reference[1].dims)
        assert field.scale == reference[1].scale
        assert field.translation == reference[1].translation
        np.testing.assert_array_equal(
            np.asarray(field.data), np.asarray(reference[1].data)
        )


def test_store_round_trip(tmp_path):
    # The documented layout: the field in a subgroup, then the image whose
    # transform points at it. Reading both back and converting reproduces the
    # ITK transform, with the component axis read from the field's metadata.
    size, spacing, origin = (5, 4, 3), (0.5, 2.0, 1.5), (10.0, 20.0, -3.0)
    dims = CANONICAL[3]
    original = _field_transform(size, spacing, origin, seed=5)
    transform, field = itk_displacement_field_to_ngff_transform(
        original, dims, path="displacement_field"
    )
    image = to_ngff_image(np.zeros(size[::-1], dtype=np.float32), dims=list(dims))
    multiscales = to_multiscales(image, scale_factors=[])
    intrinsic = multiscales.metadata.intrinsic_coordinate_system.name
    transform.input = CoordinateSystemIdentifier(name=intrinsic)
    transform.output = CoordinateSystemIdentifier(name=intrinsic)
    multiscales.metadata.coordinateTransformations = [transform]

    store = tmp_path / "warped.ome.zarr"
    to_ome_zarr(
        str(store / transform.path),
        to_multiscales(field, scale_factors=[]),
        version="0.6",
    )
    to_ome_zarr(str(store), multiscales, version="0.6", overwrite=False)

    imported = from_ome_zarr(str(store))
    read_transform = imported.metadata.coordinateTransformations[0]
    assert isinstance(read_transform, Displacements)
    read_field = from_ome_zarr(str(store / read_transform.path))
    axes = read_field.metadata.intrinsic_coordinate_system.axes
    assert [axis.type for axis in axes] == ["displacement", "space", "space", "space"]

    rebuilt = _native(
        ngff_transform_to_itk_transform(
            read_transform, dims, fields={read_transform.path: read_field}
        )
    )
    for point in _points_inside(original):
        np.testing.assert_allclose(
            _transform_point(rebuilt, point), _transform_point(original, point)
        )


def _frame_image(size, spacing, origin, orientations):
    """A geometry-only image on the grid the field is sampled on."""
    dims = CANONICAL[len(size)]
    itk_order = [dim for dim in ("x", "y", "z") if dim in dims]
    return NgffImage(
        data=da.zeros(size[::-1], dtype=np.uint8),
        dims=list(dims),
        scale={dim: spacing[itk_order.index(dim)] for dim in dims},
        translation={dim: origin[itk_order.index(dim)] for dim in dims},
        axes_orientations=orientations,
    )


def _phi(image, point_itk, itk_dims):
    """An image's intrinsic point to ITK physical space, ``D (q - o) + o``."""
    from ngff_zarr.itk_transform_resample_bounding_box import _itk_direction

    direction = _itk_direction(image, itk_dims)
    origin = np.array([image.translation[dim] for dim in itk_dims])
    return direction @ (point_itk - origin) + origin


@pytest.mark.parametrize("moving_orientation", ["same", "none"])
def test_frames_are_applied_point_by_point(moving_orientation):
    # phi_out(q + d(q)) must equal T(phi_in(q)) at every grid point q, whether
    # the two images share a frame (d = D^-1 v) or not (the general formula).
    size, spacing, origin = (4, 3, 5), (1.0, 2.0, 0.5), (5.0, -2.0, 8.0)
    dims = CANONICAL[3]
    itk_dims = ["x", "y", "z"]
    fixed = _frame_image(size, spacing, origin, RAS)
    moving = _frame_image(
        size, spacing, (1.0, 1.0, 1.0), RAS if moving_orientation == "same" else None
    )
    from ngff_zarr.itk_transform_resample_bounding_box import _itk_direction

    direction_in = _itk_direction(fixed, itk_dims)
    assert not np.allclose(direction_in, np.eye(3))
    original = _field_transform(size, spacing, origin, direction=direction_in, seed=7)

    transform, field = itk_displacement_field_to_ngff_transform(
        original, dims, path="warp", fixed=fixed, moving=moving
    )
    assert field.axes_orientations == RAS

    data = np.asarray(field.data)
    for index in [(0, 0, 0), (1, 2, 3), (4, 1, 0), (2, 2, 2)]:
        q = np.array(
            [
                field.translation[dim] + field.scale[dim] * i
                for dim, i in zip(dims, index)
            ]
        )
        d = data[(slice(None), *index)]
        q_itk, d_itk = q[::-1], d[::-1]
        expected = _transform_point(original, _phi(fixed, q_itk, itk_dims))
        np.testing.assert_allclose(_phi(moving, q_itk + d_itk, itk_dims), expected)

    rebuilt = _native(
        ngff_transform_to_itk_transform(
            transform, dims, fields={"warp": field}, fixed=fixed, moving=moving
        )
    )
    for point in _points_inside(original):
        np.testing.assert_allclose(
            _transform_point(rebuilt, point), _transform_point(original, point)
        )


def test_a_non_spatial_axis_in_dims_is_ignored():
    """``dims`` is usually the image's own dimension names, channel axis and
    all. ITK has no non-spatial axis, so the linear path drops them; the field
    path drops them the same way rather than refuse the caller's ``dims``."""
    original = _field_transform((3, 2), (1.0, 1.5), (0.0, -2.0))
    transform, field = itk_displacement_field_to_ngff_transform(
        original, ("y", "x"), path="w"
    )

    entry = ngff_transform_to_itk_transform(
        transform, ("c", "y", "x"), fields={"w": field}
    )

    assert entry[0].transformType.inputDimension == 2
    rebuilt = _native(entry)
    for point in _points_inside(original):
        np.testing.assert_allclose(
            _transform_point(rebuilt, point), _transform_point(original, point)
        )


def test_an_oriented_field_needs_its_images_to_reach_physical_space():
    """Going back to ITK without them would silently drop the orientation.

    The message names the way through for a bounding box, whose RFC-5 branch
    works on the intrinsic systems and cannot take the pair.
    """
    size, spacing, origin = (4, 3, 5), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    fixed = _frame_image(size, spacing, origin, RAS)
    from ngff_zarr.itk_transform_resample_bounding_box import _itk_direction

    original = _field_transform(
        size, spacing, origin, direction=_itk_direction(fixed, ["x", "y", "z"])
    )
    transform, field = itk_displacement_field_to_ngff_transform(
        original, CANONICAL[3], path="w", fixed=fixed, moving=fixed
    )
    assert field.axes_orientations == RAS

    with pytest.raises(ValueError, match="itk_transform_resample_bounding_box"):
        ngff_transform_to_itk_transform(transform, CANONICAL[3], fields={"w": field})


def test_grid_direction_must_match_the_fixed_image():
    size, spacing, origin = (4, 3, 5), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    flipped = _field_transform(
        size, spacing, origin, direction=np.diag([-1.0, -1.0, 1.0])
    )

    with pytest.raises(ValueError, match="non-identity direction"):
        itk_displacement_field_to_ngff_transform(flipped, CANONICAL[3], path="w")

    fixed = _frame_image(size, spacing, origin, None)
    with pytest.raises(ValueError, match="not oriented like the fixed image"):
        itk_displacement_field_to_ngff_transform(
            flipped, CANONICAL[3], path="w", fixed=fixed, moving=fixed
        )


def test_refusals():
    original = _field_transform((3, 2), (1.0, 1.0), (0.0, 0.0))
    transform, field = itk_displacement_field_to_ngff_transform(
        original, ("y", "x"), path="w"
    )

    with pytest.raises(ValueError, match="spatial axes only"):
        itk_displacement_field_to_ngff_transform(original, ("c", "y", "x"), path="w")
    with pytest.raises(ValueError, match="name 3 axes"):
        itk_displacement_field_to_ngff_transform(original, ("z", "y", "x"), path="w")
    with pytest.raises(ValueError, match="pass both fixed and moving"):
        itk_displacement_field_to_ngff_transform(
            original, ("y", "x"), path="w", fixed=field
        )
    with pytest.raises(ValueError, match="transform list of 2 entries"):
        itk_displacement_field_to_ngff_transform(
            [original, original], ("y", "x"), path="w"
        )
    with pytest.raises(
        NotImplementedError, match="itk_displacement_field_to_ngff_transform"
    ):
        itk_transform_to_ngff_transform(original, ("y", "x"))
    with pytest.raises(ValueError, match="fields="):
        ngff_transform_to_itk_transform(transform, ("y", "x"))
    with pytest.raises(ValueError, match="component axis first"):
        ngff_transform_to_itk_transform(transform, ("x", "y"), fields={"w": field})
    untyped = NgffImage(
        data=field.data,
        dims=field.dims,
        scale=field.scale,
        translation=field.translation,
    )
    with pytest.raises(ValueError, match="exactly one axis of type 'displacement'"):
        ngff_transform_to_itk_transform(transform, ("y", "x"), fields={"w": untyped})


def test_interpolation_other_than_linear_warns():
    original = _field_transform((3, 2), (1.0, 1.0), (0.0, 0.0))
    _, field = itk_displacement_field_to_ngff_transform(original, ("y", "x"), path="w")
    nearest = Displacements(path="w", interpolation="nearest")

    with pytest.warns(UserWarning, match="'nearest' interpolation"):
        rebuilt = _native(
            ngff_transform_to_itk_transform(nearest, ("y", "x"), fields={"w": field})
        )
    np.testing.assert_allclose(
        _transform_point(rebuilt, (1.0, 1.0)), _transform_point(original, (1.0, 1.0))
    )


def test_single_precision_is_preserved():
    from itkwasm import FloatTypes

    original = _field_transform((3, 2), (1.0, 1.0), (0.0, 0.0), value_type=itk.F)
    _, field = itk_displacement_field_to_ngff_transform(original, ("y", "x"), path="w")
    assert field.data.dtype == np.float32

    (entry,) = ngff_transform_to_itk_transform(
        Displacements(path="w", interpolation="linear"), ("y", "x"), fields={"w": field}
    )
    assert entry.transformType.parametersValueType == FloatTypes.Float32
    assert entry.parameters.dtype == np.float32
