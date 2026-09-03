# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import asdict

import numpy as np
import pytest
from ngff_zarr import (
    declare_field_transform,
    from_ome_zarr,
    ngff_displacement_field_to_itk_transform,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
)


def _field_multiscales(seed=0, shape=(3, 4, 5, 6)):
    """A standalone displacement-field multiscales: component axis typed, spec
    component order (the i-th component displaces the i-th spatial axis)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=3.0, size=shape).astype(np.float64)
    image = to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1.0, "z": 2.0, "y": 1.5, "x": 1.0},
        translation={"c": 0.0, "z": 5.0, "y": -3.0, "x": 7.0},
    )
    image.axes_types = {"c": "displacement"}
    return to_multiscales(image, scale_factors=[], cache=False)


def test_a_standalone_declaration_round_trips(tmp_path):
    # The default arguments are the standalone store: the transform on the
    # field's own multiscales, a spatial system declared from its own axes,
    # mapped onto itself. Written at 0.6 and read back, the declaration is
    # what a reader finds -- not an out-of-band convention.
    multiscales = declare_field_transform(_field_multiscales())
    store = tmp_path / "field.ome.zarr"
    to_ome_zarr(store, multiscales, version="0.6")

    back = from_ome_zarr(store)
    entries = back.metadata.coordinateTransformations or []
    assert [entry.type for entry in entries] == ["displacements"]
    assert entries[0].input.name == "physical"
    assert entries[0].output.name == "physical"
    assert entries[0].path == back.metadata.datasets[0].path
    assert "physical" in {system.name for system in back.metadata.coordinateSystems}
    assert back.images[0].axes_types == {"c": "displacement"}


def test_the_declared_store_rebuilds_the_itk_field_it_encodes(tmp_path):
    # The point of declaring: this package's own reader can APPLY the store
    # with no other information. The rebuilt native ITK transform must move
    # points exactly as the field's values say -- component order and grid
    # geometry both, because either mistake still yields a plausible field.
    itk = pytest.importorskip("itk")

    multiscales = declare_field_transform(_field_multiscales())
    store = tmp_path / "field.ome.zarr"
    to_ome_zarr(store, multiscales, version="0.6")

    back = from_ome_zarr(store)
    wasm = ngff_displacement_field_to_itk_transform(
        back.metadata.coordinateTransformations[0], back, ["z", "y", "x"]
    )
    rebuilt = itk.transform_from_dict(asdict(wasm[0]))
    if hasattr(rebuilt, "GetNthTransform"):
        rebuilt = rebuilt.GetNthTransform(0)

    field = np.asarray(multiscales.images[0].data)
    # The grid: index (k, j, i) sits at origin + index * spacing, and the
    # field's component m displaces spatial axis m (z, y, x). ITK speaks
    # (x, y, z), so the expectation reverses -- independently of the module.
    spacing = np.array([2.0, 1.5, 1.0])
    origin = np.array([5.0, -3.0, 7.0])
    for index in ((0, 0, 0), (1, 2, 3), (3, 4, 5)):
        point_zyx = origin + np.array(index) * spacing
        displaced_zyx = point_zyx + field[(slice(None), *index)]
        point_xyz = [float(value) for value in point_zyx[::-1]]
        expected_xyz = displaced_zyx[::-1]
        moved = np.array(rebuilt.TransformPoint(point_xyz))
        np.testing.assert_allclose(moved, expected_xyz, rtol=0, atol=1e-12)


def test_declaring_appends_to_existing_transformations(tmp_path):
    multiscales = declare_field_transform(_field_multiscales())
    again = declare_field_transform(multiscales, interpolation="nearest")
    entries = again.metadata.coordinateTransformations
    assert [entry.interpolation for entry in entries] == ["linear", "nearest"]
    # And the spatial system was declared once, not once per call.
    names = [system.name for system in again.metadata.coordinateSystems]
    assert names.count("physical") == 1


def test_the_argument_is_not_mutated():
    multiscales = _field_multiscales()
    declare_field_transform(multiscales)
    assert not multiscales.metadata.coordinateTransformations


def test_a_named_system_that_is_not_declared_is_refused():
    with pytest.raises(ValueError, match="names no declared coordinate system"):
        declare_field_transform(
            _field_multiscales(), input_system="fixed", output_system="moving"
        )


def test_a_path_without_named_systems_is_refused():
    # With path the field is elsewhere: there are no axes here to derive a
    # system from, and guessing one from the image's own axes would declare a
    # transform over the wrong space.
    with pytest.raises(ValueError, match="pass input_system and output_system"):
        declare_field_transform(_field_multiscales(), path="displacements/field")


def test_one_sided_system_naming_is_refused():
    with pytest.raises(ValueError, match="both input_system and output_system"):
        declare_field_transform(_field_multiscales(), input_system="fixed")


def test_a_field_without_a_typed_component_axis_is_refused():
    multiscales = _field_multiscales()
    multiscales.images[0].axes_types = None
    with pytest.raises(ValueError, match="exactly one axis of type"):
        declare_field_transform(multiscales)


def test_a_non_spatial_axis_is_refused():
    # A field transform is spatial: relabelling a time axis "space" to fit it
    # into a coordinate system declares a transform the package's own
    # converter then refuses to apply.
    multiscales = _field_multiscales(shape=(3, 2, 5, 6))
    image = multiscales.images[0]
    image.dims = ("c", "t", "y", "x")
    image.scale = {"c": 1.0, "t": 1.0, "y": 1.0, "x": 1.0}
    image.translation = {"c": 0.0, "t": 0.0, "y": 0.0, "x": 0.0}
    with pytest.raises(ValueError, match="spatial axes only"):
        declare_field_transform(multiscales)


def test_a_component_count_that_misses_an_axis_is_refused():
    multiscales = _field_multiscales(shape=(2, 4, 5, 6))
    with pytest.raises(ValueError, match="2 components per point"):
        declare_field_transform(multiscales)


def test_a_declaration_beside_the_image_references_its_systems():
    # The entry lives on the IMAGE's multiscales and points at the field's
    # array elsewhere in the store; the field's shape cannot be checked from
    # here, the references can and are.
    from ngff_zarr import CoordinateSystem
    from ngff_zarr.v06.zarr_metadata import Axis

    image = to_ngff_image(
        np.zeros((4, 5, 6), dtype=np.float32),
        dims=["z", "y", "x"],
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    )
    multiscales = to_multiscales(image, scale_factors=[], cache=False)
    axes = [Axis(name=name, type="space", unit=None) for name in ("z", "y", "x")]
    multiscales.metadata.coordinateSystems = [
        *multiscales.metadata.coordinateSystems,
        CoordinateSystem(name="fixed", axes=axes),
        CoordinateSystem(name="moving", axes=axes),
    ]

    declared = declare_field_transform(
        multiscales,
        path="displacements/field",
        input_system="fixed",
        output_system="moving",
    )
    entry = declared.metadata.coordinateTransformations[-1]
    assert entry.path == "displacements/field"
    assert entry.input.name == "fixed"
    assert entry.output.name == "moving"


def test_a_declared_store_streams_region_by_region(tmp_path):
    # The recipe the declaration exists for: a field too large to assemble is
    # declared first, its arrays are created with metadata_only, and the
    # regions land through open_array -- the store is identical to the one a
    # whole-array write produces.
    from ngff_zarr import open_array

    multiscales = declare_field_transform(_field_multiscales(seed=7))
    values = np.asarray(multiscales.images[0].data)

    whole = tmp_path / "whole.ome.zarr"
    to_ome_zarr(whole, multiscales, version="0.6")

    streamed = tmp_path / "streamed.ome.zarr"
    to_ome_zarr(streamed, multiscales, version="0.6", metadata_only=True)
    array = open_array(streamed, multiscales.metadata.datasets[0].path)
    for start in range(0, values.shape[1], 2):
        array[:, start : start + 2] = values[:, start : start + 2]

    for store in (whole, streamed):
        back = from_ome_zarr(store)
        assert [entry.type for entry in back.metadata.coordinateTransformations] == [
            "displacements"
        ]
        np.testing.assert_array_equal(np.asarray(back.images[0].data), values)


def test_a_declared_system_that_is_not_spatial_is_refused():
    # The names line up, so the arity check passes; the types do not. A
    # transform declared over that system is one this package's own converter
    # then refuses to apply, which is the same hole the image's own axes are
    # already checked for.
    from ngff_zarr import CoordinateSystem
    from ngff_zarr.v06.zarr_metadata import Axis

    multiscales = _field_multiscales()
    axes = [
        Axis(name="z", type="time", unit=None),
        Axis(name="y", type="space", unit=None),
        Axis(name="x", type="space", unit=None),
    ]
    multiscales.metadata.coordinateSystems = [
        *multiscales.metadata.coordinateSystems,
        CoordinateSystem(name="physical", axes=axes),
    ]

    with pytest.raises(ValueError, match="spatial axes only"):
        declare_field_transform(multiscales)


def test_named_systems_of_different_dimension_are_refused():
    # With path the field's shape is elsewhere and cannot be checked, but the
    # two systems can be checked against each other: this package's own reader
    # builds an ITK transform with one dimension for both ends, so a 2D input
    # against a 3D output is a declaration nothing here can apply.
    from ngff_zarr import CoordinateSystem
    from ngff_zarr.v06.zarr_metadata import Axis

    def system(name, dims):
        return CoordinateSystem(
            name=name, axes=[Axis(name=d, type="space", unit=None) for d in dims]
        )

    multiscales = _field_multiscales()
    multiscales.metadata.coordinateSystems = [
        *multiscales.metadata.coordinateSystems,
        system("flat", ("y", "x")),
        system("volume", ("z", "y", "x")),
    ]

    with pytest.raises(ValueError, match="same dimension"):
        declare_field_transform(
            multiscales,
            path="displacements/field",
            input_system="flat",
            output_system="volume",
        )
