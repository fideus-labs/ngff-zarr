# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""OME-Zarr v0.6 displacement / coordinate field axis types.

A displacement (or coordinate) field is an ordinary multiscale image whose
vector-component axis, in the channel position, carries ``type="displacement"``
(or ``type="coordinate"``) instead of ``type="channel"``. The axis type is set
with the ``axes_types`` argument of :class:`NgffImage`.
"""

import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr
from ngff_zarr import NgffImage, to_multiscales
from packaging import version

requires_zarr_v3 = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.6 requires zarr-python >= 3.0.0b1",
)


def _field_image(axes_types=None) -> NgffImage:
    """A 2-component displacement field over a ``yx`` image."""
    data = da.zeros((2, 4, 5), dtype=np.float32)
    return NgffImage(
        data=data,
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": 1.0, "x": 1.0},
        translation={"c": 0.0, "y": 0.0, "x": 0.0},
        axes_types=axes_types,
    )


def _axes(multiscales):
    return multiscales.metadata.intrinsic_coordinate_system.axes


def test_axes_types_sets_displacement():
    multiscales = to_multiscales(_field_image({"c": "displacement"}), scale_factors=[])
    axes = _axes(multiscales)
    assert [a.name for a in axes] == ["c", "y", "x"]
    assert [a.type for a in axes] == ["displacement", "space", "space"]
    assert axes[0].discrete is True
    assert axes[1].discrete is None


def test_axes_types_sets_coordinate():
    multiscales = to_multiscales(_field_image({"c": "coordinate"}), scale_factors=[])
    assert [a.type for a in _axes(multiscales)] == ["coordinate", "space", "space"]


def test_axes_types_default_keeps_channel():
    """Without the override the channel axis keeps its inferred type."""
    multiscales = to_multiscales(_field_image(), scale_factors=[])
    assert _axes(multiscales)[0].type == "channel"


@requires_zarr_v3
def test_displacement_field_roundtrip():
    """The displacement axis type survives a v0.6 write/read round trip."""
    multiscales = to_multiscales(_field_image({"c": "displacement"}), scale_factors=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ngff_zarr(tmpdir, multiscales, version="0.6")
        imported = nz.from_ngff_zarr(tmpdir)

    axes = imported.metadata.intrinsic_coordinate_system.axes
    assert [a.type for a in axes] == ["displacement", "space", "space"]
    assert axes[0].discrete is True


@requires_zarr_v3
def test_post_hoc_axis_type_edit_roundtrip():
    """The component axis type can also be set directly on the metadata."""
    multiscales = to_multiscales(_field_image(), scale_factors=[])
    multiscales.metadata.intrinsic_coordinate_system.axes[0].type = "displacement"
    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ngff_zarr(tmpdir, multiscales, version="0.6")
        imported = nz.from_ngff_zarr(tmpdir)

    axes = imported.metadata.intrinsic_coordinate_system.axes
    assert axes[0].type == "displacement"


@requires_zarr_v3
@pytest.mark.parametrize("transform_type", ["displacements", "coordinates"])
def test_field_transform_roundtrip(transform_type):
    """A displacements/coordinates transform referencing a field round-trips."""
    from ngff_zarr.v06.zarr_metadata import (
        Axis,
        Coordinates,
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Displacements,
    )

    image = NgffImage(
        data=da.zeros((4, 5), dtype=np.float32),
        dims=("y", "x"),
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    multiscales = to_multiscales(image, scale_factors=[])

    cls = Displacements if transform_type == "displacements" else Coordinates
    transform = cls(path="displacement_field", interpolation="linear")
    output_cs = CoordinateSystem(
        name="output",
        axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
    )
    transform.input = CoordinateSystemIdentifier(
        name=multiscales.metadata.intrinsic_coordinate_system.name
    )
    transform.output = CoordinateSystemIdentifier(name=output_cs.name)
    transform.name = f"to_{transform_type}"
    multiscales.metadata.coordinateSystems.append(output_cs)
    multiscales.metadata.coordinateTransformations = [transform]

    with tempfile.TemporaryDirectory() as tmpdir:
        nz.to_ngff_zarr(tmpdir, multiscales, version="0.6")
        imported = nz.from_ngff_zarr(tmpdir)

    out = imported.metadata.coordinateTransformations
    assert len(out) == 1
    assert out[0].type == transform_type
    assert out[0].name == f"to_{transform_type}"
    assert out[0].input == CoordinateSystemIdentifier(name="intrinsic")
    assert out[0].output == CoordinateSystemIdentifier(name="output")
    assert out[0].path == "displacement_field"
    assert out[0].interpolation == "linear"


def test_axes_types_unknown_dim_raises():
    """An axes_types entry naming a missing dimension is rejected."""
    with pytest.raises(ValueError):
        to_multiscales(_field_image({"q": "displacement"}), scale_factors=[])
