# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""OME-Zarr 0.9.dev1 metadata model: reading, conversion and schema reporting.

Covers the store shapes a 0.9.dev1 reader has to accept (RFC-5 coordinate
systems, a 0.5-style flat axis list, and axes that declare no ``type``), the
conversions to and from every other version, and the axis view the write gate
builds.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import from_ome_zarr
from ngff_zarr._supported_versions import NgffVersion
from ngff_zarr.v04.zarr_metadata import Metadata as Metadata_v04
from ngff_zarr.v05.zarr_metadata import Metadata as Metadata_v05
from ngff_zarr.v06.zarr_metadata import Metadata as Metadata_v06
from ngff_zarr.v09.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Dataset,
    Metadata,
    Scale,
    TransformSequence,
    Translation,
)
from packaging import version

zarr_v3 = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="OME-Zarr 0.9.dev1 is a Zarr v3 hierarchy; zarr version < 3.0.0b1",
)


def _write_v09(root, axes, shape, *, flat=False):
    """Write a single-level 0.9.dev1 store, in either entry shape."""
    group = zarr.open_group(zarr.storage.LocalStore(str(root)), mode="w", zarr_format=3)
    array = group.create_array("0", shape=shape, dtype="uint8", chunks=shape)
    array[...] = np.arange(int(np.prod(shape)), dtype="uint8").reshape(shape)

    scale = {"type": "scale", "scale": [1.0] * len(axes)}
    if flat:
        # 0.5-style: a flat axis list, and dataset transforms that carry no
        # coordinate-system identifiers.
        entry = {
            "axes": axes,
            "datasets": [{"path": "0", "coordinateTransformations": [scale]}],
            "name": "image",
        }
    else:
        entry = {
            "coordinateSystems": [{"name": "intrinsic", "axes": axes}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "sequence",
                            "transformations": [scale],
                            "output": {"name": "intrinsic"},
                        }
                    ],
                }
            ],
            "name": "image",
        }
    group.attrs["ome"] = {"version": "0.9.dev1", "multiscales": [entry]}
    return str(root)


def _axis(name, axis_type="space"):
    return Axis(name=name, type=axis_type)


def _metadata(coordinate_systems):
    output = CoordinateSystemIdentifier(name=coordinate_systems[0].name)
    ndim = len(coordinate_systems[0].axes)
    dataset = Dataset(
        path="0",
        coordinateTransformations=[
            TransformSequence(
                transformations=[
                    Scale(scale=[1.0] * ndim),
                    Translation(translation=[0.0] * ndim),
                ],
                output=output,
            )
        ],
    )
    return Metadata(coordinateSystems=coordinate_systems, datasets=[dataset])


@zarr_v3
def test_read_coordinate_system_shape(tmp_path):
    """The RFC-5 entry shape reads back at 0.9.dev1 with its axes intact."""
    axes = [{"name": n, "type": "space"} for n in "abcdef"]
    root = _write_v09(tmp_path / "cs.ome.zarr", axes, (2,) * 6)

    multiscales = from_ome_zarr(root, validate=False)

    assert type(multiscales.metadata).__module__.endswith("v09.zarr_metadata")
    assert multiscales.metadata.dimension_names == tuple("abcdef")


@zarr_v3
def test_read_flat_axes_shape(tmp_path):
    """A 0.5-style flat axis list is accepted, not only the RFC-5 shape.

    The v0.6 reader dereferences a transform's output coordinate system, which
    this shape does not declare.
    """
    axes = [{"name": n, "type": "space"} for n in "zyx"]
    root = _write_v09(tmp_path / "flat.ome.zarr", axes, (2, 3, 4), flat=True)

    multiscales = from_ome_zarr(root, validate=False)

    assert type(multiscales.metadata).__module__.endswith("v09.zarr_metadata")
    assert multiscales.metadata.dimension_names == ("z", "y", "x")


@zarr_v3
@pytest.mark.parametrize("flat", [False, True], ids=["coordinate-systems", "flat-axes"])
def test_read_axis_without_type(tmp_path, flat):
    """An axis declaring only ``name`` reads back with ``type`` ``None``."""
    axes = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    root = _write_v09(tmp_path / f"untyped-{flat}.ome.zarr", axes, (2, 3, 4), flat=flat)

    multiscales = from_ome_zarr(root, validate=False)

    assert multiscales.metadata.dimension_names == ("a", "b", "c")
    assert [ax.type for ax in multiscales.metadata.axes] == [None, None, None]


@zarr_v3
def test_read_keeps_unknown_axis_keys_out(tmp_path):
    """A schema-legal key the dataclass does not model is dropped, not fatal."""
    axes = [{"name": n, "type": "space", "longName": f"{n} axis"} for n in "zyx"]
    root = _write_v09(tmp_path / "longname.ome.zarr", axes, (2, 3, 4))

    multiscales = from_ome_zarr(root, validate=False)

    assert multiscales.metadata.dimension_names == ("z", "y", "x")


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6", "0.9.dev1"])
def test_to_version_round_trip(version):
    """0.9.dev1 converts to every version and back without losing the axes."""
    metadata = _metadata([CoordinateSystem("intrinsic", [_axis(n) for n in "zyx"])])

    converted = metadata.to_version(version)
    expected = {
        "0.4": Metadata_v04,
        "0.5": Metadata_v05,
        "0.6": Metadata_v06,
        "0.9.dev1": Metadata,
    }[version]
    assert isinstance(converted, expected)

    back = Metadata.from_version(converted)
    assert isinstance(back, Metadata)
    assert back.dimension_names == ("z", "y", "x")


def test_from_version_reinstantiates_axes():
    """Converting up yields v0.9.dev1 axes, not the source version's objects."""
    metadata = _metadata([CoordinateSystem("intrinsic", [_axis(n) for n in "zyx"])])
    downgraded = metadata.to_version("0.4")

    upgraded = Metadata.from_version(downgraded)

    assert all(isinstance(ax, Axis) for ax in upgraded.axes)
    # The source keeps its own axis objects.
    assert upgraded.axes is not downgraded.axes


def test_axis_views_cover_every_coordinate_system():
    """The write gate inspects every coordinate system, not just the intrinsic one.

    ``Metadata`` exposes an ``axes`` property returning only the intrinsic
    system's axes, so a view built from that attribute would miss the others.
    """
    from ngff_zarr.to_ngff_zarr import _axis_views

    metadata = _metadata(
        [
            CoordinateSystem("intrinsic", [_axis(n) for n in "zyx"]),
            CoordinateSystem("other", [_axis(n) for n in "abcdef"]),
        ]
    )

    views = _axis_views(metadata)

    assert len(views) == 2
    assert [len(view.axes) for _, view in views] == [3, 6]


def test_axes_property_is_not_a_field():
    """``axes`` must not be serialized into the multiscales entry."""
    from dataclasses import asdict

    metadata = _metadata([CoordinateSystem("intrinsic", [_axis(n) for n in "zyx"])])

    assert "axes" not in asdict(metadata)
    assert metadata.axes == metadata.coordinateSystems[0].axes


def test_no_bundled_schema_is_reported_explicitly():
    """``load_schema`` names the missing 0.9.dev1 schema instead of failing on I/O."""
    from ngff_zarr.validate import load_schema

    with pytest.raises(ValueError, match="0.9.dev1"):
        load_schema(version="0.9.dev1")


def test_version_is_supported():
    from inspect import signature

    from ngff_zarr import to_ome_zarr
    from ngff_zarr._supported_versions import SUPPORTED_VERSIONS

    assert NgffVersion("0.9.dev1") is NgffVersion.V09dev1
    assert NgffVersion.V09dev1 in SUPPORTED_VERSIONS
    # 0.9.dev1 is opt-in: it must not become the default target. Checked on the
    # writer's own default, which is what decides how a store is written.
    # Pinning ``NgffVersion.LATEST`` to a particular release would fail here on
    # every spec bump for a reason unrelated to 0.9.dev1, and that alias is
    # read by no code in either port.
    assert signature(to_ome_zarr).parameters["version"].default != "0.9.dev1"
    assert NgffVersion.LATEST is not NgffVersion.V09dev1


@pytest.mark.parametrize(
    "version, accepted",
    [("0.4", False), ("0.5", False), ("0.6", False), ("0.9.dev1", True)],
)
def test_validate_structural_is_version_aware(version, accepted):
    """The axis rules are inert only at 0.9.dev1.

    The conformance driver depends on this: validating an RFC-3 dataset at a
    pre-0.9.dev1 version would report every case as a failure.
    """
    from ngff_zarr.structural_validation import ValidationError, validate_structural

    metadata = _metadata(
        [CoordinateSystem("intrinsic", [_axis(n) for n in "abcdef"])]
    ).to_version("0.4")

    if accepted:
        validate_structural(metadata, version=version)
    else:
        with pytest.raises(ValidationError):
            validate_structural(metadata, version=version)
