# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-3 ("More Dimensions for Thee") end-to-end axis handling.

RFC-3 removes the historical OME-Zarr caps on the number, names, types, and
ordering of axes. These tests exercise the read path and a read/write round-trip
against local, in-memory synthetic fixtures mirroring the official sample
datasets (astronaut_xcy, ecg_1d, ramp_6d, FLIM, EBSD) -- no network access. The
focus is that reading keeps the exact axis order, names and types, and that
writing the result back at 0.9.dev1 -- which runs the write gate -- preserves
all three.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import from_ome_zarr, to_ome_zarr
from packaging import version as packaging_version

#: Writing OME-Zarr 0.5 and later needs a Zarr v3 hierarchy.
needs_zarr_v3 = pytest.mark.skipif(
    packaging_version.parse(zarr.__version__) < packaging_version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)


def _open_group(root):
    """Open a writable v2 group at ``root`` across zarr-python 2 and 3.

    v0.4 OME-Zarr is a Zarr v2 hierarchy, so force ``zarr_format=2`` under
    zarr-python 3 (which defaults to v3).
    """
    if hasattr(zarr.storage, "LocalStore"):  # zarr-python 3
        store = zarr.storage.LocalStore(root)
        return zarr.open_group(store, mode="w", zarr_format=2)
    store = zarr.DirectoryStore(root)  # zarr-python 2
    return zarr.open_group(store, mode="w")


def _ramp(shape):
    """Deterministic uint8 payload for a shape (values wrap modulo 256)."""
    return np.arange(int(np.prod(shape)), dtype="uint8").reshape(shape)


def _axis_pairs(multiscales):
    """The ``(name, type)`` of every axis, read off the metadata as parsed.

    Axis types are declared per axis and are not recoverable from
    :attr:`NgffImage.dims`, so they need their own assertion: a regression that
    coerced or dropped a custom type would leave ``dims`` intact.
    """
    axes = multiscales.metadata.coordinateSystems[0].axes
    return [(axis.name, axis.type) for axis in axes]


def _write_v04(root, axes, shape):
    """Write a minimal single-level v0.4 OME-Zarr with the given axes/shape."""
    group = _open_group(root)
    data = _ramp(shape)
    if hasattr(group, "create_array"):  # zarr-python 3
        array = group.create_array("0", shape=shape, dtype="uint8", chunks=shape)
    else:  # zarr-python 2
        array = group.create_dataset("0", shape=shape, dtype="uint8")
    array[...] = data
    group.attrs["multiscales"] = [
        {
            "axes": axes,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0] * len(axes)}
                    ],
                }
            ],
            "version": "0.4",
        }
    ]
    return str(root)


# (label, axes, shape): one entry per official RFC-3 sample-dataset shape.
_RFC3_SHAPES = [
    # ecg_1d: a single time axis (breaks the historical 2-5 count floor).
    ("ecg_1d", [{"name": "t", "type": "time"}], (16,)),
    # ramp_6d: six axes (exceeds the historical max of 5).
    ("ramp_6d", [{"name": n, "type": "space"} for n in "abcdef"], (2,) * 6),
    # astronaut_xcy: non-TCZYX order (x, c, y).
    (
        "astronaut_xcy",
        [
            {"name": "x", "type": "space"},
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space"},
        ],
        (4, 3, 5),
    ),
    # FLIM-like: multiple axes of the same type (two channel-family axes).
    (
        "flim_multi",
        [
            {"name": "c", "type": "channel"},
            {"name": "tau", "type": "channel"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        (2, 3, 4, 5),
    ),
    # EBSD-like: arbitrary names and custom (non-standard) axis types. The
    # official EBSD sample uses four type:space axes; this variant additionally
    # exercises a custom type string, which RFC-3 permits.
    (
        "ebsd_custom",
        [
            {"name": "sy", "type": "space"},
            {"name": "sx", "type": "space"},
            {"name": "dy", "type": "diffraction"},
            {"name": "dx", "type": "diffraction"},
        ],
        (2, 2, 4, 5),
    ),
]


@pytest.mark.parametrize("label, axes, shape", _RFC3_SHAPES, ids=lambda v: v)
def test_rfc3_read_preserves_axis_order(tmp_path, label, axes, shape):
    """Reading an RFC-3 dataset keeps the exact declared axis order/names."""
    root = _write_v04(tmp_path / f"{label}.ome.zarr", axes, shape)
    multiscales = from_ome_zarr(root, validate=False)

    image = multiscales.images[0]
    dims = tuple(image.dims)
    assert dims == tuple(ax["name"] for ax in axes)
    # The array is read at full dimensionality (not squeezed or transposed) and
    # its payload is intact.
    assert image.data.ndim == len(axes)
    assert image.data.shape == shape
    assert np.array_equal(np.asarray(image.data), _ramp(shape))
    # Types too, including the custom ones RFC-3 permits (ebsd_custom).
    assert _axis_pairs(multiscales) == [(ax["name"], ax["type"]) for ax in axes]


@needs_zarr_v3
@pytest.mark.parametrize("label, axes, shape", _RFC3_SHAPES, ids=lambda v: v)
def test_rfc3_round_trip_preserves_axis_order(tmp_path, label, axes, shape):
    """A read then write then read keeps the axis order and data."""
    src = _write_v04(tmp_path / f"{label}-src.ome.zarr", axes, shape)
    multiscales = from_ome_zarr(src, validate=False)
    dims_in = tuple(multiscales.images[0].dims)
    pairs_in = _axis_pairs(multiscales)

    # 0.9.dev1 is the only version that can express an RFC-3 axis model; writing
    # these shapes at 0.4/0.5/0.6 is refused by the write gate (asserted below).
    out = str(tmp_path / f"{label}-out.ome.zarr")
    to_ome_zarr(out, multiscales, version="0.9.dev1")
    multiscales_out = from_ome_zarr(out, validate=False)
    image_out = multiscales_out.images[0]

    assert tuple(image_out.dims) == dims_in
    assert _axis_pairs(multiscales_out) == pairs_in
    assert np.array_equal(np.asarray(image_out.data), _ramp(shape))


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
@pytest.mark.parametrize("label, axes, shape", _RFC3_SHAPES, ids=lambda v: v)
def test_rfc3_write_is_refused_below_v09dev1(tmp_path, label, axes, shape, version):
    """Serializing an RFC-3 axis model to a pre-0.9.dev1 version is refused."""
    src = _write_v04(tmp_path / f"{label}-{version}-src.ome.zarr", axes, shape)
    multiscales = from_ome_zarr(src, validate=False)

    out = str(tmp_path / f"{label}-{version}-out.ome.zarr")
    with pytest.raises(ValueError, match="Cannot write OME-Zarr"):
        to_ome_zarr(out, multiscales, version=version)


@needs_zarr_v3
@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_channel_last_write_is_refused_below_v09dev1(tmp_path, version):
    """Axis order is a spec MUST, so the writer refuses it, it does not warn.

    ``(z, y, x, c)`` satisfies every other axis rule: 4 axes, one channel, and
    the space names are the ``(z, y, x)`` suffix. Only the class ordering is
    wrong, so this reaches ``validate_axis_order`` and nothing else.
    """
    axes = [{"name": n, "type": "space"} for n in "zyx"]
    axes.append({"name": "c", "type": "channel"})
    src = _write_v04(tmp_path / f"cl-{version}-src.ome.zarr", axes, (2, 3, 4, 2))
    multiscales = from_ome_zarr(src, validate=False)

    out = str(tmp_path / f"cl-{version}-out.ome.zarr")
    with pytest.raises(ValueError, match="axis-order"):
        to_ome_zarr(out, multiscales, version=version)

    # RFC-3 lifts the ordering rule, so the same model writes at 0.9.dev1.
    to_ome_zarr(
        str(tmp_path / f"cl-{version}-ok.ome.zarr"), multiscales, version="0.9.dev1"
    )


@needs_zarr_v3
@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6", "0.9.dev1"])
def test_conventional_axes_write_at_every_version(tmp_path, version):
    """A conventional (z, y, x) image still writes everywhere, 0.9.dev1 included."""
    axes = [{"name": n, "type": "space"} for n in "zyx"]
    src = _write_v04(tmp_path / f"zyx-{version}-src.ome.zarr", axes, (2, 3, 4))
    multiscales = from_ome_zarr(src, validate=False)

    out = str(tmp_path / f"zyx-{version}-out.ome.zarr")
    to_ome_zarr(out, multiscales, version=version)
    image_out = from_ome_zarr(out, validate=False).images[0]
    assert tuple(image_out.dims) == ("z", "y", "x")
    assert np.array_equal(np.asarray(image_out.data), _ramp((2, 3, 4)))


# The gate's message bytes are part of the cross-language contract: these
# literals are pinned identically in ``ts/test/write_gate_test.ts``.
CANONICAL_GATE_MESSAGE_REPEATED_NAME = (
    "Cannot write OME-Zarr version=\"0.9.dev1\": Axis name 'z' is repeated; "
    "axis names must be unique within a dataset. Axes at multiscales[0].axes: "
    "['z'(type='space'), 'z'(type='space'), 'x'(type='space')]."
)

CANONICAL_GATE_MESSAGE_AXIS_COUNT = (
    'Cannot write OME-Zarr version="0.4": this axis model violates that '
    "version's [axis-count] rule. OME-Zarr v0.4, v0.5 and v0.6 require between "
    "2 and 5 axes, inclusive; found 6. Axes at multiscales[0].axes: "
    "['a'(type='space'), 'b'(type='space'), 'c'(type='space'), "
    "'d'(type='space'), 'e'(type='space'), 'f'(type='space')]. Pass "
    'version="0.9.dev1" to write it: 0.9.dev1 is the only OME-Zarr version '
    "that adopts RFC-3 (arbitrary axis count, names, types and ordering)."
)


def _flat_metadata(names):
    """Single-level v0.4 metadata over ``names``, all ``space`` axes."""
    from ngff_zarr.v04.zarr_metadata import Axis, Dataset, Metadata, Scale

    return Metadata(
        axes=[Axis(name=name, type="space") for name in names],
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[Scale([1.0] * len(names))],
            )
        ],
        coordinateTransformations=None,
    )


def test_gate_message_bytes_match_the_typescript_port():
    """The refusal text is part of the contract, not an implementation detail."""
    from ngff_zarr.to_ngff_zarr import _gate_axis_model

    with pytest.raises(ValueError) as exc_info:
        _gate_axis_model(_flat_metadata(["z", "z", "x"]), "0.9.dev1")
    assert str(exc_info.value) == CANONICAL_GATE_MESSAGE_REPEATED_NAME

    with pytest.raises(ValueError) as exc_info:
        _gate_axis_model(_flat_metadata(list("abcdef")), "0.4")
    assert str(exc_info.value) == CANONICAL_GATE_MESSAGE_AXIS_COUNT


@needs_zarr_v3
@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_gate_skips_coordinate_systems_the_target_version_drops(tmp_path, version):
    """A coordinate system the downgrade discards must not block the write.

    ``_gate_axis_model`` runs after ``Metadata.to_version``, so a 0.4/0.5 target
    sees only the surviving system. The TypeScript port reaches the same set by
    checking the target version in ``axisViews``.
    """
    from ngff_zarr.v06.zarr_metadata import Axis as AxisV06
    from ngff_zarr.v06.zarr_metadata import CoordinateSystem

    src = _write_v04(
        tmp_path / "src.ome.zarr",
        [{"name": n, "type": "space"} for n in ("z", "y", "x")],
        (2, 3, 4),
    )
    multiscales = from_ome_zarr(src, validate=False)
    multiscales.metadata.coordinateSystems.append(
        CoordinateSystem(
            name="extra",
            axes=[AxisV06(name=n, type="space") for n in "abcdef"],
        )
    )

    out = str(tmp_path / f"out-{version}.ome.zarr")
    to_ome_zarr(out, multiscales, version=version)
    assert tuple(from_ome_zarr(out, validate=False).images[0].dims) == (
        "z",
        "y",
        "x",
    )

    # 0.6 writes both systems, so there the six-axis one is refused.
    with pytest.raises(ValueError, match="coordinateSystems\\[1\\].axes"):
        to_ome_zarr(str(tmp_path / "out-0.6.ome.zarr"), multiscales, version="0.6")
