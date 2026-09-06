# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The multiscale pipeline normalizes axes to the spec order (t, c, z, y, x)."""

import dask.array
import numpy as np
from ngff_zarr import to_multiscales, to_ngff_image
from ngff_zarr.methods._support import _canonical_axis_order
from ngff_zarr.ngff_image import NgffImage


def test_channel_last_input_is_normalized():
    """A (y, x, c) image comes out as (c, y, x) with matching metadata."""
    data = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
    image = to_ngff_image(data, dims=("y", "x", "c"))

    multiscales = to_multiscales(image, scale_factors=[])

    assert multiscales.images[0].dims == ("c", "y", "x")
    assert multiscales.images[0].data.shape == (3, 32, 64)
    axes = multiscales.metadata.coordinateSystems[0].axes
    assert [ax.name for ax in axes] == ["c", "y", "x"]

    # The transpose is a relabeling, not a data change.
    original = np.moveaxis(data, -1, 0)
    np.testing.assert_array_equal(np.asarray(multiscales.images[0].data), original)


def test_default_dims_inference_is_normalized():
    """The 4-D default dims (z, y, x, c) are normalized to (c, z, y, x)."""
    data = np.zeros((8, 16, 16, 2), dtype=np.uint8)

    multiscales = to_multiscales(data, scale_factors=[])

    assert multiscales.images[0].dims == ("c", "z", "y", "x")
    assert multiscales.images[0].data.shape == (2, 8, 16, 16)


def test_canonical_input_is_unchanged():
    """An image already in spec order passes through untouched."""
    data = dask.array.zeros((2, 8, 16, 16), dtype=np.uint8)
    image = to_ngff_image(data, dims=("c", "z", "y", "x"))

    multiscales = to_multiscales(image, scale_factors=[])

    assert multiscales.images[0].dims == ("c", "z", "y", "x")
    axes = multiscales.metadata.coordinateSystems[0].axes
    assert [ax.name for ax in axes] == ["c", "z", "y", "x"]


def test_scale_and_translation_follow_the_axes():
    """Dim-keyed metadata keeps its values across the reorder."""
    data = np.zeros((32, 64, 3), dtype=np.uint8)
    image = to_ngff_image(
        data, dims=("y", "x", "c"), scale={"y": 2.0, "x": 4.0}, translation={"y": 1.0}
    )

    multiscales = to_multiscales(image, scale_factors=[])

    out = multiscales.images[0]
    assert out.scale["y"] == 2.0
    assert out.scale["x"] == 4.0
    assert out.translation["y"] == 1.0
    sequence = multiscales.metadata.datasets[0].coordinateTransformations[0]
    assert sequence.transformations[0].scale == [1.0, 2.0, 4.0]  # (c, y, x) order


def test_unknown_dims_are_left_untouched():
    """Axis models outside {t, c, z, y, x} have no spec order to normalize to."""
    data = dask.array.zeros((4, 5, 6), dtype=np.uint8)
    image = NgffImage(
        data=data, dims=("b", "a", "x"), scale={"x": 1.0}, translation={"x": 0.0}
    )

    result = _canonical_axis_order(image)

    assert result is image


def test_preserve_keeps_the_order_the_caller_chose():
    """RFC-3 lifts the ordering rule, so an order can be asked for (gh-734)."""
    data = np.random.randint(0, 255, (8, 16, 16, 2), dtype=np.uint8)
    image = to_ngff_image(
        data, dims=("z", "y", "x", "c"), scale={"z": 0.3, "y": 0.3, "x": 0.3}
    )

    multiscales = to_multiscales(image, scale_factors=[2], axis_order="preserve")

    assert multiscales.images[0].dims == ("z", "y", "x", "c")
    assert [axis.name for axis in multiscales.metadata.axes] == ["z", "y", "x", "c"]
    # The pyramid follows the axes it was given: the channel axis is not scaled.
    assert multiscales.images[1].data.shape == (4, 8, 8, 2)


def test_preserve_round_trips_through_a_0_9_dev1_store(tmp_path):
    from ngff_zarr import from_ome_zarr, to_ome_zarr

    source = np.random.randint(0, 255, (8, 16, 16, 2), dtype=np.uint8)
    image = to_ngff_image(
        source, dims=("z", "y", "x", "c"), scale={"z": 0.3, "y": 0.3, "x": 0.3}
    )
    multiscales = to_multiscales(image, scale_factors=[], axis_order="preserve")

    store = tmp_path / "preserved.ome.zarr"
    to_ome_zarr(str(store), multiscales, version="0.9.dev1")

    read = from_ome_zarr(str(store))
    assert tuple(read.images[0].dims) == ("z", "y", "x", "c")
    assert np.array_equal(np.asarray(read.images[0].data), source)


def test_preserve_survives_the_writer_regenerating_a_level(tmp_path):
    """The writer re-derives generated levels; the order must come along."""
    import json

    from ngff_zarr import from_ome_zarr, to_ome_zarr

    source = np.random.randint(0, 255, (8, 16, 16, 2), dtype=np.uint8)
    image = to_ngff_image(
        source, dims=("z", "y", "x", "c"), scale={"z": 0.3, "y": 0.3, "x": 0.3}
    )
    multiscales = to_multiscales(image, scale_factors=[2], axis_order="preserve")

    store = tmp_path / "preserved.ome.zarr"
    to_ome_zarr(str(store), multiscales, version="0.9.dev1")

    level = json.loads((store / "scale1" / "image" / "zarr.json").read_text())
    assert level["shape"] == [4, 8, 8, 2]
    assert level["dimension_names"] == ["z", "y", "x", "c"]
    read = from_ome_zarr(str(store))
    assert tuple(read.images[1].dims) == ("z", "y", "x", "c")
    assert read.images[1].data.shape == (4, 8, 8, 2)


def test_canonical_stays_the_default():
    data = np.zeros((8, 16, 16, 2), dtype=np.uint8)
    image = to_ngff_image(data, dims=("z", "y", "x", "c"))

    assert to_multiscales(image, scale_factors=[]).images[0].dims == (
        "c",
        "z",
        "y",
        "x",
    )


def test_an_unknown_axis_order_is_refused():
    import pytest

    data = np.zeros((8, 16, 16, 2), dtype=np.uint8)
    image = to_ngff_image(data, dims=("z", "y", "x", "c"))

    with pytest.raises(ValueError, match="axis_order must be"):
        to_multiscales(image, scale_factors=[], axis_order="spec")
