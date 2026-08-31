# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Resampling through a field that is read a window at a time.

The anchor everywhere is a single undecomposed ITK call with the whole field
converted: the streamed result has to be that result, pixel for pixel. What
the streaming changes is what has to be in memory to reach it, and what a
region of the grid is allowed to displace onto.
"""

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import (
    NgffImage,
    ngff_transform_to_itk_transform,
    resample,
    to_multiscales,
)
from ngff_zarr.displacement_field_transform import field_displacement_bound
from ngff_zarr.v06.zarr_metadata import Coordinates, Displacements

from .test_resample import _whole_image_reference
from .test_resample_bounding_box import _interior_bump

pytest.importorskip("itkwasm_downsample")


def _grid(dims, shape, chunks=None, scale=1.0, translation=0.0, seed=None):
    if seed is None:
        array = np.zeros(shape, dtype=np.float32)
    else:
        array = (np.random.default_rng(seed).random(shape) * 100.0).astype(np.float32)
    return NgffImage(
        data=da.from_array(array, chunks=chunks or shape),
        dims=tuple(dims),
        scale=dict.fromkeys(dims, scale),
        translation=dict.fromkeys(dims, translation),
    )


def _field(values, dims, chunks, absolute=False):
    """A field image over ``dims``, its component axis first."""
    component = "coordinate" if absolute else "displacement"
    return NgffImage(
        data=da.from_array(values, chunks=chunks),
        dims=("c", *dims),
        scale=dict.fromkeys(("c", *dims), 1.0),
        translation=dict.fromkeys(("c", *dims), 0.0),
        axes_types={"c": component},
    )


def _bump(extent, radius, peaks):
    """An interior bump, checked to be exactly zero on the grid boundary.

    The boundary walk the pipeline runs sees only that zero, which is what
    makes this the case a region has to be widened for.
    """
    values = _interior_bump(extent, radius, peaks)
    for axis in range(1, values.ndim):
        for edge in (0, -1):
            assert float(np.abs(values.take(edge, axis=axis)).max()) == 0.0
    return values


def _as_itk(transform, field, dims):
    return ngff_transform_to_itk_transform(transform, dims, fields={"warp": field})


def test_a_field_resamples_like_one_whole_image_call():
    """The streamed result is the undecomposed result, pixel for pixel."""
    moving = _grid("yx", (96, 96), chunks=(16, 16), seed=0)
    fixed = _grid("yx", (64, 64), chunks=(16, 16))
    values = (np.random.default_rng(1).random((2, 64, 64)) * 6.0 - 3.0).astype(
        np.float32
    )
    field = _field(values, ("y", "x"), chunks=(2, 16, 16))
    transform = Displacements(path="warp")

    result = np.asarray(resample(transform, fixed, moving, fields={"warp": field}).data)

    expected = _whole_image_reference(
        _as_itk(transform, field, ("y", "x")), fixed, moving
    )
    np.testing.assert_array_equal(result, expected)


def test_a_bump_inside_the_grid_is_not_missed():
    """A displacement the grid's boundary does not show still gets its pixels.

    The pipeline sizes a region by walking the boundary of the transformed
    grid, so a bump strictly inside it left the region unchanged and the
    pixels it displaces onto unread: they came back as the default value.
    """
    moving = _grid("yx", (160, 160), chunks=(32, 32), seed=0)
    fixed = _grid("yx", (64, 64), chunks=(16, 16))
    field = _field(_bump(64, 20, (24.0, -24.0)), ("y", "x"), chunks=(2, 16, 16))
    transform = Displacements(path="warp")

    result = np.asarray(resample(transform, fixed, moving, fields={"warp": field}).data)

    expected = _whole_image_reference(
        _as_itk(transform, field, ("y", "x")), fixed, moving
    )
    np.testing.assert_array_equal(result, expected)
    # The bump reaches pixels the unwidened region excluded, so this is only
    # a real test while those pixels carry something.
    assert float(np.abs(result[24:40, 24:40]).max()) > 0.0


def test_the_field_is_read_a_window_at_a_time():
    """No block depends on more of the field than its own window covers."""
    moving = _grid("yx", (64, 64), chunks=(16, 16))
    fixed = _grid("yx", (64, 64), chunks=(16, 16))
    field = _field(
        np.zeros((2, 64, 64), dtype=np.float32), ("y", "x"), chunks=(2, 16, 16)
    )

    result = resample(Displacements(path="warp"), fixed, moving, fields={"warp": field})

    graph = dict(result.data.__dask_graph__())
    per_block = [
        len(task[1])
        for key, task in graph.items()
        if isinstance(key, tuple) and str(key[0]).endswith("-field")
    ]
    assert per_block, "no per-block field task was built"
    # 16 chunks in the field; a block covers one and its neighbours.
    assert max(per_block) <= 9


def test_a_coordinates_field_resamples_like_the_displacements_it_equals():
    moving = _grid("yx", (96, 96), chunks=(16, 16), seed=0)
    fixed = _grid("yx", (64, 64), chunks=(16, 16))
    values = (np.random.default_rng(2).random((2, 64, 64)) * 6.0 - 3.0).astype(
        np.float32
    )
    grid = np.mgrid[0:64, 0:64].astype(np.float32)
    displacements = _field(values, ("y", "x"), chunks=(2, 16, 16))
    coordinates = _field(values + grid, ("y", "x"), chunks=(2, 16, 16), absolute=True)

    relative = resample(
        Displacements(path="warp"), fixed, moving, fields={"warp": displacements}
    )
    absolute = resample(
        Coordinates(path="warp"), fixed, moving, fields={"warp": coordinates}
    )

    np.testing.assert_allclose(
        np.asarray(absolute.data), np.asarray(relative.data), rtol=1e-5, atol=1e-3
    )


def test_a_field_read_from_a_multiscales_streams(tmp_path):
    """A field whose component axis is typed in the metadata, not on the image."""
    from ngff_zarr import from_ome_zarr, to_ome_zarr

    values = (np.random.default_rng(3).random((2, 64, 64)) * 6.0 - 3.0).astype(
        np.float32
    )
    path = str(tmp_path / "warp.ome.zarr")
    to_ome_zarr(
        path,
        to_multiscales(
            _field(values, ("y", "x"), chunks=(2, 16, 16)),
            scale_factors=[],
            chunks=(2, 16, 16),
        ),
        version="0.6",
    )
    stored = from_ome_zarr(path)

    moving = _grid("yx", (96, 96), chunks=(16, 16), seed=0)
    fixed = _grid("yx", (64, 64), chunks=(16, 16))
    transform = Displacements(path="warp")

    result = resample(transform, fixed, moving, fields={"warp": stored})

    expected = _whole_image_reference(
        _as_itk(transform, stored, ("y", "x")), fixed, moving
    )
    np.testing.assert_array_equal(np.asarray(result.data), expected)


def test_a_grid_beyond_the_field_takes_the_identity():
    """Outside the field ITK displaces nothing, and so does an empty window."""
    moving = _grid("yx", (96, 96), chunks=(16, 16), seed=0)
    fixed = NgffImage(
        data=da.zeros((16, 16), chunks=(16, 16), dtype=np.float32),
        dims=("y", "x"),
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 70.0, "x": 70.0},
    )
    field = _field(
        np.full((2, 32, 32), 5.0, dtype=np.float32), ("y", "x"), chunks=(2, 16, 16)
    )
    transform = Displacements(path="warp")

    result = resample(transform, fixed, moving, fields={"warp": field})

    expected = _whole_image_reference(
        _as_itk(transform, field, ("y", "x")), fixed, moving
    )
    np.testing.assert_array_equal(np.asarray(result.data), expected)


def test_the_bound_pass_reads_only_the_chunks_the_grid_touches():
    """A grid smaller than the field it is defined on pays for its own part."""
    read = []

    def counting(block, block_info=None):
        read.append(block_info[0]["chunk-location"])
        return block

    raw = da.zeros((2, 256, 256), chunks=(2, 32, 32), dtype=np.float32)
    field = NgffImage(
        data=raw.map_blocks(counting, dtype=np.float32),
        dims=("c", "y", "x"),
        scale=dict.fromkeys(("c", "y", "x"), 1.0),
        translation=dict.fromkeys(("c", "y", "x"), 0.0),
        axes_types={"c": "displacement"},
    )
    moving = _grid("yx", (256, 256), chunks=(32, 32))
    fixed = _grid("yx", (32, 32), chunks=(16, 16))

    resample(Displacements(path="warp"), fixed, moving, fields={"warp": field})

    assert int(np.prod([len(sizes) for sizes in raw.chunks])) == 64
    assert len(set(read)) == 4


@pytest.mark.parametrize("absolute", [False, True])
def test_a_restricted_bound_is_the_bound_over_the_same_window(absolute):
    """Reading fewer chunks has to give the same range, and a real range.

    Random windows against uneven chunks, since the pass cuts on chunk
    borders and an off-border window is where an index offset goes wrong.
    """
    rng = np.random.default_rng(0)
    for _ in range(12):
        extent = int(rng.integers(20, 70))
        borders = np.linspace(0, extent, int(rng.integers(3, 7))).astype(int)
        chunks = tuple(int(size) for size in np.diff(borders) if size > 0)
        values = (rng.random((2, extent, extent)) * 20 - 10).astype(np.float32)
        stored = values
        if absolute:
            stored = values + np.mgrid[0:extent, 0:extent].astype(np.float32)
        field = NgffImage(
            data=da.from_array(stored, chunks=(2, chunks, chunks)),
            dims=("c", "y", "x"),
            scale=dict.fromkeys(("c", "y", "x"), 1.0),
            translation=dict.fromkeys(("c", "y", "x"), 0.0),
            axes_types={"c": "coordinate" if absolute else "displacement"},
        )
        transform = Coordinates(path="w") if absolute else Displacements(path="w")
        window = tuple(
            (int(low), int(rng.integers(low + 1, extent + 1)))
            for low in rng.integers(0, extent, size=2)
        )

        restricted = field_displacement_bound(transform, field, ("y", "x"), window)
        whole = field_displacement_bound(transform, field, ("y", "x"))

        ranges = restricted.over(window)
        assert ranges.keys() == whole.over(window).keys()
        for dim, (low, high) in ranges.items():
            reference = whole.over(window)[dim]
            np.testing.assert_allclose((low, high), reference, atol=1e-4)
        inside = values[
            :, window[0][0] : window[0][1], window[1][0] : window[1][1]
        ].reshape(2, -1)
        for axis, dim in enumerate(("y", "x")):
            low, high = ranges[dim]
            assert low <= inside[axis].min() + 1e-4
            assert high >= inside[axis].max() - 1e-4


def test_a_grid_reaching_past_the_field_keeps_the_points_it_displaces_nothing():
    """A grid half on the field and half off it reads both halves.

    ITK displaces a point beyond the field by nothing, so zero belongs in the
    range as much as the values do. A field that displaces every point it
    covers the same way otherwise carries the region off the points it does
    not cover, and those come back as the default value.
    """
    moving = _grid("yx", (160, 160), chunks=(32, 32), seed=0)
    fixed = _grid("yx", (64, 64), chunks=(64, 64))
    field = _field(
        np.full((2, 32, 32), -5.0, dtype=np.float32), ("y", "x"), chunks=(2, 16, 16)
    )
    transform = Displacements(path="warp")

    result = np.asarray(resample(transform, fixed, moving, fields={"warp": field}).data)

    expected = _whole_image_reference(
        _as_itk(transform, field, ("y", "x")), fixed, moving
    )
    np.testing.assert_array_equal(result, expected)
    # The half beyond the field is the half at stake, so it has to carry
    # something for this to be a test.
    assert float(np.abs(result[32:, 32:]).max()) > 0.0
