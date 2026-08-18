# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for block-wise resampling onto a fixed grid."""

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import (
    NgffImage,
    itk_transform_resample,
    itk_transform_resample_bounding_box,
)
from ngff_zarr.itk_transform_resample import _INTERPOLATOR_PADDING

itk = pytest.importorskip("itk")


def _image(dims, shape, scale, translation, chunks=None, dtype=np.float32):
    rng = np.random.default_rng(0)
    array = (rng.random(tuple(shape[d] for d in dims)) * 100.0).astype(dtype)
    data = da.from_array(array, chunks=chunks or tuple(shape[d] for d in dims))
    return NgffImage(
        data=data,
        dims=tuple(dims),
        scale={d: scale[d] for d in dims},
        translation={d: translation[d] for d in dims},
    )


def _translation(dimension, offset):
    transform = itk.TranslationTransform[itk.D, dimension].New()
    transform.SetOffset([float(v) for v in offset])
    return transform


def _identity(dimension):
    return _translation(dimension, [0.0] * dimension)


def _whole_image_reference(transform, fixed, moving, interpolator="linear"):
    """Resample in a single ITK-Wasm call, without any block decomposition."""
    from itkwasm_downsample import resample_to_reference
    from ngff_zarr.itk_transform_resample import _component_type
    from ngff_zarr.itk_transform_resample_bounding_box import (
        _as_itk_transform_list,
        _itk_direction,
        _metadata_only_itk_image,
        _spatial_dims,
    )
    from ngff_zarr.ngff_image_to_itk_image import ngff_image_to_itk_image

    itk_dims = list(reversed(_spatial_dims(fixed)))
    reference = _metadata_only_itk_image(
        fixed, itk_dims, _itk_direction(fixed, itk_dims)
    )
    reference.imageType.componentType = _component_type(moving.data.dtype)
    resampled = resample_to_reference(
        ngff_image_to_itk_image(moving, wasm=True),
        reference,
        transform=_as_itk_transform_list(transform),
        interpolator=interpolator,
    )
    return np.asarray(resampled.data).reshape(fixed.data.shape)


def test_block_decomposition_matches_a_single_whole_image_call():
    """Splitting into blocks must not change a single pixel."""
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )
    transform = _translation(2, [3.5, -2.25])

    result = itk_transform_resample(transform, fixed, moving)
    assert result.data.numblocks == (4, 4)

    expected = _whole_image_reference(transform, fixed, moving)
    np.testing.assert_array_equal(np.asarray(result.data), expected)


def test_geometry_and_dtype_follow_the_right_image():
    moving = _image(
        "yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, dtype=np.uint8
    )
    fixed = _image("yx", {"y": 16, "x": 8}, {"y": 2.0, "x": 4.0}, {"y": 5.0, "x": 7.0})

    result = itk_transform_resample(_identity(2), fixed, moving)

    assert result.data.shape == (16, 8)
    assert result.dims == ("y", "x")
    assert result.scale == fixed.scale
    assert result.translation == fixed.translation
    assert result.data.dtype == np.uint8


def test_nothing_is_computed_until_the_result_is():
    """The moving image must not be read while the graph is only built."""
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(32, 32)
    )
    reads = []

    def counting(block, block_info=None):
        reads.append(1)
        return block

    moving = NgffImage(
        data=moving.data.map_blocks(counting, dtype=moving.data.dtype),
        dims=moving.dims,
        scale=moving.scale,
        translation=moving.translation,
    )

    # dask probes the wrapper as it is built, so measure from there.
    baseline = len(reads)

    result = itk_transform_resample(_identity(2), fixed, moving)
    assert len(reads) == baseline

    np.asarray(result.data)
    assert len(reads) > baseline


def test_a_block_outside_the_moving_image_gets_the_default_value():
    moving = _image("yx", {"y": 16, "x": 16}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 1000, "x": 1000})

    result = itk_transform_resample(_identity(2), fixed, moving, default_value=7.0)

    assert np.all(np.asarray(result.data) == 7.0)


def test_each_block_reads_only_its_own_region():
    """The regions the blocks read must stay inside what the whole grid reads."""
    moving = _image("yx", {"y": 128, "x": 128}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )
    transform = _translation(2, [2.0, 3.0])

    whole = itk_transform_resample_bounding_box(transform, fixed, moving)
    whole_bounds = whole.clamped()

    from ngff_zarr.itk_transform_resample import _block_grid

    for y_start in range(0, 64, 16):
        for x_start in range(0, 64, 16):
            grid = _block_grid(fixed, {"y": y_start, "x": x_start}, (16, 16))
            region = itk_transform_resample_bounding_box(transform, grid, moving)
            bounds = region.clamped()
            for dim in ("y", "x"):
                assert bounds[dim][0] >= whole_bounds[dim][0]
                assert bounds[dim][1] <= whole_bounds[dim][1]
                assert bounds[dim][1] - bounds[dim][0] <= 16 + 2 * 1 + 1


@pytest.mark.parametrize("interpolator", ["linear", "nearest_neighbor"])
def test_interpolators_are_forwarded(interpolator):
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )
    transform = _translation(2, [0.5, 0.5])

    result = np.asarray(
        itk_transform_resample(transform, fixed, moving, interpolator=interpolator).data
    )
    expected = _whole_image_reference(
        transform, fixed, moving, interpolator=interpolator
    )
    np.testing.assert_array_equal(result, expected)


def test_an_unknown_interpolator_is_rejected():
    fixed = _image("yx", {"y": 8, "x": 8}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="interpolator must be one of"):
        itk_transform_resample(_identity(2), fixed, fixed, interpolator="cubic")


@pytest.mark.parametrize("bad", [-1, 1.5, float("nan")])
def test_padding_that_is_not_a_non_negative_integer_is_rejected(bad):
    fixed = _image("yx", {"y": 8, "x": 8}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    with pytest.raises(ValueError, match="padding must be a non-negative integer"):
        itk_transform_resample(_identity(2), fixed, fixed, padding=bad)


def test_mismatched_dims_are_rejected():
    fixed = _image("yx", {"y": 8, "x": 8}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    moving = _image(
        "zyx",
        {"z": 8, "y": 8, "x": 8},
        {"z": 1, "y": 1, "x": 1},
        {"z": 0, "y": 0, "x": 0},
    )
    with pytest.raises(ValueError, match="they must match"):
        itk_transform_resample(_identity(2), fixed, moving)


def test_a_three_dimensional_grid_is_resampled_block_wise():
    moving = _image(
        "zyx",
        {"z": 32, "y": 32, "x": 32},
        {"z": 1, "y": 1, "x": 1},
        {"z": 0, "y": 0, "x": 0},
    )
    fixed = _image(
        "zyx",
        {"z": 32, "y": 32, "x": 32},
        {"z": 1, "y": 1, "x": 1},
        {"z": 0, "y": 0, "x": 0},
        chunks=(16, 16, 16),
    )
    transform = _translation(3, [1.5, -2.0, 0.75])

    result = itk_transform_resample(transform, fixed, moving)
    assert result.data.numblocks == (2, 2, 2)

    expected = _whole_image_reference(transform, fixed, moving)
    np.testing.assert_array_equal(np.asarray(result.data), expected)


@pytest.mark.parametrize("interpolator", list(_INTERPOLATOR_PADDING))
def test_default_padding_reproduces_an_undecomposed_resample(interpolator):
    """The default padding must make blocks agree with a single call, exactly.

    Too small a padding does not raise, it silently returns wrong pixels near
    block borders, so the defaults are pinned here.
    """
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )
    transform = _translation(2, [3.5, -2.25])

    result = np.asarray(
        itk_transform_resample(transform, fixed, moving, interpolator=interpolator).data
    )
    expected = _whole_image_reference(
        transform, fixed, moving, interpolator=interpolator
    )
    np.testing.assert_array_equal(result, expected)


def test_the_fixed_image_pixels_are_never_read():
    """Only the fixed grid's geometry is used, so its blocks must stay unread.

    Mapping over ``fixed.data`` instead of a placeholder would read every one
    of its blocks while returning exactly the same pixels, so nothing else in
    this file can catch it.
    """
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )
    reads = []

    fixed = NgffImage(
        data=fixed.data.map_blocks(
            lambda block: reads.append(1) or block, dtype=fixed.data.dtype
        ),
        dims=fixed.dims,
        scale=fixed.scale,
        translation=fixed.translation,
    )
    baseline = len(reads)

    result = itk_transform_resample(_identity(2), fixed, moving)
    np.asarray(result.data)

    assert len(reads) == baseline


def test_moving_chunks_shared_by_several_blocks_are_read_once():
    """Blocks are tasks of one graph that share the moving chunks they touch.

    With 16-pixel output blocks over 32-pixel moving chunks and a fractional
    translation, every moving chunk lies in the region of several output
    blocks. Reading it once per block would multiply I/O and decoding by that
    factor, which is what makes a remote or compressed moving image slow.
    """
    moving = _image("yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx", {"y": 64, "x": 64}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )
    reads = []

    moving = NgffImage(
        data=moving.data.rechunk((32, 32)).map_blocks(
            lambda block: reads.append(1) or block, dtype=moving.data.dtype
        ),
        dims=moving.dims,
        scale=moving.scale,
        translation=moving.translation,
    )
    baseline = len(reads)

    result = itk_transform_resample(_translation(2, [0.5, 0.5]), fixed, moving)
    np.asarray(result.data)

    assert len(reads) - baseline == moving.data.npartitions


def test_a_transform_that_shifts_off_the_moving_image_reads_no_chunks():
    """Blocks whose region misses the moving image are constants in the graph."""
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    fixed = _image(
        "yx",
        {"y": 32, "x": 32},
        {"y": 1, "x": 1},
        {"y": 500, "x": 500},
        chunks=(16, 16),
    )
    reads = []
    moving = NgffImage(
        data=moving.data.map_blocks(
            lambda block: reads.append(1) or block, dtype=moving.data.dtype
        ),
        dims=moving.dims,
        scale=moving.scale,
        translation=moving.translation,
    )
    baseline = len(reads)

    result = itk_transform_resample(_identity(2), fixed, moving, default_value=3.0)

    assert np.all(np.asarray(result.data) == 3.0)
    assert len(reads) == baseline


def test_moving_geometry_is_part_of_the_graph_name():
    """Two resamples differing only in where the moving image sits must not
    share Dask keys.

    Keys are global to a computation, so a collision does not raise: one
    result silently stands in for the other.
    """
    moving = _image("yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0})
    shifted = NgffImage(
        data=moving.data,
        dims=moving.dims,
        scale=moving.scale,
        translation={"y": 8.0, "x": 0.0},
    )
    fixed = _image(
        "yx", {"y": 32, "x": 32}, {"y": 1, "x": 1}, {"y": 0, "x": 0}, chunks=(16, 16)
    )

    first = itk_transform_resample(_identity(2), fixed, moving)
    second = itk_transform_resample(_identity(2), fixed, shifted)
    assert first.data.name != second.data.name

    alone_first = np.asarray(first.data)
    alone_second = np.asarray(second.data)
    assert not np.array_equal(alone_first, alone_second)

    together = np.asarray(da.stack([first.data, second.data]))
    np.testing.assert_array_equal(together[0], alone_first)
    np.testing.assert_array_equal(together[1], alone_second)


def test_the_graph_does_not_carry_a_buffer_for_every_block():
    """The tasks have to stay small enough to ship to a worker.

    A block's grid is geometry, but holding an array of the block's shape in
    every task costs one byte per output voxel. Locally that hides, since the
    pages are never touched; it is paid in full the moment the graph is
    serialized to a distributed worker.
    """
    import pickle

    def lazy(shape, chunks):
        return NgffImage(
            data=da.zeros(shape, chunks=chunks, dtype=np.uint8),
            dims=("y", "x"),
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
        )

    fixed = lazy((1024, 1024), (256, 256))
    moving = lazy((1024, 1024), (256, 256))

    result = itk_transform_resample(_identity(2), fixed, moving)
    layer = result.data.dask.layers[result.data.name]
    weight = sum(len(pickle.dumps(task, protocol=5)) for task in layer.values())

    assert weight < result.data.size // 16, (
        f"the graph weighs {weight} bytes to describe {result.data.size} voxels"
    )
