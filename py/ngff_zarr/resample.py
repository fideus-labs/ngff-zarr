# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Resample a moving image onto a fixed image grid, one block at a time."""

import functools
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from .displacement_field_transform import field_window
from .ngff_image import NgffImage
from .ngff_transform_to_itk_transform import ngff_transform_to_itk_transform
from .resample_bounding_box import (
    _as_itk_transform_list,
    _check_geometry,
    _field_stream,
    _grown,
    _identity_region,
    _identity_transform_list,
    _is_ngff_transform,
    _itk_direction,
    _metadata_only_itk_image,
    _shifted_translation,
    _spatial_dims,
    resample_bounding_box,
)
from .v06.zarr_metadata import Coordinates, Displacements

_INTERPOLATORS = (
    "linear",
    "nearest_neighbor",
    "label_image",
    "b_spline",
    "windowed_sinc",
    "gaussian",
)

#: Padding each interpolator needs for a block-wise resample to reproduce an
#: undecomposed one exactly. Measured rather than assumed: see
#: ``test_default_padding_reproduces_an_undecomposed_resample``. The wide
#: values are not kernel radii. ``b_spline`` prefilters coefficients over the
#: whole image it is handed, so cropping perturbs them everywhere and the
#: perturbation only decays with distance; 16 is where it falls below float32
#: resolution.
_INTERPOLATOR_PADDING = {
    "linear": 1,
    "nearest_neighbor": 1,
    "label_image": 1,
    "gaussian": 2,
    "windowed_sinc": 3,
    "b_spline": 16,
}

#: float64 resolves the prefilter perturbation further out than float32 does,
#: so ``b_spline`` needs twice the padding to stay exact on float64 images.
_FLOAT64_INTERPOLATOR_PADDING = {"b_spline": 32}


def _default_padding(interpolator: str, dtype) -> int:
    if np.dtype(dtype) == np.float64 and interpolator in _FLOAT64_INTERPOLATOR_PADDING:
        return _FLOAT64_INTERPOLATOR_PADDING[interpolator]
    return _INTERPOLATOR_PADDING[interpolator]


def _component_type(out_dtype) -> str:
    from itkwasm import FloatTypes, IntTypes

    mapping = {
        "uint8": IntTypes.UInt8,
        "int8": IntTypes.Int8,
        "uint16": IntTypes.UInt16,
        "int16": IntTypes.Int16,
        "uint32": IntTypes.UInt32,
        "int32": IntTypes.Int32,
        "uint64": IntTypes.UInt64,
        "int64": IntTypes.Int64,
        "float32": FloatTypes.Float32,
        "float64": FloatTypes.Float64,
    }
    name = np.dtype(out_dtype).name
    if name not in mapping:
        msg = f"unsupported dtype {name!r} for resampling"
        raise ValueError(msg)
    return mapping[name]


def _shape_only(shape: tuple) -> np.ndarray:
    """An array carrying a shape without owning memory for it.

    Only ``shape`` and ``dtype`` are ever read off these stand-ins. Allocating
    for real costs one byte per output voxel, which looks free while the pages
    stay untouched and is then paid in full the moment something serializes
    the object holding it.
    """
    return np.broadcast_to(np.zeros((), dtype=np.uint8), shape)


def _block_grid(fixed: NgffImage, starts: dict, shape: tuple) -> NgffImage:
    """A geometry-only stand-in for one output block of the fixed grid."""
    return NgffImage(
        data=_shape_only(shape),
        dims=tuple(fixed.dims),
        scale=dict(fixed.scale),
        translation=_shifted_translation(fixed, starts),
        name=fixed.name,
        axes_units=fixed.axes_units,
        axes_orientations=fixed.axes_orientations,
        axes_types=fixed.axes_types,
    )


def _chunk_offsets(chunks):
    """The index of each chunk's first element, per axis."""
    return [np.concatenate([[0], np.cumsum(sizes)[:-1]]) for sizes in chunks]


def _nested_key(keys, index):
    """The dask key at ``index`` in a nested key list."""
    for axis in index:
        keys = keys[axis]
    return keys


def _chunk_span(offsets, bounds):
    """Which chunks a region touches: the first, how many, and where it starts.

    ``offsets`` holds the index of each chunk's first element per axis, as
    :func:`_chunk_offsets` gives them, and ``bounds`` the region as
    ``(start, stop)`` per axis.
    """
    first = tuple(
        int(np.searchsorted(offsets[axis], start, side="right") - 1)
        for axis, (start, _stop) in enumerate(bounds)
    )
    last = tuple(
        int(np.searchsorted(offsets[axis], stop - 1, side="right"))
        for axis, (_start, stop) in enumerate(bounds)
    )
    nchunks = tuple(stop - start for start, stop in zip(first, last))
    offset = tuple(int(offsets[axis][start]) for axis, start in enumerate(first))
    return first, nchunks, offset


def _field_block_transform(
    chunks,
    *,
    transform,
    nchunks,
    offset,
    bounds,
    field_dims,
    field_scale,
    field_translation,
    field_axes_types,
):
    """The ITK transform one output block needs, from that block's field window.

    Built inside the task, from the field chunks the window touches, so the
    field reaches ITK a window at a time instead of whole.
    """
    from .displacement_field_transform import ngff_displacement_field_to_itk_transform

    crop = NgffImage(
        data=_assemble_region(chunks, nchunks, offset, bounds),
        dims=field_dims,
        scale=dict(field_scale),
        translation=dict(field_translation),
        axes_types=dict(field_axes_types),
    )
    return ngff_displacement_field_to_itk_transform(transform, crop, field_dims[1:])


def _assemble_region(chunks, nchunks, offset, bounds):
    """Reassemble whole chunks into the sub-array ``bounds`` selects.

    ``chunks`` are whole chunks in C order over the ``nchunks`` grid that
    covers the region, ``offset`` the index of the first of them in the array
    they came from, and ``bounds`` the region itself, as ``(start, stop)`` per
    axis.
    """
    nested = np.empty(nchunks, dtype=object)
    for flat, index in enumerate(np.ndindex(*nchunks)):
        nested[index] = chunks[flat]

    # Where each chunk of the grid starts in the array it came from, per axis.
    starts = []
    for axis in range(len(nchunks)):
        picker = [0] * len(nchunks)
        sizes = []
        for position in range(nchunks[axis]):
            picker[axis] = position
            sizes.append(nested[tuple(picker)].shape[axis])
        starts.append(np.concatenate([[0], np.cumsum(sizes)[:-1]]) + offset[axis])

    # Trim every chunk to its share of the region before assembling. Assembling
    # first and slicing after would allocate the union of the chunks, which for
    # a block whose region straddles chunk borders is several times the region.
    for index in np.ndindex(*nchunks):
        chunk = nested[index]
        nested[index] = chunk[
            tuple(
                slice(
                    max(0, low - starts[axis][position]),
                    min(chunk.shape[axis], high - starts[axis][position]),
                )
                for axis, (position, (low, high)) in enumerate(zip(index, bounds))
            )
        ]

    if len(chunks) == 1:
        return np.ascontiguousarray(nested[(0,) * len(nchunks)])
    return np.block(nested.tolist())


def _resample_block(
    chunks,
    transform_list,
    *,
    nchunks,
    offset,
    bounds,
    moving_dims,
    moving_scale,
    moving_translation,
    moving_orientations,
    grid_dims,
    grid_scale,
    grid_translation,
    grid_shape,
    grid_orientations,
    interpolator: str,
    default_value: float,
    out_dtype,
):
    """Resample one output block from the moving chunks its region touches.

    ``chunks`` are whole moving-image chunks, in C order over the ``nchunks``
    grid of chunks that covers the block's region; ``offset`` is the index of
    the first of them in the moving image and ``bounds`` the region itself.
    ``transform_list`` arrives as a task argument rather than bound into the
    callable, so the graph carries it once for every block instead of once per
    block: see where the tasks are built.
    """
    from itkwasm_downsample import resample_to_reference

    from .ngff_image_to_itk_image import ngff_image_to_itk_image

    region = _assemble_region(chunks, nchunks, offset, bounds)
    moving_geometry = NgffImage(
        data=_shape_only(tuple(stop - start for start, stop in bounds)),
        dims=moving_dims,
        scale=dict(moving_scale),
        translation=dict(moving_translation),
        axes_orientations=moving_orientations,
    )
    crop = NgffImage(
        data=region,
        dims=moving_dims,
        scale=dict(moving_scale),
        translation=_shifted_translation(
            moving_geometry,
            {dim: start for dim, (start, _stop) in zip(moving_dims, bounds)},
        ),
        axes_orientations=moving_orientations,
    )

    grid = NgffImage(
        data=_shape_only(grid_shape),
        dims=grid_dims,
        scale=dict(grid_scale),
        translation=dict(grid_translation),
        axes_orientations=grid_orientations,
    )
    itk_dims = list(reversed(_spatial_dims(grid)))
    reference = _metadata_only_itk_image(grid, itk_dims, _itk_direction(grid, itk_dims))
    # resample_to_reference reads only the reference geometry, but it aborts
    # when the component type disagrees with the moving image, so align it.
    reference.imageType.componentType = _component_type(out_dtype)

    resampled = resample_to_reference(
        ngff_image_to_itk_image(crop, wasm=True),
        reference,
        transform=transform_list,
        interpolator=interpolator,
        default_value=float(default_value),
    )
    return np.asarray(resampled.data).reshape(grid_shape).astype(out_dtype, copy=False)


def _add_field_transform(
    graph,
    name,
    index,
    transform,
    field: NgffImage,
    field_keys,
    field_offsets,
    field_geometry,
    window,
):
    """Add the task that builds one block's transform, and return its key."""
    bounds = ((0, int(field.data.shape[0])), *window)
    first, nchunks, offset = _chunk_span(field_offsets, bounds)
    spatial = tuple(field.dims[1:])
    key = (name, *index)
    graph[key] = (
        functools.partial(
            _field_block_transform,
            transform=transform,
            nchunks=nchunks,
            offset=offset,
            bounds=bounds,
            field_translation=_shifted_translation(
                field, {dim: window[axis][0] for axis, dim in enumerate(spatial)}
            ),
            **field_geometry,
        ),
        [
            _nested_key(
                field_keys,
                tuple(start + relative for start, relative in zip(first, position)),
            )
            for position in np.ndindex(*nchunks)
        ],
    )
    return key


def resample(
    transform,
    fixed: NgffImage,
    moving: NgffImage,
    padding: int | None = None,
    interpolator: str = "linear",
    default_value: float = 0.0,
    *,
    fields: Mapping[str, object] | None = None,
) -> NgffImage:
    """Resample ``moving`` onto the grid of ``fixed`` through ``transform``.

    The result is a lazy :class:`NgffImage`: nothing is read or computed until
    the returned Dask array is. Each output block is resampled on its own from
    the moving chunks inside the region its resample reads, as reported by
    :func:`ngff_zarr.resample_bounding_box`. The regions are computed once,
    when the graph is built, and the blocks are tasks of a single Dask graph
    that reference the moving chunks directly, so a chunk that several blocks
    need is read and decoded once per computation. The
    full moving image is never loaded, which is what makes this usable when it
    is larger than memory, remote, or chunked.

    An RFC-5 ``displacements`` or ``coordinates`` field is streamed the same
    way. A block reads the window of the field its own points fall in, and
    that window becomes the block's ITK transform; what sizes the block's
    moving read is the range of displacement that window holds. The field is
    passed over once when the graph is built, a chunk at a time, to learn that
    range per chunk. Neither the moving image nor the field has to fit in
    memory.

    A window declares its own origin, and on a float64 moving image that
    changes the last bits of the continuous index ITK computes from it, so
    the result is bit-identical to an undecomposed call on float32 and equal
    to about ``1e-13`` on float64.

    Resampling runs through ``itkwasm-downsample``, so no native ITK build is
    required and the result is identical to one across platforms.

    :param transform: An RFC-5 coordinate transformation, an ``itk.Transform``
        (including the ``CompositeTransform`` an Elastix registration
        returns), or an ITK-Wasm ``Transform`` / ``TransformList``. It maps
        *fixed* points into *moving* space.

        The two are interpreted in different coordinate spaces, exactly as
        :func:`ngff_zarr.resample_bounding_box` interprets them. An RFC-5
        transformation acts on the intrinsic coordinate systems, where a point
        is ``translation + scale * index`` and no direction matrix applies, so
        the anatomical orientation of either image does not enter; its
        ``input`` and ``output`` identifiers are not resolved either. An ITK
        transform acts on ITK physical space, direction matrix included.

    :param fixed: The image whose grid defines the output. Geometry only, its
        pixels are never read.
    :type  fixed: NgffImage

    :param moving: The image to sample.
    :type  moving: NgffImage

    :param padding: Pixels of padding added per side when computing each
        block's source region. Defaults to what ``interpolator`` reads beyond
        the continuous index bound, so the two stay in step; ``b_spline``
        defaults to 16, or 32 on float64 images.
    :type  padding: int | None

    :param interpolator: One of ``"linear"`` (default),
        ``"nearest_neighbor"``, ``"label_image"``, ``"b_spline"``,
        ``"windowed_sinc"`` or ``"gaussian"``.
    :type  interpolator: str

    :param default_value: Value written where a block samples outside
        ``moving``.
    :type  default_value: float

    :param fields: The field images an RFC-5 ``displacements`` or
        ``coordinates`` transformation points at, keyed by its ``path``, as
        :func:`ngff_zarr.ngff_transform_to_itk_transform` takes them. Required
        for those two, ignored otherwise.
    :type  fields: Mapping[str, NgffImage | NgffMultiscales], optional

    :return: The resampled image, carrying the geometry of ``fixed`` and the
        dtype of ``moving``.
    :rtype: NgffImage
    """
    import dask.array as da
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    if interpolator not in _INTERPOLATORS:
        msg = f"interpolator must be one of {_INTERPOLATORS}, got {interpolator!r}"
        raise ValueError(msg)
    if padding is None:
        padding = _default_padding(interpolator, moving.data.dtype)
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        msg = f"padding must be a non-negative integer, got {padding!r}"
        raise ValueError(msg)

    fixed_spatial = _spatial_dims(fixed)
    moving_spatial = _spatial_dims(moving)
    if fixed_spatial != moving_spatial:
        msg = (
            f"fixed image has spatial dims {tuple(fixed_spatial)} but moving "
            f"image has {tuple(moving_spatial)}; they must match"
        )
        raise ValueError(msg)
    if len(fixed_spatial) not in (2, 3):
        msg = (
            f"images have {len(fixed_spatial)} spatial dims "
            f"{tuple(fixed_spatial)}; only 2 and 3 are supported"
        )
        raise ValueError(msg)
    if tuple(fixed.dims) != tuple(moving.dims):
        msg = (
            f"fixed image has dims {tuple(fixed.dims)} but moving image has "
            f"{tuple(moving.dims)}; they must match"
        )
        raise ValueError(msg)
    non_spatial = tuple(dim for dim in fixed.dims if dim not in fixed_spatial)
    if non_spatial:
        msg = (
            f"images have non-spatial dims {non_spatial}; index or squeeze "
            "them so that only spatial dims remain before resampling"
        )
        raise ValueError(msg)
    for label, image in (("fixed", fixed), ("moving", moving)):
        _check_geometry(label, image, fixed_spatial)

    dtype = moving.data.dtype
    out_orientations = fixed.axes_orientations
    field = None
    field_bound = None
    if _is_ngff_transform(transform):
        if isinstance(transform, (Coordinates, Displacements)):
            # The blocks divide the grid, so the grid's own window is the
            # union of theirs: the pass reads no chunk no block asks about.
            field, field_bound, _window, _outside = _field_stream(
                transform,
                fields,
                fixed,
                fixed_spatial,
                dict(zip(fixed.dims, fixed.data.shape)),
            )
            transform_list = _identity_transform_list(fixed.dims)
        else:
            transform_list = ngff_transform_to_itk_transform(
                transform, fixed.dims, fields=fields
            )
        # An RFC-5 transformation acts on the intrinsic coordinate systems,
        # which carry no direction matrix, so the blocks below are resampled
        # with none either. That is what resample_bounding_box does with the
        # same transformation, and the two have to read the same pixels.
        fixed = replace(fixed, axes_orientations=None)
        moving = replace(moving, axes_orientations=None)
    else:
        transform_list = _as_itk_transform_list(transform)

    out_chunks = fixed.data.chunks
    out_offsets = _chunk_offsets(out_chunks)
    moving_offsets = _chunk_offsets(moving.data.chunks)
    moving_keys = moving.data.__dask_keys__()

    field_data = None
    if field is not None:
        field_data = field.data
        if not isinstance(field_data, da.Array):
            field_data = da.from_array(field_data)
        field_offsets = _chunk_offsets(field_data.chunks)
        field_keys = field_data.__dask_keys__()
        # The geometry every block's crop of the field shares, built once.
        field_geometry = {
            "field_dims": tuple(field.dims),
            "field_scale": dict(field.scale),
            "field_axes_types": dict(field.axes_types),
        }

    # Every input the result depends on belongs in the token. The moving
    # image's geometry is not carried by its array name, so leaving it out
    # gives two resamples of the same array at different origins the same key,
    # and one silently stands in for the other.
    name = "ngff-zarr-resample-" + tokenize(
        moving.data.name,
        moving.dims,
        moving.scale,
        moving.translation,
        moving.axes_orientations,
        out_chunks,
        fixed.dims,
        fixed.scale,
        fixed.translation,
        fixed.axes_orientations,
        transform_list,
        None if field_data is None else field_data.name,
        None if field is None else (field.scale, field.translation, transform),
        padding,
        interpolator,
        default_value,
    )
    # One entry for the transform, named by every block, rather than one copy
    # bound into each block's callable. Dask ships a task's run spec to a worker
    # on its own, so a displacement field or a long composite would otherwise
    # cross the wire once per block.
    transform_key = f"{name}-transform"
    graph = {transform_key: transform_list}
    for index in np.ndindex(*[len(sizes) for sizes in out_chunks]):
        starts = {
            dim: int(out_offsets[axis][index[axis]])
            for axis, dim in enumerate(fixed.dims)
        }
        shape = tuple(int(out_chunks[axis][index[axis]]) for axis in range(len(index)))
        grid = _block_grid(fixed, starts, shape)
        block_transform = transform_key
        if field is not None:
            window, outside = field_window(
                field, tuple(fixed.dims), grid.translation, grid.scale, shape
            )
            region = _grown(
                _identity_region(grid, moving, padding),
                field_bound.over(window, outside),
                moving,
            )
        else:
            region = resample_bounding_box(
                transform_list, grid, moving, padding=padding
            )
        if region.is_empty:
            graph[(name, *index)] = (np.full, shape, default_value, dtype)
            continue
        if field is not None and not any(stop <= start for start, stop in window):
            # A block whose window is empty falls through to transform_key,
            # which on this path holds the identity: outside the field ITK
            # displaces nothing either.
            block_transform = _add_field_transform(
                graph,
                f"{name}-field",
                index,
                transform,
                field,
                field_keys,
                field_offsets,
                field_geometry,
                window,
            )
        clamped = region.clamped()
        bounds = tuple(
            clamped[dim] if dim in clamped else (0, moving.data.shape[axis])
            for axis, dim in enumerate(moving.dims)
        )
        first_chunk, nchunks, offset = _chunk_span(moving_offsets, bounds)
        chunk_keys = [
            _nested_key(
                moving_keys,
                tuple(first + rel for first, rel in zip(first_chunk, relative)),
            )
            for relative in np.ndindex(*nchunks)
        ]
        graph[(name, *index)] = (
            functools.partial(
                _resample_block,
                nchunks=nchunks,
                offset=offset,
                bounds=bounds,
                moving_dims=tuple(moving.dims),
                moving_scale=dict(moving.scale),
                moving_translation=dict(moving.translation),
                moving_orientations=moving.axes_orientations,
                grid_dims=tuple(fixed.dims),
                grid_scale=dict(fixed.scale),
                grid_translation=dict(grid.translation),
                grid_shape=shape,
                grid_orientations=fixed.axes_orientations,
                interpolator=interpolator,
                default_value=default_value,
                out_dtype=dtype,
            ),
            chunk_keys,
            block_transform,
        )

    dependencies = [moving.data]
    if field_data is not None:
        dependencies.append(field_data)
    data = da.Array(
        HighLevelGraph.from_collections(name, graph, dependencies=dependencies),
        name,
        chunks=out_chunks,
        dtype=dtype,
    )

    return NgffImage(
        data=data,
        dims=tuple(fixed.dims),
        scale=dict(fixed.scale),
        translation=dict(fixed.translation),
        name=moving.name,
        axes_units=fixed.axes_units,
        axes_orientations=out_orientations,
        axes_types=fixed.axes_types,
        channel_names=moving.channel_names,
        channel_colors=moving.channel_colors,
    )
