# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Bridge ITK displacement fields and RFC-5 ``displacements`` transformations.

A displacement field is a transformation and an array at once. ITK keeps the
array inside the transform, as a vector image with its own grid; RFC-5 keeps it
in the store, as a multiscale image the ``displacements`` entry points at by
``path``. So the two conversions here are the only ones in this package whose
result has two parts, and they stay pure: nothing is read from or written to a
store. The caller writes the field next to its image, and loads it back, with
:func:`~ngff_zarr.to_ome_zarr` and :func:`~ngff_zarr.from_ome_zarr`.

The same two conventions as the affine conversion apply, plus one.

Axis order
    ITK orders a vector's components fastest-axis-first, x then y then z, and
    lays the field out as ``[z][y][x][component]``. RFC-5 says the ``i``-th
    component of the field refers to the ``i``-th axis of the output
    coordinate system, so components follow ``dims``; the same permutation as
    for an affine's matrix is applied to the component axis.

Frames
    An ITK vector is a difference of two physical points. An RFC-5
    displacement is a difference of two intrinsic points, ``d = q' - q``, so
    the two differ by more than a rotation as soon as the images sit in
    different frames. Both directions follow from one identity,
    ``phi_out(q + d(q)) = phi_in(q) + v(q)``, worked out in
    :func:`_frame_terms` on top of the affine conversion's change of frame.
    The field's own grid follows ``phi_in^-1``, so its origin moves and its
    spacing does not.

Grid direction
    RFC-5 maps the field's array coordinates to the input system with the
    field's own ``coordinateTransformations``, which this package writes as a
    scale and a translation. That can only express a grid oriented like the
    input image, so a field whose direction matrix differs from the fixed
    image's (the identity when no frames are given) is refused rather than
    written with a mapping a reader would misread. Registrations sample the
    field on the fixed grid, so the common case is exact.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .itk_transform_to_ngff_transform import _FrameGeometry
from .ngff_image import NgffImage
from .v06.zarr_metadata import Coordinates, Displacements

#: The name given to the component axis of a converted field. RFC-5 puts that
#: axis after a time axis and before the spatial ones, with
#: ``type: "displacement"``.
_COMPONENT_DIM = "c"


def _check_dims(dims: Sequence[str]) -> tuple[str, ...]:
    """Validate the spatial-only ``dims`` a displacement field is defined on."""
    from .itk_transform_to_ngff_transform import _SPATIAL_DIMS

    dims = tuple(dims)
    other = [dim for dim in dims if dim not in _SPATIAL_DIMS]
    if not dims or other:
        msg = (
            f"a displacement field is defined on spatial axes only; dims {dims} "
            f"name {other or 'none'}. ITK has no notion of a time or channel "
            "axis, and RFC-5 requires one field dimension per input axis."
        )
        raise ValueError(msg)
    if len(set(dims)) != len(dims):
        msg = f"dims {dims} name an axis twice"
        raise ValueError(msg)
    return dims


def _frames(fixed, moving, dims):
    """The two images' directions and origins in ITK order, or ``None``."""
    from .itk_transform_to_ngff_transform import (
        _check_frame_images,
        _frame_geometry,
        _itk_axis_order,
    )

    if not _check_frame_images(fixed, moving, dims):
        return None
    return _frame_geometry(fixed, moving, _itk_axis_order(dims))


def _unoriented_frames(dimension: int) -> _FrameGeometry:
    """The frame pair of two images with no orientation and no translation.

    With it the terms below all vanish, so the conversion that changes frames
    and the one that does not are the same arithmetic rather than two branches.
    """
    identity = np.eye(dimension)
    zero = np.zeros(dimension)
    return _FrameGeometry(identity, identity, zero, zero)


def _frame_terms(frames: _FrameGeometry):
    """The per-point terms relating ITK vectors and RFC-5 displacements.

    An ITK vector is a difference of two *physical* points; an RFC-5
    displacement is a difference of two *intrinsic* points. Writing
    ``phi_out^-1 . phi_in`` as ``q -> M q + b`` -- the change of frame the
    affine conversion applies, given the identity as its mapping -- the
    defining identity ``phi_out(q + d(q)) = phi_in(q) + v(q)`` rearranges to

        d(q) = D_out^-1 v + (M - I) q + b
        v(q) = D_out (d - (M - I) q - b)

    so one derivation serves both directions. The two ``q`` terms vanish when
    the images share a frame, leaving ``d = D_out^-1 v``.

    :return: ``(D_out, M - I, b)``.
    """
    from .itk_transform_to_ngff_transform import _change_of_frame

    dimension = len(frames.origin_in)
    matrix, offset = _change_of_frame(np.eye(dimension), np.zeros(dimension), *frames)
    return frames.direction_out, matrix - np.eye(dimension), offset


def _grid_shift(shape, origin, spacing, matrix, vector):
    """``matrix q + vector`` at every grid point, or ``None`` when it is zero.

    ``shape`` is the field's ``[z][y][x]`` layout, so the index arrays come out
    slowest-axis-first and are reversed into ITK component order. The result is
    ``(*shape, N)``, ready to add to or subtract from the field.
    """
    if not matrix.any() and not vector.any():
        return None
    indices = np.stack(np.indices(shape, dtype=np.float64)[::-1], axis=-1)
    return (origin + spacing * indices) @ matrix.T + vector


def _is_identity(matrix: np.ndarray) -> bool:
    return bool(np.array_equal(matrix, np.eye(len(matrix))))


def _decode_field(transform):
    """Read any supported field input into ITK conventions.

    Returns ``(vectors, origin, spacing, direction)``: the field as a
    ``[z][y][x][component]`` array with components in ITK order, and its grid
    in ITK component order.
    """
    if isinstance(transform, (list, tuple)):
        if len(transform) != 1:
            msg = (
                f"expected a single displacement field, got a transform list of "
                f"{len(transform)} entries. A registration that chains an affine "
                "and a field is not converted as one transform."
            )
            raise ValueError(msg)
        return _decode_field(transform[0])

    try:
        from itkwasm import Image as ItkWasmImage
        from itkwasm import Transform as ItkWasmTransform
    except ImportError:  # pragma: no cover - itkwasm is a hard dependency
        ItkWasmImage = ItkWasmTransform = ()

    if isinstance(transform, ItkWasmTransform):
        return _decode_itkwasm_transform(transform)
    if isinstance(transform, ItkWasmImage):
        from dataclasses import asdict

        return _decode_image_dict(asdict(transform))

    if hasattr(transform, "GetDisplacementField"):
        import itk

        return _decode_image_dict(itk.dict_from_image(transform.GetDisplacementField()))

    try:
        import itk
    except ImportError:
        itk = None
    if itk is not None and isinstance(transform, (itk.Image, itk.VectorImage)):
        return _decode_image_dict(itk.dict_from_image(transform))

    msg = (
        f"unsupported displacement field input {type(transform).__name__}. "
        "Expected an itk.DisplacementFieldTransform, a vector itk.Image, an "
        "itkwasm.Image, or an ITK-Wasm 'DisplacementField' transform."
    )
    raise TypeError(msg)


def _decode_itkwasm_transform(transform):
    """The field an ITK-Wasm ``DisplacementField`` entry packs."""
    transform_type = transform.transformType
    parameterization = getattr(
        transform_type.transformParameterization,
        "value",
        transform_type.transformParameterization,
    )
    if parameterization != "DisplacementField":
        msg = (
            f"expected an ITK-Wasm 'DisplacementField' transform, got "
            f"'{parameterization}'. Linear transforms are converted by "
            "itk_transform_to_ngff_transform."
        )
        raise ValueError(msg)
    dimension = int(transform_type.inputDimension)
    fixed = np.asarray(transform.fixedParameters, dtype=np.float64)
    expected = 3 * dimension + dimension * dimension
    if fixed.size != expected:
        msg = (
            f"an ITK-Wasm 'DisplacementField' transform of dimension {dimension} "
            f"packs size, origin, spacing and direction into {expected} fixed "
            f"parameters; got {fixed.size}"
        )
        raise ValueError(msg)
    size = fixed[:dimension].astype(int)
    origin = fixed[dimension : 2 * dimension]
    spacing = fixed[2 * dimension : 3 * dimension]
    direction = fixed[3 * dimension :].reshape(dimension, dimension)
    parameters = np.asarray(transform.parameters)
    if parameters.size != int(np.prod(size)) * dimension:
        msg = (
            f"an ITK-Wasm 'DisplacementField' transform over a grid of size "
            f"{size.tolist()} holds {int(np.prod(size)) * dimension} parameters; "
            f"got {parameters.size}"
        )
        raise ValueError(msg)
    # ITK packs the field slowest-axis-first with the component innermost.
    vectors = parameters.reshape(*size[::-1], dimension)
    return vectors, origin, spacing, direction


def _decode_image_dict(image):
    """The field a vector image dictionary holds."""
    data = np.asarray(image["data"])
    components = int(image["imageType"]["components"])
    dimension = data.ndim - 1 if components > 1 else data.ndim
    if components != dimension or data.ndim != dimension + 1:
        msg = (
            f"a displacement field over {dimension} dimensions needs {dimension} "
            f"components per pixel; got {components}"
        )
        raise ValueError(msg)
    origin = np.asarray(image["origin"], dtype=np.float64)
    spacing = np.asarray(image["spacing"], dtype=np.float64)
    direction = np.asarray(image["direction"], dtype=np.float64).reshape(
        dimension, dimension
    )
    return data, origin, spacing, direction


def itk_displacement_field_to_ngff_transform(
    transform,
    dims: Sequence[str],
    *,
    path: str,
    fixed: NgffImage | None = None,
    moving: NgffImage | None = None,
) -> tuple[Displacements, NgffImage]:
    """Convert an ITK displacement field to an RFC-5 ``displacements`` transform.

    The result has two parts, because the field is an array: the transform
    metadata, which names ``path``, and the field itself as an image to be
    written at that path. Write the field first, into a subgroup of the store
    the image goes to, then the image::

        transform, field = itk_displacement_field_to_ngff_transform(
            itk_transform, ["z", "y", "x"], path="displacement_field"
        )
        to_ome_zarr(f"{store}/displacement_field", to_multiscales(field))
        multiscales.metadata.coordinateTransformations = [transform]
        to_ome_zarr(store, multiscales, overwrite=False)

    :param transform: An ``itk.DisplacementFieldTransform``, a vector
        ``itk.Image`` or ``itkwasm.Image`` holding the field directly (the form
        a warp comes in from most registration tools), an ITK-Wasm
        ``DisplacementField`` transform, or a one-entry list of it. The field
        maps *fixed* points into *moving* space.
    :param dims: The spatial axis names of the input coordinate system, in
        RFC-5 (Zarr) order. The field must be defined on these axes and no
        others.
    :type  dims: Sequence[str]
    :param path: Where the field will be written, relative to the image's
        group. Recorded on the returned transform.
    :type  path: str
    :param fixed: The fixed and moving images the field relates. Passing both
        re-expresses the vectors on the images' intrinsic coordinate systems,
        including the direction matrix derived from RFC-4 anatomical
        orientation, and gives the field the fixed image's orientation and
        units. Omitting them is exact only when neither image carries an
        anatomical orientation.
    :type  fixed: NgffImage, optional
    :param moving: See ``fixed``. Pass both or neither.
    :type  moving: NgffImage, optional
    :return: The ``displacements`` transform, with ``interpolation`` set to
        ``linear`` (ITK's interpolator for a field), and the field as an
        ``NgffImage`` whose first axis carries the components, ``type:
        "displacement"``, followed by ``dims``.
    :rtype: tuple[Displacements, NgffImage]
    :raises ValueError: If the field's grid is not oriented like the fixed
        image (the identity without frames), which the field's scale and
        translation could not express; resample the field onto the fixed grid
        first. Also for a field whose dimensionality does not match ``dims``.
    """
    import dask.array

    from .itk_transform_to_ngff_transform import _inverse_direction, _itk_axis_order

    dims = _check_dims(dims)
    vectors, origin, spacing, direction = _decode_field(transform)
    dimension = vectors.shape[-1]
    if dimension != len(dims):
        msg = (
            f"the field has {dimension} components over {vectors.ndim - 1} "
            f"dimensions, but dims {dims} name {len(dims)} axes"
        )
        raise ValueError(msg)
    itk_dims = _itk_axis_order(dims)
    canonical = list(reversed(itk_dims))

    frames = _frames(fixed, moving, dims)
    if frames is None:
        if not np.allclose(direction, np.eye(dimension)):
            msg = (
                "the field's grid has a non-identity direction matrix, which its "
                "scale and translation cannot express. Pass the fixed and "
                "moving images so it is read against the fixed image's "
                "orientation, or resample the field onto the fixed grid."
            )
            raise ValueError(msg)
        frames = _unoriented_frames(dimension)
    elif not np.allclose(direction, frames.direction_in):
        msg = (
            "the field's grid is not oriented like the fixed image: its "
            f"direction is {direction.tolist()} where the fixed image gives "
            f"{frames.direction_in.tolist()}. Resample the field onto the fixed "
            "grid first."
        )
        raise ValueError(msg)

    # The field's grid follows phi_in^-1, so its origin moves and its spacing
    # does not.
    grid_origin = (
        _inverse_direction(frames.direction_in) @ (origin - frames.origin_in)
        + frames.origin_in
    )

    direction_out, shift_matrix, shift_vector = _frame_terms(frames)
    inverse_out = _inverse_direction(direction_out)
    displacements = vectors if _is_identity(inverse_out) else vectors @ inverse_out.T
    shift = _grid_shift(
        vectors.shape[:-1], grid_origin, spacing, shift_matrix, shift_vector
    )
    if shift is not None:
        displacements = displacements + shift

    # [z][y][x][c] with ITK components -> (c, *dims) with components in dims
    # order. The transposes are views; indexing the component axis is the one
    # copy, and it lands contiguous.
    field = np.moveaxis(displacements, -1, 0)
    field = np.transpose(field, [0] + [1 + canonical.index(dim) for dim in dims])
    field = field[[itk_dims.index(dim) for dim in dims]].astype(
        vectors.dtype, copy=False
    )

    scale = {_COMPONENT_DIM: 1.0}
    translation = {_COMPONENT_DIM: 0.0}
    for dim in dims:
        scale[dim] = float(spacing[itk_dims.index(dim)])
        translation[dim] = float(grid_origin[itk_dims.index(dim)])

    orientations = None
    units = None
    if fixed is not None:
        if fixed.axes_orientations:
            orientations = {
                dim: fixed.axes_orientations[dim]
                for dim in dims
                if dim in fixed.axes_orientations
            } or None
        if fixed.axes_units:
            units = {
                dim: fixed.axes_units[dim] for dim in dims if dim in fixed.axes_units
            } or None

    image = NgffImage(
        data=dask.array.from_array(field),
        dims=(_COMPONENT_DIM, *dims),
        scale=scale,
        translation=translation,
        name=path.rstrip("/").rsplit("/", 1)[-1] or "displacement_field",
        axes_units=units,
        axes_orientations=orientations,
        axes_types={_COMPONENT_DIM: "displacement"},
    )
    return Displacements(path=path, interpolation="linear"), image


def field_image(
    transform: Displacements | Coordinates, field, dims: Sequence[str]
) -> NgffImage:
    """The field image a field transform names, checked against ``dims``.

    Takes the ``NgffImage`` or the ``NgffMultiscales`` a caller passes in
    ``fields``, and returns the single-scale image, after checking that its
    axes are the component axis followed by ``dims`` in order and that it
    holds one component per input axis.

    :param transform: The ``displacements`` or ``coordinates`` transform.
    :type  transform: Displacements | Coordinates
    :param field: The field image or multiscales, as ``fields`` holds it.
    :param dims: The spatial axis names of the input coordinate system, in
        RFC-5 (Zarr) order.
    :type  dims: Sequence[str]
    :return: The field as a single-scale image; the finest level of a
        multiscales.
    :rtype: NgffImage
    :raises ValueError: If the field has no single component axis of the type
        the transform calls for, if its axes are not that axis followed by
        ``dims``, or if it holds a number of components other than ``len(dims)``.
    """
    dims = tuple(dims)
    component_type = "coordinate" if transform.type == "coordinates" else "displacement"
    if hasattr(field, "images") and hasattr(field, "metadata"):
        # A read multiscales keeps the axis types in its metadata, not on the
        # image: the component axis is the one typed there.
        axes = field.metadata.intrinsic_coordinate_system.axes
        component_dims = [axis.name for axis in axes if axis.type == component_type]
        field = field.images[0]
    else:
        component_dims = [
            dim
            for dim, axis_type in (field.axes_types or {}).items()
            if axis_type == component_type
        ]
    if len(component_dims) != 1:
        msg = (
            f"the field image must have exactly one axis of type "
            f"'{component_type}' (axes_types on an NgffImage, the axes metadata "
            f"of a multiscales); got {component_dims or 'none'} on dims "
            f"{tuple(field.dims)}"
        )
        raise ValueError(msg)
    expected_dims = (component_dims[0], *dims)
    if tuple(field.dims) != expected_dims:
        msg = (
            f"the field's dims are {tuple(field.dims)}; a {transform.type} "
            f"transform over dims {dims} needs {expected_dims}: the component "
            "axis first, then the input axes in order"
        )
        raise ValueError(msg)
    if field.data.shape[0] != len(dims):
        msg = (
            f"the field holds {field.data.shape[0]} components per point, but dims "
            f"{dims} name {len(dims)} axes"
        )
        raise ValueError(msg)
    return field


def check_unoriented_field(field: NgffImage, dims: Sequence[str]) -> None:
    """Refuse a field whose grid cannot be placed without the two images.

    An ITK transform lives in physical space, and an anatomical orientation
    only says where a grid sits once the images it relates are known.

    :param field: The field image.
    :type  field: NgffImage
    :param dims: The spatial axis names, in RFC-5 order.
    :type  dims: Sequence[str]
    :raises ValueError: If the field carries an anatomical orientation.
    """
    from .itk_transform_to_ngff_transform import _itk_axis_order
    from .resample_bounding_box import _itk_direction

    if _is_identity(_itk_direction(field, _itk_axis_order(tuple(dims)))):
        return
    msg = (
        "the field carries an anatomical orientation, so its grid "
        "cannot be placed in ITK physical space on its own; pass the "
        "fixed and moving images. resample_bounding_box and resample "
        "have no place for them, because their RFC-5 branch works on "
        "the intrinsic systems where no orientation applies: call "
        "ngff_transform_to_itk_transform with both images yourself and "
        "hand them the ITK transform it returns."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class FieldBound:
    """The range of displacement each chunk of a field can produce.

    A field is read through a kernel that is non-negative and sums to one, so
    a displacement anywhere in a chunk is a convex combination of that chunk's
    values and lies between their smallest and largest. That makes the range
    a bound on every interpolated displacement, not a sample of one: walking
    the boundary of a region instead misses a bump the region encloses.

    The range is kept rather than its magnitude, so a field that shifts every
    point the same way moves the region it bounds instead of widening it.

    The granularity is the field's own chunking, since that is what a read
    costs. A field stored in one chunk therefore reports one range for the
    whole volume, which is also the case where the field fits in memory.
    """

    #: Spatial dimension names, in the field's (RFC-5) order.
    dims: tuple[str, ...]
    #: Smallest and largest displacement per component, per chunk:
    #: ``(len(dims), *chunk counts)`` each, components ordered like ``dims``.
    low: np.ndarray
    high: np.ndarray
    #: Index of the first element of each chunk, per spatial axis.
    offsets: tuple[np.ndarray, ...]

    def over(
        self, window: Sequence[tuple[int, int]], outside: bool = False
    ) -> dict[str, tuple[float, float]]:
        """The displacement range over the chunks ``window`` touches.

        :param window: ``(start, stop)`` per spatial axis, in field index
            space and in ``dims`` order.
        :type  window: Sequence[tuple[int, int]]
        :param outside: Whether the grid the window came from also has points
            beyond the field. ITK displaces those by nothing, so zero belongs
            in the range as much as the values do; leaving it out lets a
            field that displaces every point it covers one way carry the
            region away from the points it does not cover.
        :type  outside: bool
        :return: ``{dim: (low, high)}`` per component, keyed by dimension.
            Zero for an empty window, where the field displaces nothing.
        :rtype: dict[str, tuple[float, float]]
        """
        if any(stop <= start for start, stop in window):
            return dict.fromkeys(self.dims, (0.0, 0.0))
        selection = []
        for axis, (start, stop) in enumerate(window):
            offsets = self.offsets[axis]
            first = int(np.searchsorted(offsets, start, side="right") - 1)
            last = int(np.searchsorted(offsets, stop - 1, side="right"))
            selection.append(slice(max(0, first), last))
        index = (slice(None), *selection)
        low = self.low[index].reshape(len(self.dims), -1).min(axis=1)
        high = self.high[index].reshape(len(self.dims), -1).max(axis=1)
        if outside:
            low = np.minimum(low, 0.0)
            high = np.maximum(high, 0.0)
        return {
            dim: (float(low[axis]), float(high[axis]))
            for axis, dim in enumerate(self.dims)
        }


def _block_extrema(block, block_info=None, *, origin=None, spacing=None, offset=None):
    """Smallest and largest displacement per component over one block.

    Comes back as one array of twice the components, the minima before the
    maxima, so a single pass over the field answers both. ``offset`` is where
    the array this block came from starts in the field, since the block is
    located against its own array rather than the field.
    """
    values = np.asarray(block, dtype=np.float64)
    if origin is not None:
        # A coordinates field holds the output position of each grid point
        # rather than the offset from it, so the grid point comes off first.
        if values is block:
            values = values.copy()
        location = block_info[0]["array-location"][1:]
        for axis, (start, stop) in enumerate(location):
            shape = [1] * (values.ndim - 1)
            shape[axis] = -1
            grid = origin[axis] + spacing[axis] * np.arange(
                start + offset[axis], stop + offset[axis], dtype=np.float64
            )
            values[axis] -= grid.reshape(shape)
    flat = values.reshape(values.shape[0], -1)
    extrema = np.concatenate([flat.min(axis=1), flat.max(axis=1)])
    return extrema.reshape((2 * values.shape[0],) + (1,) * (values.ndim - 1))


def field_displacement_bound(
    transform: Displacements | Coordinates,
    field: NgffImage,
    dims: Sequence[str],
    window: Sequence[tuple[int, int]] | None = None,
) -> FieldBound:
    """Bound the displacement of every chunk of ``field``, one chunk at a time.

    One pass over the field, reading a chunk and keeping two numbers per
    component. The values are not held: what comes back is a few floats per
    chunk, which is what lets a caller size its reads against a field that
    does not fit in memory.

    :param transform: The ``displacements`` or ``coordinates`` transform the
        field belongs to.
    :type  transform: Displacements | Coordinates
    :param field: The field image, as :func:`field_image` returns it.
    :type  field: NgffImage
    :param dims: The spatial axis names of the input coordinate system, in
        RFC-5 (Zarr) order.
    :type  dims: Sequence[str]
    :param window: The field indices a caller will ask about, as
        :func:`field_window` returns them. The pass then covers the chunks
        that window touches and no others, which is what a grid smaller than
        the field it is defined on saves. Defaults to the whole field.
    :type  window: Sequence[tuple[int, int]] | None
    :return: The per-chunk range, keyed by the field's own indices.
    :rtype: FieldBound
    """
    dims = tuple(dims)
    data = field.data
    origin = spacing = None
    if transform.type == "coordinates":
        origin = np.array([float(field.translation[dim]) for dim in dims])
        spacing = np.array([float(field.scale[dim]) for dim in dims])

    if not hasattr(data, "chunks") or data.chunks is None:
        values = np.asarray(data)
        info = [{"array-location": [(0, size) for size in values.shape]}]
        extrema = _block_extrema(
            values, info, origin=origin, spacing=spacing, offset=(0,) * len(dims)
        )
        offsets = tuple(np.array([0]) for _ in dims)
    else:
        import dask.array as da

        # The component axis is bounded as a whole: a block holding a subset
        # of the components would report a range for the wrong ones.
        data = data.rechunk({0: -1})
        # Cut on chunk borders, so the pass drops whole chunks rather than
        # reading one to use part of it, and every kept chunk keeps its own
        # first index.
        starts = [
            np.concatenate([[0], np.cumsum(sizes)[:-1]]) for sizes in data.chunks[1:]
        ]
        if window is None:
            window = [(0, int(size)) for size in data.shape[1:]]
        keep = []
        offsets = []
        for axis, (begin, end) in enumerate(window):
            first = max(0, int(np.searchsorted(starts[axis], begin, side="right") - 1))
            last = int(np.searchsorted(starts[axis], max(begin, end - 1), side="right"))
            high = int(
                starts[axis][last] if last < len(starts[axis]) else data.shape[1 + axis]
            )
            keep.append(slice(int(starts[axis][first]), high))
            offsets.append(starts[axis][first:last])
        offset = tuple(int(kept[0]) for kept in offsets)
        offsets = tuple(offsets)
        # Full slices are normalized away, so an unrestricted pass is the same
        # array and the same one code path.
        data = data[(slice(None), *keep)]
        counts = tuple((1,) * len(sizes) for sizes in data.chunks[1:])
        extrema = np.asarray(
            da.map_blocks(
                _block_extrema,
                data,
                origin=origin,
                spacing=spacing,
                offset=offset,
                dtype=np.float64,
                chunks=((2 * data.shape[0],), *counts),
                # The reductions have no identity on the zero-size block the
                # meta probe hands in; naming the meta skips the probe.
                meta=np.empty((0,) * data.ndim, dtype=np.float64),
            ).compute()
        )
    return FieldBound(
        dims=dims,
        low=extrema[: len(dims)],
        high=extrema[len(dims) :],
        offsets=offsets,
    )


def field_window(
    field: NgffImage,
    dims: Sequence[str],
    translation: Mapping[str, float],
    scale: Mapping[str, float],
    shape: Sequence[int],
    margin: int = 1,
) -> tuple[tuple[tuple[int, int], ...], bool]:
    """The field indices a grid of ``shape`` at ``translation`` reads.

    The field is evaluated at the grid's own points, so the window is that
    grid's extent expressed in field indices, whatever the displacement is:
    what a displacement sizes is the *moving* read, which
    :class:`FieldBound` answers.

    :param field: The field image.
    :type  field: NgffImage
    :param dims: The spatial axis names, in RFC-5 order.
    :type  dims: Sequence[str]
    :param translation: The grid's translation, keyed by dimension.
    :type  translation: Mapping[str, float]
    :param scale: The grid's scale, keyed by dimension.
    :type  scale: Mapping[str, float]
    :param shape: The grid's extent, in ``dims`` order.
    :type  shape: Sequence[int]
    :param margin: Lattice points kept beyond the bracketing pair, so that
        linear interpolation at a point on the boundary reads the same values
        it reads from the whole field.
    :type  margin: int
    :return: ``(start, stop)`` per axis, in ``dims`` order, clamped to the
        field, and whether the grid also has points beyond the field, which
        ITK displaces by nothing. :meth:`FieldBound.over` takes the second as
        its ``outside``.
    :rtype: tuple[tuple[tuple[int, int], ...], bool]
    """
    window = []
    outside = False
    for axis, dim in enumerate(dims):
        extent = int(field.data.shape[1 + axis])
        if shape[axis] == 0:
            window.append((0, 0))
            continue
        corners = [
            (translation[dim] + scale[dim] * index - field.translation[dim])
            / field.scale[dim]
            for index in (0, shape[axis] - 1)
        ]
        if min(corners) < 0 or max(corners) > extent - 1:
            outside = True
        low = int(np.floor(min(corners))) - margin
        high = int(np.ceil(max(corners))) + margin + 1
        window.append((max(0, min(low, extent)), max(0, min(high, extent))))
    return tuple(window), outside


def ngff_displacement_field_to_itk_transform(
    transform: Displacements | Coordinates,
    field,
    dims: Sequence[str],
    *,
    fixed: NgffImage | None = None,
    moving: NgffImage | None = None,
) -> list:
    """Convert an RFC-5 ``displacements`` or ``coordinates`` transform to ITK.

    The counterpart of :func:`itk_displacement_field_to_ngff_transform`, and
    what :func:`~ngff_zarr.ngff_transform_to_itk_transform` calls when handed
    a field transform with its field. The field is the image stored at
    ``transform.path``; load it with
    ``from_ome_zarr(f"{store}/{transform.path}")``.

    A ``coordinates`` field holds the absolute output position of each grid
    point where a ``displacements`` field holds the offset from it. The two
    differ by the position of the grid point itself, so both reach ITK as one
    ``DisplacementField``: ITK has no absolute-coordinate transform.

    :param transform: The ``displacements`` or ``coordinates`` transform.
    :type  transform: Displacements | Coordinates
    :param field: The field image: an ``NgffImage`` whose component axis is
        the one with ``axes_types`` ``displacement`` (``coordinate`` for a
        ``coordinates`` transform), followed by ``dims`` in order; or an
        ``NgffMultiscales``, whose finest level is used.
    :param dims: The spatial axis names of the input coordinate system, in
        RFC-5 (Zarr) order.
    :type  dims: Sequence[str]
    :param fixed: The fixed and moving images the field relates; see
        :func:`itk_displacement_field_to_ngff_transform`. The field's own
        orientation, if any, must be the fixed image's.
    :type  fixed: NgffImage, optional
    :param moving: See ``fixed``. Pass both or neither.
    :type  moving: NgffImage, optional
    :return: A single-entry ITK-Wasm ``TransformList`` of parameterization
        ``DisplacementField``. ``itk.transform_from_dict`` turns it into a
        native ``itk.DisplacementFieldTransform``.
    :rtype: list[itkwasm.Transform]
    :raises ValueError: If the field's axes are not the component axis followed
        by ``dims``, or if its orientation is not the fixed image's.
    """
    from itkwasm import FloatTypes, TransformParameterizations, TransformType
    from itkwasm import Transform as ItkWasmTransform

    from .itk_transform_to_ngff_transform import _itk_axis_order
    from .resample_bounding_box import _itk_direction

    dims = _check_dims(dims)
    absolute = transform.type == "coordinates"
    field = field_image(transform, field, dims)

    data = field.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)
    dimension = len(dims)

    itk_dims = _itk_axis_order(dims)
    canonical = list(reversed(itk_dims))
    # (c, *dims) with components in dims order -> [z][y][x][c] with ITK
    # components. The transpose is a view; indexing the component axis, which
    # the transpose has moved last, is the one copy.
    arranged = np.transpose(data, [1 + dims.index(dim) for dim in canonical] + [0])
    displacements = arranged[..., [dims.index(dim) for dim in itk_dims]]

    spacing = np.array([float(field.scale[dim]) for dim in itk_dims])
    translation = np.array([float(field.translation[dim]) for dim in itk_dims])
    own_direction = _itk_direction(field, itk_dims)

    frames = _frames(fixed, moving, dims)
    if frames is None:
        check_unoriented_field(field, dims)
        frames = _unoriented_frames(dimension)
    elif not _is_identity(own_direction) and not np.allclose(
        own_direction, frames.direction_in
    ):
        msg = (
            "the field's orientation is not the fixed image's: it gives "
            f"{own_direction.tolist()} where the fixed image gives "
            f"{frames.direction_in.tolist()}"
        )
        raise ValueError(msg)

    direction = frames.direction_in
    origin = direction @ (translation - frames.origin_in) + frames.origin_in

    direction_out, shift_matrix, shift_vector = _frame_terms(frames)
    # ``shift_matrix`` is ``M - I``, so subtracting it turns an RFC-5
    # displacement into an ITK vector. A coordinates field holds ``q + d``
    # rather than ``d``, so subtracting ``M`` instead removes the grid point
    # along with the frame term, in one pass over the grid.
    grid_matrix = shift_matrix + np.eye(dimension) if absolute else shift_matrix
    shift = _grid_shift(
        displacements.shape[:-1], translation, spacing, grid_matrix, shift_vector
    )
    vectors = displacements if shift is None else displacements - shift
    if not _is_identity(direction_out):
        vectors = vectors @ direction_out.T

    if transform.interpolation not in (None, "linear"):
        warnings.warn(
            f"the {transform.type} transform asks for '{transform.interpolation}' "
            "interpolation; ITK interpolates a displacement field linearly. RFC-5 "
            "leaves the choice to the consumer.",
            stacklevel=2,
        )

    if vectors.dtype == np.float32:
        value_type = FloatTypes.Float32
    else:
        value_type = FloatTypes.Float64
        vectors = vectors.astype(np.float64, copy=False)
    size = np.array(displacements.shape[:-1][::-1], dtype=np.float64)
    fixed_parameters = np.concatenate(
        [size, origin, spacing, direction.ravel(order="C")]
    ).astype(np.float64)
    parameters = np.ascontiguousarray(vectors).ravel(order="C")
    transform_type = TransformType(
        transformParameterization=TransformParameterizations.DisplacementField,
        parametersValueType=value_type,
        inputDimension=dimension,
        outputDimension=dimension,
    )
    return [
        ItkWasmTransform(
            transformType=transform_type,
            numberOfFixedParameters=len(fixed_parameters),
            numberOfParameters=len(parameters),
            fixedParameters=fixed_parameters,
            parameters=parameters,
            name="DisplacementFieldTransform",
        )
    ]


def _fields_entry(
    transform: Displacements | Coordinates, fields: Mapping[str, object] | None
):
    """The field ``fields`` holds for ``transform``, with a message otherwise."""
    if not fields or transform.path not in fields:
        available = sorted(fields) if fields else []
        msg = (
            f"the {transform.type} transform points at '{transform.path}', but no "
            f"field was passed for it (fields given: {available}). Load it with "
            f'from_ome_zarr(f"{{store}}/{transform.path}") and pass '
            f"fields={{'{transform.path}': field}}."
        )
        raise ValueError(msg)
    return fields[transform.path]
