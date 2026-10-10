# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A moving-image box that every point of a block lands in, for an ITK transform list.

The box is carried through the list one stage at a time, ITK's last entry first.
A linear stage maps the box's corners. A B-spline stage samples the transform on a
lattice over the box and widens the lattice's box by the interpolation error
bound, read off the second differences of the control points, then intersects
the result with the signed range of the control points; where the box leaves the
B-spline's domain, ITK displaces nothing, so the signed range with zero in it is
taken alone. A displacement-field stage takes the box of the landing points of
the samples under the box. Every rule holds because a point moves by a convex
combination of the samples or control points around it.

The arrays are handled through the array API namespace of the coefficients, so
a transform whose parameters live in another array library goes through the same
code.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np

from .displacement_field_transform import _ON_LATTICE

#: Upper bound on lattice points per stage; the step grows past it.
_MAX_POINTS = 1 << 14


def _namespace(array):
    get = getattr(array, "__array_namespace__", None)
    return get() if get is not None else np


def _lattice_geometry(entry, dimension: int):
    """``(size, origin, spacing, direction)`` of a grid-based stage, or ``None``."""
    fixed = np.asarray(
        [] if entry.fixedParameters is None else entry.fixedParameters, dtype=float
    )
    if entry.parameters is None or fixed.size != dimension * (3 + dimension):
        return None
    size = fixed[:dimension].astype(int)
    origin = fixed[dimension : 2 * dimension]
    spacing = fixed[2 * dimension : 3 * dimension]
    direction = fixed[3 * dimension :].reshape(dimension, dimension)
    if np.any(size < 1) or np.any(spacing <= 0):
        return None
    return size, origin, spacing, direction


def _corners(low, high):
    """The 2^d corners of a box, as a (2^d, d) array."""
    return np.array(list(itertools.product(*zip(low, high))), dtype=float)


def _box_of(points):
    return np.min(points, axis=0), np.max(points, axis=0)


def _index_box(low, high, origin, spacing, direction):
    """A physical box in a grid's continuous index space, as its corners' box."""
    inverse = np.linalg.inv(direction)
    corners = (_corners(low, high) - origin) @ inverse.T / spacing
    return _box_of(corners)


def bspline_weights(xp, t):
    """ITK's cubic B-spline weights for the four support points of a cell.

    ``t`` is the fractional position in the cell, in ``[0, 1)``. The support
    starts one control point below the cell.
    """
    one = 1 - t
    return (
        one * one * one / 6,
        (3 * t * t * t - 6 * t * t + 4) / 6,
        (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6,
        t * t * t / 6,
    )


def bspline_displacement(coefficients, size, index):
    """Displacement of points at continuous grid ``index`` through a cubic B-spline.

    :param coefficients: ``(dimension, *size[::-1])`` control-point vectors, one
        array per output component, laid out z, y, x like ITK's parameters.
    :param size: The grid size, x first.
    :param index: ``(n, dimension)`` continuous indices, x first.
    :return: ``(n, dimension)`` displacements; zero outside ITK's valid region,
        where ITK returns the point unchanged.
    """
    xp = _namespace(coefficients)
    dimension = len(size)
    size = np.asarray(size, dtype=int)
    # ITK's valid region: one control point in from the low edge, two from the
    # high edge, the high limit itself nudged inside.
    inside = xp.ones(index.shape[0], dtype=xp.bool)
    for axis in range(dimension):
        column = index[:, axis]
        inside = inside & (column >= 1) & (column <= size[axis] - 2)
    cell = xp.floor(xp.clip(index, 1, size - 2 - 1e-9))
    fraction = xp.clip(index - cell, 0, 1)
    first = xp.astype(cell, xp.int64) - 1
    weights = [bspline_weights(xp, fraction[:, axis]) for axis in range(dimension)]
    # Linear index of a control point: x fastest, matching the parameter layout.
    strides = [int(np.prod(size[:axis])) for axis in range(dimension)]
    flat = [xp.reshape(coefficients[c], (-1,)) for c in range(dimension)]
    out = [xp.zeros(index.shape[0], dtype=flat[0].dtype) for _ in range(dimension)]
    for tap in range(4**dimension):
        weight = None
        offset = None
        rest = tap
        for axis in range(dimension):
            k = rest % 4
            rest //= 4
            w = weights[axis][k]
            weight = w if weight is None else weight * w
            term = (first[:, axis] + k) * strides[axis]
            offset = term if offset is None else offset + term
        for c in range(dimension):
            out[c] = out[c] + weight * xp.take(flat[c], offset)
    stacked = xp.stack(out, axis=1)
    return xp.where(inside[:, None], stacked, xp.zeros_like(stacked))


def _second_differences(block, axis_i, axis_m):
    """``max |Δ_i Δ_m c|`` over an array, along array axes."""
    if axis_i == axis_m:
        d = np.diff(block, 2, axis=axis_i)
    else:
        d = np.diff(np.diff(block, axis=axis_i), axis=axis_m)
    return float(np.max(np.abs(d))) if d.size else 0.0


def _mapped_box(low, high, matrix, offset):
    """The box of a box's corners through a linear map."""
    return _box_of(_corners(low, high) @ matrix.T + offset)


def _bspline_stage(entry, dimension: int, low, high, matrix, offset, voxel):
    """The box through a cubic B-spline stage.

    ``low`` and ``high`` bound the block in the fixed image, ``matrix`` and
    ``offset`` are the linear stages applied since, so the lattice is laid
    over the block itself and mapped exactly, and the block's rotation does
    not widen the box.
    """
    geometry = _lattice_geometry(entry, dimension)
    if geometry is None:
        return None
    size, origin, spacing, direction = geometry
    parameters = np.asarray(entry.parameters)
    if parameters.size != dimension * int(np.prod(size)):
        return None
    coefficients = parameters.reshape(dimension, *size[::-1])
    box_low, box_high = _mapped_box(low, high, matrix, offset)
    index_low, index_high = _index_box(box_low, box_high, origin, spacing, direction)
    inside = bool(np.all(index_low >= 1) and np.all(index_high <= size - 2))

    # The signed range of the control points whose support meets the box.
    first = np.clip(np.floor(index_low).astype(int) - 1, 0, size - 1)
    last = np.clip(np.floor(index_high).astype(int) + 2, 0, size - 1)
    band = tuple(
        slice(first[axis], last[axis] + 1) for axis in reversed(range(dimension))
    )
    range_low = np.array([coefficients[c][band].min() for c in range(dimension)])
    range_high = np.array([coefficients[c][band].max() for c in range(dimension)])
    if not inside:
        range_low = np.minimum(range_low, 0)
        range_high = np.maximum(range_high, 0)
        return box_low + range_low, box_high + range_high

    # The second derivative of the displacement along a fixed axis, bounded by
    # the second differences of the control points in the band: the index of a
    # fixed point moves by ``derivative`` per unit of fixed coordinate.
    inverse = np.linalg.inv(direction)
    derivative = inverse @ matrix / spacing[:, None]
    factor = np.zeros((dimension, dimension))
    for c in range(dimension):
        block = coefficients[c][band]
        bound = np.zeros((dimension, dimension))
        for i in range(dimension):
            for m in range(i, dimension):
                value = _second_differences(block, dimension - 1 - i, dimension - 1 - m)
                bound[i, m] = bound[m, i] = value
        factor[c] = np.diag(np.abs(derivative).T @ bound @ np.abs(derivative))

    # The lattice step: half a control cell at most, the voxel at least, and
    # in between the coarsest whose margin stays within a voxel.
    cell = np.abs(np.linalg.inv(derivative)) @ np.ones(dimension)
    nominal = np.maximum(cell / 2, voxel)
    budget = float(np.min(voxel))
    worst = np.max(factor, axis=0)
    total = float(np.sum(nominal**2 * worst))
    scale = 1.0 if total <= 0 else min(1.0, math.sqrt(8 * budget / total))
    step = np.maximum(voxel, scale * nominal)
    span = high - low
    count = np.maximum(1, np.ceil(span / step - 1e-9)).astype(int)
    while int(np.prod(count + 1)) > _MAX_POINTS:
        count = np.maximum(1, count // 2)
    axes = [
        np.linspace(low[axis], high[axis], count[axis] + 1) for axis in range(dimension)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
    actual = span / count
    mapped = points @ matrix.T + offset
    index = (mapped - origin) @ inverse.T / spacing
    landing = mapped + bspline_displacement(coefficients, size, index)
    lattice_low, lattice_high = _box_of(landing)
    # Interpolation error between lattice points: s^2/8 times the bound.
    margin = (actual**2 / 8) @ factor.T
    return (
        np.maximum(lattice_low - margin, box_low + range_low),
        np.minimum(lattice_high + margin, box_high + range_high),
    )


def _field_stage(entry, dimension: int, low, high):
    """The box through a displacement-field stage."""
    geometry = _lattice_geometry(entry, dimension)
    if geometry is None:
        return None
    size, origin, spacing, direction = geometry
    parameters = np.asarray(entry.parameters)
    if parameters.size != dimension * int(np.prod(size)):
        return None
    field = parameters.reshape(*size[::-1], dimension)
    index_low, index_high = _index_box(low, high, origin, spacing, direction)
    # A box on the field's own lattice computes to within rounding of whole
    # indices; without the tolerance each side would pick up a sample the box
    # does not reach.
    first = np.clip(np.floor(index_low + _ON_LATTICE).astype(int), 0, size - 1)
    last = np.clip(np.ceil(index_high - _ON_LATTICE).astype(int), 0, size - 1)
    outside = bool(
        np.any(index_low < -_ON_LATTICE) or np.any(index_high > size - 1 + _ON_LATTICE)
    )
    window = tuple(
        slice(first[axis], last[axis] + 1) for axis in reversed(range(dimension))
    )
    samples = field[window].reshape(-1, dimension)
    grids = np.meshgrid(
        *[
            np.arange(first[axis], last[axis] + 1)
            for axis in reversed(range(dimension))
        ],
        indexing="ij",
    )
    index = np.stack(grids[::-1], axis=-1).reshape(-1, dimension).astype(float)
    positions = index * spacing @ direction.T + origin
    # Two boxes hold every point of the block: the landing points of the samples
    # around it, exact when the field is at the block's resolution, and the box
    # plus the signed range of those samples, exact when the field is coarse and
    # smooth. Their intersection holds every point too.
    land_low, land_high = _box_of(positions + samples)
    range_low, range_high = _box_of(samples)
    if outside:
        # ITK displaces a point up to half a sample past the lattice by what the
        # nearest sample holds, and any further point by nothing.
        half = np.abs(direction) @ spacing / 2
        land_low = np.minimum(low, land_low - half)
        land_high = np.maximum(high, land_high + half)
        range_low = np.minimum(range_low, 0)
        range_high = np.maximum(range_high, 0)
    return (
        np.maximum(land_low, low + range_low),
        np.minimum(land_high, high + range_high),
    )


def transform_box(transform_list: Sequence, dimension: int, low, high, voxel):
    """The box in the last stage's output space, or ``None`` for an unsupported stage.

    ITK applies the last entry of a list first. Linear stages are composed and
    applied to the points a B-spline stage samples, so a rotation ahead of a
    B-spline does not widen its box; they map the box's corners otherwise.
    """
    from .itk_transform_to_ngff_transform import (
        _matrix_offset_from_itkwasm,
        _parameterization_name,
    )

    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    matrix = np.eye(dimension)
    offset = np.zeros(dimension)
    for entry in reversed(list(transform_list)):
        name = _parameterization_name(entry.transformType)
        if name == "Composite":
            # A parameterless 'Composite' entry is a pipeline grouping header
            # or a nested composite whose children the serialization dropped
            # (InsightSoftwareConsortium/ITK#6792); the two cannot be told
            # apart, so the pipeline sizes such a list.
            return None
        if name == "BSpline":
            result = _bspline_stage(entry, dimension, low, high, matrix, offset, voxel)
        elif name == "DisplacementField":
            result = _field_stage(
                entry, dimension, *_mapped_box(low, high, matrix, offset)
            )
        else:
            try:
                stage, shift = _matrix_offset_from_itkwasm(entry, dimension)
            except (NotImplementedError, ValueError, ImportError):
                return None
            stage = np.asarray(stage, dtype=float)
            matrix = stage @ matrix
            offset = stage @ offset + np.asarray(shift, dtype=float)
            continue
        if result is None:
            return None
        low, high = result
        matrix = np.eye(dimension)
        offset = np.zeros(dimension)
    return _mapped_box(low, high, matrix, offset)
