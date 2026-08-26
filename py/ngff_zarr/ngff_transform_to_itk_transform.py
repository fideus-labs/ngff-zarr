# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Bridge RFC-5 coordinate transformations to ITK transforms.

RFC-5 and ITK describe the same affine geometry with two different
conventions, and the differences are silent rather than loud: getting one
wrong yields a plausible transform that is simply in the wrong place.

Axis order
    RFC-5 orders transformation parameters the same way the Zarr array is
    ordered -- parameter ``i`` belongs to coordinate-system axis ``i``, so a
    ``zyx`` image has ``z`` first. ITK orders points fastest-axis-first by
    name: x, then y, then z. Writing ``P`` for the permutation sending an
    ITK-order vector to the ``dims``-order vector, an RFC-5 affine
    ``q = M p + b`` becomes ``A = P^T M P`` and ``t = P^T b`` in ITK. For
    the canonical ``zyx`` order ``P`` is the axis reversal.

Composition order
    An RFC-5 ``sequence`` applies its first entry first. An ITK transform
    list applies its *last* entry first. Rather than emit a list and rely on
    that inversion, this module composes the chain into a single matrix here,
    where the order is explicit and testable.

Both conventions place the pixel center at the integer index, so no
half-pixel correction is involved.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .v06.zarr_metadata import (
    Affine,
    Bijection,
    ByDimension,
    ByDimensionItem,
    Coordinates,
    Displacements,
    Identity,
    MapAxis,
    Rotation,
    Scale,
    Transform,
    TransformSequence,
    Translation,
)

_SPATIAL_DIMS = ("x", "y", "z")


def _homogeneous_from_transform(transform: Transform, ndim: int) -> np.ndarray:
    """Collapse one RFC-5 transformation into an ``(ndim+1, ndim+1)`` matrix.

    The matrix is in RFC-5 (Zarr) axis order and acts on column vectors in
    homogeneous coordinates.
    """
    if isinstance(transform, Identity):
        return np.eye(ndim + 1)

    if isinstance(transform, Scale):
        if len(transform.scale) != ndim:
            msg = (
                f"scale transformation has {len(transform.scale)} parameters "
                f"but the coordinate system has {ndim} axes"
            )
            raise ValueError(msg)
        matrix = np.eye(ndim + 1)
        matrix[:ndim, :ndim] = np.diag(np.asarray(transform.scale, dtype=float))
        return matrix

    if isinstance(transform, Translation):
        if len(transform.translation) != ndim:
            msg = (
                f"translation transformation has {len(transform.translation)} "
                f"parameters but the coordinate system has {ndim} axes"
            )
            raise ValueError(msg)
        matrix = np.eye(ndim + 1)
        matrix[:ndim, ndim] = np.asarray(transform.translation, dtype=float)
        return matrix

    if isinstance(transform, Rotation):
        rotation = _as_matrix(transform.rotation, transform.path, "rotation")
        if rotation.shape != (ndim, ndim):
            # Formatted from the whole shape tuple: a flat parameter array is
            # 1-D, and indexing a fixed axis would raise IndexError from
            # inside the message meant to explain the problem.
            shape = "x".join(str(extent) for extent in rotation.shape)
            msg = (
                f"rotation transformation is {shape} but the coordinate "
                f"system has {ndim} axes"
            )
            raise ValueError(msg)
        matrix = np.eye(ndim + 1)
        matrix[:ndim, :ndim] = rotation
        return matrix

    if isinstance(transform, Affine):
        affine = _as_matrix(transform.affine, transform.path, "affine")
        # RFC-5 stores the upper M x (N+1) block of a homogeneous matrix: the
        # rotation/scale/shear part followed by the translation as the last
        # column, with the trailing [0 ... 0 1] row omitted.
        if affine.shape != (ndim, ndim + 1):
            shape = "x".join(str(extent) for extent in affine.shape)
            msg = (
                f"affine transformation is {shape} but a coordinate system "
                f"with {ndim} axes requires {ndim}x{ndim + 1} (the "
                "translation is the last column)"
            )
            raise ValueError(msg)
        matrix = np.eye(ndim + 1)
        matrix[:ndim, :] = affine
        return matrix

    if isinstance(transform, TransformSequence):
        if not transform.transformations:
            msg = "sequence transformation has no transformations"
            raise ValueError(msg)
        # RFC-5 applies the first entry first, so the matrix product runs
        # right-to-left over the list.
        matrix = np.eye(ndim + 1)
        for sub_transform in transform.transformations:
            matrix = _homogeneous_from_transform(sub_transform, ndim) @ matrix
        return matrix

    if isinstance(transform, MapAxis):
        if len(transform.mapAxis) != ndim:
            msg = (
                f"mapAxis transformation permutes {len(transform.mapAxis)} axes "
                f"but the coordinate system has {ndim}"
            )
            raise ValueError(msg)
        matrix = np.zeros((ndim + 1, ndim + 1))
        matrix[ndim, ndim] = 1.0
        # The value at position i names the input axis that becomes output
        # axis i, so the row for output i selects that input.
        for output_axis, input_axis in enumerate(transform.mapAxis):
            matrix[output_axis, input_axis] = 1.0
        return matrix

    if isinstance(transform, ByDimension):
        return _homogeneous_from_by_dimension(transform, ndim)

    if isinstance(transform, Bijection):
        # The forward direction is the mapping; RFC-5 keeps the inverse
        # alongside it so a reader need not invert the forward one, which is
        # what ITK does for an affine anyway.
        return _homogeneous_from_transform(transform.forward, ndim)

    # ``ngff_zarr.Scale`` and friends are the v0.4 dataclasses, which carry the
    # same parameters under the same names but do not share the v0.6 base
    # class. Accept them rather than making callers hunt for the v0.6 twin.
    transform_type = getattr(transform, "type", None)
    if transform_type == "identity":
        return np.eye(ndim + 1)
    if transform_type == "scale" and hasattr(transform, "scale"):
        return _homogeneous_from_transform(Scale(scale=transform.scale), ndim)
    if transform_type == "translation" and hasattr(transform, "translation"):
        return _homogeneous_from_transform(
            Translation(translation=transform.translation), ndim
        )

    msg = (
        f"transformation type '{transform_type or type(transform).__name__}'"
        " cannot be converted to an ITK transform. identity, scale, "
        "translation, rotation, affine, mapAxis, byDimension, bijection and "
        "sequences of them describe a linear mapping that ITK can represent as "
        "a single affine transform; displacements and coordinates convert to a "
        "displacement field, given the field they point at."
    )
    raise NotImplementedError(msg)


def _homogeneous_from_by_dimension(transform: ByDimension, ndim: int) -> np.ndarray:
    """Assemble a byDimension transformation into one homogeneous matrix.

    Each item is a lower-dimensional transformation between two subsets of
    axes, so its own matrix is written into the rows its ``outputAxes`` name
    and the columns its ``inputAxes`` name. Axes no item produces would leave
    a zero row, which collapses the image rather than transforming it, so a
    gap is refused here rather than resampled.
    """
    matrix = np.zeros((ndim + 1, ndim + 1))
    matrix[ndim, ndim] = 1.0
    claimed: dict[int, int] = {}
    for index, item in enumerate(transform.transformations):
        block, offset = _by_dimension_item_block(item, ndim)
        for row, output_axis in enumerate(item.outputAxes):
            # Two items writing the same row would leave the second overwriting
            # the first and the union below none the wiser, so the mapping would
            # come back silently wrong rather than refused.
            if output_axis in claimed:
                msg = (
                    f"byDimension items {claimed[output_axis]} and {index} both "
                    f"produce output axis {output_axis}; every output axis must "
                    "be produced by exactly one item"
                )
                raise ValueError(msg)
            claimed[output_axis] = index
            for column, input_axis in enumerate(item.inputAxes):
                matrix[output_axis, input_axis] = block[row, column]
            matrix[output_axis, ndim] = offset[row]

    produced = transform.produced_outputAxes
    missing = sorted(set(range(ndim)) - produced)
    if missing:
        msg = (
            f"byDimension transformation produces output axes {sorted(produced)}, "
            f"leaving {missing} of the {ndim} axes of the coordinate system "
            "unset; every output axis must be produced by exactly one item"
        )
        raise ValueError(msg)
    return matrix


def _by_dimension_item_block(
    item: ByDimensionItem, ndim: int
) -> tuple[np.ndarray, np.ndarray]:
    """One byDimension item as a matrix and offset over its own axes."""
    if len(item.inputAxes) != len(item.outputAxes):
        msg = (
            f"byDimension item of type '{item.transformation.type}' maps "
            f"{len(item.inputAxes)} input axes to {len(item.outputAxes)} "
            "output axes; only a square mapping converts to an ITK transform"
        )
        raise ValueError(msg)
    for axes in (item.inputAxes, item.outputAxes):
        beyond = [axis for axis in axes if axis >= ndim]
        if beyond:
            msg = (
                f"byDimension axis indices {beyond} exceed the {ndim} axes of "
                "the coordinate system"
            )
            raise ValueError(msg)
    if len(set(item.inputAxes)) != len(item.inputAxes):
        msg = f"byDimension input axes {item.inputAxes} name an axis twice"
        raise ValueError(msg)

    sub_ndim = len(item.inputAxes)
    homogeneous = _homogeneous_from_transform(item.transformation, sub_ndim)
    return homogeneous[:sub_ndim, :sub_ndim], homogeneous[:sub_ndim, sub_ndim]


def _as_matrix(values, path: str | None, field: str) -> np.ndarray:
    if values is None or len(values) == 0:
        if path is not None:
            msg = (
                f"{field} transformation stores its parameters at '{path}'. "
                "Reading matrix parameters from a Zarr array is not supported; "
                f"supply the '{field}' values inline."
            )
            raise NotImplementedError(msg)
        msg = f"{field} transformation has no parameters"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)


def _ngff_transform_to_itk_matrix(
    transform: Transform,
    dims: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an RFC-5 transformation to an ITK matrix and offset.

    Internal. :func:`ngff_transform_to_itk_transform` is the public entry
    point; nothing outside this package needs the raw numbers, since ITK is
    what the caller wants to hand them to. The tests use it as a fine probe on
    the axis and composition conventions.

    :param transform: An RFC-5 (OME-Zarr v0.6) coordinate transformation. It
        must describe a linear mapping -- ``identity``, ``scale``,
        ``translation``, ``rotation``, ``affine``, ``mapAxis``,
        ``byDimension``, ``bijection``, or a ``sequence`` of those.
    :type  transform: Transform

    :param dims: The axis names of the coordinate system the transformation is
        defined on, in RFC-5 (Zarr) order, e.g. ``("z", "y", "x")``.
    :type  dims: Sequence[str]

    :return: ``(matrix, offset)`` for the spatial axes only, in ITK
        (fastest-axis-first) order. ITK has no notion of a non-spatial axis, so
        a component acting purely on ``t`` or ``c`` -- a frame interval, say --
        is *projected away*. That is lossless for the spatial mapping, which is
        all an ITK transform describes, but the returned transform is not a
        faithful copy of the input. A component that *couples* the two kinds of
        axis is refused instead, because dropping it would move the image.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]

    :raises NotImplementedError: If the transformation is not linear.
    :raises ValueError: If the transformation couples spatial and non-spatial
        axes, or its parameters do not match ``dims``.
    """
    dims = tuple(dims)
    homogeneous = _homogeneous_from_transform(transform, len(dims))

    spatial_indices = [i for i, dim in enumerate(dims) if dim in _SPATIAL_DIMS]
    if not spatial_indices:
        msg = f"no spatial axes among dims {dims}"
        raise ValueError(msg)
    other_indices = [i for i in range(len(dims)) if i not in spatial_indices]

    # A transform that mixes spatial and non-spatial axes has no ITK
    # equivalent, and silently dropping the coupling would move the image.
    if other_indices:
        cross = homogeneous[np.ix_(spatial_indices, other_indices)]
        if np.any(cross != 0.0):
            msg = (
                "transformation couples spatial and non-spatial axes "
                f"{[dims[i] for i in other_indices]}; it cannot be expressed as "
                "an ITK spatial transform"
            )
            raise ValueError(msg)

    matrix = homogeneous[np.ix_(spatial_indices, spatial_indices)]
    offset = homogeneous[spatial_indices, len(dims)]

    # RFC-5 (`dims`) order -> ITK (fastest-axis-first) order. ITK orders
    # components by *name* (x, then y, then z), so the permutation is built
    # by name rather than by reversing `dims`, which is only equivalent for
    # the canonical (z, y, x).
    spatial = [dims[index] for index in spatial_indices]
    itk_dims = [dim for dim in _SPATIAL_DIMS if dim in spatial]
    permutation = np.zeros((len(spatial), len(spatial)))
    for row, dim in enumerate(spatial):
        permutation[row, itk_dims.index(dim)] = 1.0
    return permutation.T @ matrix @ permutation, permutation.T @ offset


def ngff_transform_to_itk_transform(
    transform: Transform,
    dims: Sequence[str],
    *,
    fields: Mapping[str, object] | None = None,
    fixed=None,
    moving=None,
) -> list:
    """Convert an RFC-5 transformation to an ITK-Wasm transform list.

    A linear transformation is collapsed into a single ``Affine`` entry, so
    the result is independent of ITK's own list-composition order. A
    ``displacements`` or ``coordinates`` transformation becomes a single
    ``DisplacementField`` entry built from the field passed in ``fields``.

    :param transform: An RFC-5 (OME-Zarr v0.6) coordinate transformation:
        a linear mapping -- ``identity``, ``scale``, ``translation``,
        ``rotation``, ``affine``, ``mapAxis``, ``byDimension``, ``bijection``,
        or a ``sequence`` of them -- or a ``displacements`` or ``coordinates``
        transformation.
    :type  transform: Transform

    :param dims: The axis names of the coordinate system the transformation is
        defined on, in RFC-5 (Zarr) order. Only the spatial axes take part.
    :type  dims: Sequence[str]

    :param fields: The field images a ``displacements`` or ``coordinates``
        transformation points at, keyed by its ``path``: an ``NgffImage``, or
        the ``NgffMultiscales`` read from ``f"{store}/{transform.path}"``.
        Required for those two, ignored otherwise.
    :type  fields: Mapping[str, NgffImage | NgffMultiscales], optional

    :param fixed: The fixed and moving images the transform relates. Passing
        both lets the conversion re-express the intrinsic-space mapping on ITK
        physical space, including the direction matrix derived from RFC-4
        anatomical orientation. Omitting them is exact only when neither image
        carries an anatomical orientation.
    :type  fixed: NgffImage, optional

    :param moving: See ``fixed``. Pass both or neither.
    :type  moving: NgffImage, optional

    :return: A single-entry ITK-Wasm ``TransformList``.
    :rtype: list[itkwasm.Transform]
    """
    if isinstance(transform, (Displacements, Coordinates)):
        from .displacement_field_transform import (
            _fields_entry,
            ngff_displacement_field_to_itk_transform,
        )

        # ITK has no notion of a non-spatial axis, and neither has a field:
        # take the same spatial subset the linear path below takes, so `dims`
        # can be the image's own dimension names on either branch.
        return ngff_displacement_field_to_itk_transform(
            transform,
            _fields_entry(transform, fields),
            [dim for dim in dims if dim in _SPATIAL_DIMS],
            fixed=fixed,
            moving=moving,
        )

    from itkwasm import (
        FloatTypes,
        TransformParameterizations,
        TransformType,
    )
    from itkwasm import (
        Transform as ItkTransform,
    )

    matrix, offset = _ngff_transform_to_itk_matrix(transform, dims)
    dimension = offset.shape[0]

    from .itk_transform_to_ngff_transform import (
        _change_of_frame,
        _check_frame_images,
        _frame_geometry,
        _inverse_direction,
    )

    spatial = [dim for dim in dims if dim in _SPATIAL_DIMS]
    if _check_frame_images(fixed, moving, spatial):
        # The intrinsic systems -> ITK physical space: phi_m . T . phi_f^-1,
        # which is the same change of frame with the directions inverted.
        itk_dims = [dim for dim in _SPATIAL_DIMS if dim in spatial]
        direction_fixed, direction_moving, origin_fixed, origin_moving = (
            _frame_geometry(fixed, moving, itk_dims)
        )
        matrix, offset = _change_of_frame(
            matrix,
            offset,
            _inverse_direction(direction_fixed),
            _inverse_direction(direction_moving),
            origin_fixed,
            origin_moving,
        )

    # ITK's MatrixOffsetTransformBase packs the row-major matrix followed by
    # the translation, with the center of rotation as the fixed parameters.
    # An RFC-5 affine has no center, so it stays at the origin.
    parameters = np.concatenate([matrix.ravel(order="C"), offset])
    transform_type = TransformType(
        transformParameterization=TransformParameterizations.Affine,
        parametersValueType=FloatTypes.Float64,
        inputDimension=dimension,
        outputDimension=dimension,
    )
    return [
        ItkTransform(
            transformType=transform_type,
            numberOfFixedParameters=dimension,
            numberOfParameters=len(parameters),
            fixedParameters=np.zeros(dimension, dtype=np.float64),
            parameters=parameters,
            name="AffineTransform",
        )
    ]
