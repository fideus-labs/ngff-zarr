# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Bridge RFC-5 coordinate transformations to ITK transforms.

RFC-5 and ITK describe the same affine geometry with two different
conventions, and the differences are silent rather than loud: getting one
wrong yields a plausible transform that is simply in the wrong place.

Axis order
    RFC-5 orders transformation parameters the same way the Zarr array is
    ordered -- parameter ``i`` belongs to coordinate-system axis ``i``, so a
    ``zyx`` image has ``z`` first. ITK orders points fastest-axis-first, so
    the same point is ``xyz``. Writing ``R`` for the axis-reversal
    permutation, an RFC-5 affine ``q = M p + b`` becomes ``A = R M R`` and
    ``t = R b`` in ITK.

Composition order
    An RFC-5 ``sequence`` applies its first entry first. An ITK transform
    list applies its *last* entry first. Rather than emit a list and rely on
    that inversion, this module composes the chain into a single matrix here,
    where the order is explicit and testable.

Both conventions place the pixel center at the integer index, so no
half-pixel correction is involved.
"""

from collections.abc import Sequence

import numpy as np

from .v06.zarr_metadata import (
    Affine,
    Identity,
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
            msg = (
                f"rotation transformation is {rotation.shape[0]}x{rotation.shape[1]} "
                f"but the coordinate system has {ndim} axes"
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
            msg = (
                f"affine transformation is {affine.shape[0]}x{affine.shape[1]} but "
                f"a coordinate system with {ndim} axes requires "
                f"{ndim}x{ndim + 1} (the translation is the last column)"
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
        " cannot be converted to an ITK transform. Only identity, scale, "
        "translation, rotation, affine and sequences of them describe a linear "
        "mapping that ITK can represent as a single affine transform."
    )
    raise NotImplementedError(msg)


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


def ngff_transform_to_itk_matrix(
    transform: Transform,
    dims: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an RFC-5 transformation to an ITK matrix and offset.

    :param transform: An RFC-5 (OME-Zarr v0.6) coordinate transformation. It
        must describe a linear mapping -- ``identity``, ``scale``,
        ``translation``, ``rotation``, ``affine``, or a ``sequence`` of those.
    :type  transform: Transform

    :param dims: The axis names of the coordinate system the transformation is
        defined on, in RFC-5 (Zarr) order, e.g. ``("z", "y", "x")``.
    :type  dims: Sequence[str]

    :return: ``(matrix, offset)`` for the spatial axes only, in ITK
        (fastest-axis-first) order.
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

    # RFC-5 (Zarr) order -> ITK (fastest-axis-first) order.
    reversal = np.eye(len(spatial_indices))[::-1]
    return reversal @ matrix @ reversal, reversal @ offset


def ngff_transform_to_itk_transform(
    transform: Transform,
    dims: Sequence[str],
) -> list:
    """Convert an RFC-5 transformation to an ITK-Wasm transform list.

    The transformation is collapsed into a single ``Affine`` entry, so the
    result is independent of ITK's own list-composition order.

    :param transform: An RFC-5 (OME-Zarr v0.6) coordinate transformation
        describing a linear mapping.
    :type  transform: Transform

    :param dims: The axis names of the coordinate system the transformation is
        defined on, in RFC-5 (Zarr) order.
    :type  dims: Sequence[str]

    :return: A single-entry ITK-Wasm ``TransformList``.
    :rtype: list[itkwasm.Transform]
    """
    from itkwasm import (
        FloatTypes,
        TransformParameterizations,
        TransformType,
    )
    from itkwasm import (
        Transform as ItkTransform,
    )

    matrix, offset = ngff_transform_to_itk_matrix(transform, dims)
    dimension = offset.shape[0]

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
