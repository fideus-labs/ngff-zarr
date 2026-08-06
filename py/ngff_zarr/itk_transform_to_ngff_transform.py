# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Convert ITK transforms to RFC-5 coordinate transformations.

The inverse of :mod:`ngff_zarr.ngff_transform_to_itk_transform`. Its purpose
is persistence: a registration produces an ITK transform, and RFC-5 is where
that result belongs once it is written next to the image.

The same two conventions apply in reverse -- the spatial block and the offset
are reversed from ITK's fastest-axis-first order back to Zarr order, and ITK's
center of rotation is folded into the offset, since an RFC-5 affine has no
center:

    ITK:   y = A (x - c) + t + c
    RFC-5: y = A x + b          with  b = t + c - A c
"""

from collections.abc import Sequence

import numpy as np

from .v06.zarr_metadata import (
    Affine,
    Identity,
    Scale,
    Transform,
    TransformSequence,
    Translation,
)

_SPATIAL_DIMS = ("x", "y", "z")

#: itkwasm parameterizations this module decodes without help from ``itk``.
#: Everything else (Euler, Versor, Similarity, ...) packs angles or quaternions
#: rather than a matrix, so it is rebuilt through ``itk`` and probed instead.
_DIRECTLY_DECODED = frozenset({"Identity", "Translation", "Scale", "Affine"})


def _matrix_offset_by_probing(itk_transform, dimension: int):
    """Recover ``(matrix, offset)`` by evaluating the transform.

    ``offset = T(0)`` and ``matrix[:, j] = T(e_j) - T(0)``. This is exact for
    any linear transform and, unlike decoding ``GetParameters()``, does not
    depend on how the particular transform type packs its parameters -- an
    ``Euler3DTransform`` stores angles and a ``VersorRigid3DTransform`` a
    quaternion, but both answer ``TransformPoint`` the same way.
    """
    origin = [0.0] * dimension
    offset = np.asarray(itk_transform.TransformPoint(origin), dtype=float)
    matrix = np.zeros((dimension, dimension))
    for axis in range(dimension):
        basis = [0.0] * dimension
        basis[axis] = 1.0
        column = np.asarray(itk_transform.TransformPoint(basis), dtype=float)
        matrix[:, axis] = column - offset
    return matrix, offset


def _parameterization_name(transform_type) -> str:
    """The parameterization as a plain name, e.g. ``"Affine"``.

    ``itkwasm.TransformParameterizations`` mixes in ``str``, and since Python
    3.11 ``str()`` on such a member returns ``"TransformParameterizations.Affine"``
    rather than its value. Reading ``.value`` keeps a member and a plain string
    on the same footing.
    """
    parameterization = transform_type.transformParameterization
    return str(getattr(parameterization, "value", parameterization))


def _matrix_offset_from_itkwasm(entry, dimension: int):
    """Decode an ITK-Wasm transform, falling back to ``itk`` when needed."""
    transform_type = entry.transformType
    parameterization = _parameterization_name(transform_type)
    parameters = np.asarray(
        [] if entry.parameters is None else entry.parameters, dtype=float
    )
    fixed = np.asarray(
        [] if entry.fixedParameters is None else entry.fixedParameters, dtype=float
    )

    if parameterization == "Identity":
        return np.eye(dimension), np.zeros(dimension)
    if parameterization == "Translation":
        return np.eye(dimension), parameters[:dimension].copy()
    if parameterization == "Scale":
        return np.diag(parameters[:dimension]), np.zeros(dimension)
    if parameterization == "Affine":
        matrix = parameters[: dimension * dimension].reshape(dimension, dimension)
        translation = parameters[dimension * dimension :][:dimension]
        center = fixed[:dimension] if fixed.size >= dimension else np.zeros(dimension)
        # ITK applies the matrix about the center, so fold it into the offset.
        return matrix, translation + center - matrix @ center

    try:
        import itk
    except ImportError as error:
        msg = (
            f"cannot decode an ITK-Wasm '{parameterization}' transform without "
            "itk, because that parameterization stores angles or a quaternion "
            "rather than a matrix. Install the ngff-zarr[itk] extra, or pass an "
            "Affine transform."
        )
        raise ImportError(msg) from error

    from dataclasses import asdict

    rebuilt = itk.transform_from_dict(asdict(entry))
    if hasattr(rebuilt, "GetNthTransform") and rebuilt.GetNumberOfTransforms() == 1:
        rebuilt = rebuilt.GetNthTransform(0)
    return _matrix_offset_by_probing(rebuilt, dimension)


def _itk_matrix_offset(transform, dimension: int):
    """Reduce any supported ITK transform input to a single matrix and offset."""
    from itkwasm import Transform as ItkWasmTransform

    # A native itk.Transform, including the CompositeTransform Elastix returns.
    if hasattr(transform, "TransformPoint"):
        if hasattr(transform, "IsLinear") and not transform.IsLinear():
            msg = (
                "only linear ITK transforms can be expressed as an RFC-5 affine; "
                f"{type(transform).__name__} is not linear"
            )
            raise NotImplementedError(msg)
        return _matrix_offset_by_probing(transform, dimension)

    entries = (
        [transform] if isinstance(transform, ItkWasmTransform) else list(transform)
    )
    if not entries:
        msg = "transform list is empty"
        raise ValueError(msg)

    # An ITK transform list applies its last entry first, so the homogeneous
    # matrices multiply left to right in list order.
    total = np.eye(dimension + 1)
    for entry in entries:
        matrix, offset = _matrix_offset_from_itkwasm(entry, dimension)
        homogeneous = np.eye(dimension + 1)
        homogeneous[:dimension, :dimension] = matrix
        homogeneous[:dimension, dimension] = offset
        total = total @ homogeneous
    return total[:dimension, :dimension], total[:dimension, dimension]


def itk_transform_to_ngff_matrix(
    transform,
    dims: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an ITK transform to a matrix and offset in RFC-5 axis order.

    :param transform: An ``itk.Transform`` (including a ``CompositeTransform``),
        an ITK-Wasm ``Transform``, or an ITK-Wasm ``TransformList``.

    :param dims: The axis names of the coordinate system the result should be
        expressed on, in RFC-5 (Zarr) order, e.g. ``("z", "y", "x")``. Only the
        spatial axes take part in the transform.
    :type  dims: Sequence[str]

    :return: ``(matrix, offset)`` over the spatial axes, in Zarr order.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]

    :raises NotImplementedError: If the transform is not linear.
    """
    dims = tuple(dims)
    spatial = [dim for dim in dims if dim in _SPATIAL_DIMS]
    if not spatial:
        msg = f"no spatial axes among dims {dims}"
        raise ValueError(msg)

    matrix, offset = _itk_matrix_offset(transform, len(spatial))

    # ITK (fastest-axis-first) order -> RFC-5 (Zarr) order.
    reversal = np.eye(len(spatial))[::-1]
    return reversal @ matrix @ reversal, reversal @ offset


def itk_transform_to_ngff_transform(
    transform,
    dims: Sequence[str],
    simplify: bool = True,
) -> Transform:
    """Convert an ITK transform to an RFC-5 coordinate transformation.

    This is what lets a registration result be written into an OME-Zarr store:
    run the registration, convert the transform, and attach it to the
    multiscales metadata.

    :param transform: An ``itk.Transform`` (including the ``CompositeTransform``
        an Elastix registration returns), an ITK-Wasm ``Transform``, or an
        ITK-Wasm ``TransformList``.

    :param dims: The axis names of the coordinate system the transformation is
        expressed on, in RFC-5 (Zarr) order. Non-spatial axes (``t``, ``c``) are
        left untransformed.
    :type  dims: Sequence[str]

    :param simplify: Return the least expressive transformation that represents
        the mapping exactly -- ``identity``, ``translation``, ``scale``, or a
        ``sequence`` of scale and translation -- falling back to ``affine``.
        RFC-5 recommends this, and only these simpler forms are legal inside
        ``multiscales > datasets``. Set to ``False`` to always get an ``affine``.
    :type  simplify: bool

    :return: An RFC-5 coordinate transformation over ``dims``.
    :rtype: Transform

    :raises NotImplementedError: If the transform is not linear. A non-linear
        registration has no affine equivalent.
    """
    dims = tuple(dims)
    matrix, offset = itk_transform_to_ngff_matrix(transform, dims)

    spatial_indices = [index for index, dim in enumerate(dims) if dim in _SPATIAL_DIMS]
    ndim = len(dims)

    # Embed the spatial block into the full coordinate system, leaving any
    # non-spatial axis untouched.
    full = np.eye(ndim)
    full_offset = np.zeros(ndim)
    for row, row_index in enumerate(spatial_indices):
        for col, col_index in enumerate(spatial_indices):
            full[row_index, col_index] = matrix[row, col]
        full_offset[row_index] = offset[row]

    if simplify:
        simplified = _simplify(full, full_offset, ndim)
        if simplified is not None:
            return simplified

    # RFC-5 stores the upper M x (N+1) block: the matrix followed by the
    # translation as the last column.
    return Affine(affine=np.hstack([full, full_offset.reshape(-1, 1)]).tolist())


def _simplify(matrix: np.ndarray, offset: np.ndarray, ndim: int) -> Transform | None:
    """Return a less expressive equivalent transformation, or ``None``."""
    is_identity = np.array_equal(matrix, np.eye(ndim))
    is_diagonal = np.array_equal(matrix, np.diag(np.diag(matrix)))
    no_offset = not offset.any()

    if is_identity and no_offset:
        return Identity()
    if is_identity:
        return Translation(translation=offset.tolist())
    if is_diagonal and no_offset:
        return Scale(scale=np.diag(matrix).tolist())
    if is_diagonal:
        # Scale first, then translate: y = scale * x + translation, which is
        # the form `multiscales > datasets` accepts.
        return TransformSequence(
            transformations=[
                Scale(scale=np.diag(matrix).tolist()),
                Translation(translation=offset.tolist()),
            ]
        )
    return None
