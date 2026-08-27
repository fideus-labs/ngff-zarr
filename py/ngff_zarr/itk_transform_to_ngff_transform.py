# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Convert ITK transforms to RFC-5 coordinate transformations.

The inverse of :mod:`ngff_zarr.ngff_transform_to_itk_transform`. Its purpose
is persistence: a registration produces an ITK transform, and RFC-5 is where
that result belongs once it is written next to the image.

The same two conventions apply in reverse -- the spatial block and the offset
are permuted from ITK's fastest-axis-first order back to ``dims`` order, and ITK's
center of rotation is folded into the offset, since an RFC-5 affine has no
center:

    ITK:   y = A (x - c) + t + c
    RFC-5: y = A x + b          with  b = t + c - A c
"""

from collections.abc import Sequence
from typing import NamedTuple

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

#: ITK-Wasm parameterizations that describe a deformation rather than an affine
#: mapping. Probing one recovers a matrix from three points that says nothing
#: about the rest of the field, so they are refused by name before any decoding
#: is attempted. RFC-5 represents deformations with ``displacements`` or
#: ``coordinates`` field arrays instead.
_NON_LINEAR_PARAMETERIZATIONS = frozenset(
    {
        "AzimuthElevationToCartesian",
        "BSpline",
        "BSplineSmoothingOnUpdateDisplacementField",
        "ConstantVelocityField",
        "DisplacementField",
        "GaussianExponentialDiffeomorphic",
        "GaussianSmoothingOnUpdateDisplacementField",
        "GaussianSmoothingOnUpdateTimeVaryingVelocityField",
        "Rigid3DPerspective",
        "TimeVaryingVelocityField",
        "VelocityField",
    }
)


def _direct_parameter_count(parameterization: str, dimension: int) -> int | None:
    """Exact parameter count a directly decoded parameterization must carry.

    What ITK's ``MatrixOffsetTransformBase`` and friends pack, and therefore
    the ground truth for the transform's dimensionality. ``None`` for the
    parameterizations this module does not decode arithmetically.
    """
    if parameterization == "Identity":
        return 0
    if parameterization in ("Translation", "Scale"):
        return dimension
    if parameterization == "Affine":
        return dimension * dimension + dimension
    return None


def _reject_non_linear(itk_transform) -> None:
    """Refuse a transform that ITK itself reports as non-linear."""
    if hasattr(itk_transform, "IsLinear") and not itk_transform.IsLinear():
        msg = (
            "only linear ITK transforms can be expressed as an RFC-5 affine; "
            f"{type(itk_transform).__name__} is not linear"
        )
        raise NotImplementedError(msg)


#: Relative agreement required at the check point, by working precision.
#: float64 is 1e-7 rather than machine-level: a composite transform may pass
#: through internal frames (e.g. a global stage position) that dwarf both its
#: probes and its outputs, and the rounding they leave behind is invisible to
#: any output-side bound. 1e-7 absorbs hidden intermediates up to ~1e9 while
#: a deformation still disagrees at the check point by orders of magnitude.
_AFFINE_CHECK_RTOL = {"float32": 1e-5, "float64": 1e-7}


def _working_precision(*samples: np.ndarray) -> str:
    """Guess whether the transform computes in single or double precision.

    ``itk.AffineTransform[itk.F, 3]`` answers ``TransformPoint`` with about
    seven digits, so holding it to a double-precision tolerance rejects every
    valid single-precision transform. Rather than read the ITK type name,
    which is a private spelling, look at whether the values coming back are
    exactly representable in ``float32``: they are when the transform computes
    in ``float32``, and a double-precision result generally is not. Guessing
    ``float32`` for a double-precision transform that happens to return exact
    values only loosens the check on values that carry no round-off anyway.
    """
    values = np.concatenate([np.ravel(np.asarray(s, dtype=float)) for s in samples])
    if values.size and np.all(values.astype(np.float32) == values):
        return "float32"
    return "float64"


def _working_scale(offset: np.ndarray) -> float:
    """The magnitude the transform's own numbers work at.

    Both the probing step and the point the recovered affine is checked at
    follow it, because either one fixed near the origin stops meaning anything
    once the offset dominates: with a step of 1 and an offset of 1e15,
    ``T(h e_j) - T(0)`` keeps no significant digit and probing reports a zero
    matrix that a check near the origin cannot tell from the truth. An affine
    map is exact for *any* step, so there is no truncation error to trade
    against and the usual "small step" rule is exactly backwards here.
    """
    return max(1.0, float(np.abs(offset).max()))


def _reject_non_finite(itk_transform, matrix, offset) -> None:
    """Refuse an affine whose own numbers are not usable."""
    if np.all(np.isfinite(matrix)) and np.all(np.isfinite(offset)):
        return
    msg = (
        f"{type(itk_transform).__name__} does not describe a finite affine "
        "mapping; its matrix or offset holds a non-finite value"
    )
    raise ValueError(msg)


def _check_affine_model(itk_transform, matrix, offset, samples) -> None:
    """Refuse a transform the recovered affine does not actually describe.

    Neither reading a transform's matrix nor probing it proves the transform
    *is* affine. Probing a deformation succeeds and returns a plausible affine
    that is simply wrong away from the probed points, and
    ``AzimuthElevationToCartesianTransform`` inherits an affine's matrix and
    offset, overrides ``TransformPoint``, and still reports itself linear
    (InsightSoftwareConsortium/ITK#6791). So the recovered model is confronted
    with one more evaluation, at a point no probe reached.
    """
    dimension = offset.shape[0]
    # Distinct from every probe and from the origin, whatever the dimension.
    check = _working_scale(offset) * (np.arange(dimension) + 2.0) / (dimension + 2.0)
    predicted = matrix @ check + offset
    actual = np.asarray(itk_transform.TransformPoint(check.tolist()), dtype=float)
    rtol = _AFFINE_CHECK_RTOL[_working_precision(offset, *samples, actual)]
    # Rounding in TransformPoint scales with the magnitudes the evaluation
    # passes through (|A| |x| and |t|), not with the result: a component of
    # T(check) can legitimately be tiny while the terms producing it are huge.
    # Comparing against the output alone would reject such transforms as
    # non-linear, so the comparison floor is the working scale.
    working_scale = max(
        float(np.abs(matrix).max() * np.abs(check).max()),
        float(np.abs(offset).max()),
        1.0,
    )
    magnitude = np.maximum(np.maximum(np.abs(predicted), np.abs(actual)), working_scale)
    if not np.all(np.abs(predicted - actual) <= rtol * magnitude):
        msg = (
            "only linear ITK transforms can be expressed as an RFC-5 affine; "
            f"{type(itk_transform).__name__} does not map "
            f"{np.round(check, 6).tolist()} the way the affine recovered from "
            "it would"
        )
        raise NotImplementedError(msg)


def _matrix_offset_by_probing(itk_transform, dimension: int):
    """Recover ``(matrix, offset)`` by evaluating the transform.

    The fallback for a transform that carries no matrix and offset of its own:
    ``offset = T(0)`` and ``matrix[:, j] = (T(h e_j) - T(0)) / h``. Exact for
    any linear transform whatever parameterization it stores, but it pays for
    that in cancellation, which is why the step follows the transform's own
    scale (see :func:`_working_scale`). The limit it leaves: a matrix
    coefficient whose contribution at the probe scale falls below the rounding
    of the offset itself is unrecoverable by evaluation and folds into the
    offset. Reading the matrix directly has no such limit, so this runs only
    where that is impossible.
    """
    origin = [0.0] * dimension
    offset = np.asarray(itk_transform.TransformPoint(origin), dtype=float)
    step = _working_scale(offset)

    matrix = np.zeros((dimension, dimension))
    probes = []
    for axis in range(dimension):
        basis = [0.0] * dimension
        basis[axis] = step
        column = np.asarray(itk_transform.TransformPoint(basis), dtype=float)
        probes.append(column)
        matrix[:, axis] = (column - offset) / step

    _reject_non_finite(itk_transform, matrix, offset)
    _check_affine_model(itk_transform, matrix, offset, probes)
    return matrix, offset


def _homogeneous(matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """``y = A x + b`` as a square matrix acting on homogeneous coordinates."""
    dimension = offset.shape[0]
    result = np.eye(dimension + 1)
    result[:dimension, :dimension] = matrix
    result[:dimension, dimension] = offset
    return result


def _matrix_offset_direct(itk_transform, dimension: int):
    """``(matrix, offset)`` read from the transform's own numbers, or ``None``.

    Every ``MatrixOffsetTransformBase`` -- an affine, and the rigid, rotation
    and similarity types built on it -- evaluates
    ``y = GetMatrix() x + GetOffset()`` with the center of rotation already
    folded into the offset, whatever it stores its parameters as. Reading that
    pair beats recomposing it from three evaluations, which is a difference of
    nearly equal points: a transform combining a large offset with a small
    matrix coefficient, as nanometre coordinates produce, loses the
    coefficient to cancellation and would be persisted as a singular mapping.

    ``None`` for a transform that carries no such pair, which then falls back
    to probing.
    """
    try:
        import itk
    except ImportError:  # a transform this package did not get from itk
        return None
    try:
        # A CompositeTransform hands its children out as base pointers, which
        # expose neither accessor until they are cast back to their own type.
        itk_transform = itk.down_cast(itk_transform)
    except (AttributeError, TypeError, RuntimeError):
        return None

    if hasattr(itk_transform, "GetNumberOfTransforms"):
        # ITK applies the *last* transform in the queue first, so the
        # homogeneous matrices multiply left to right in queue order.
        total = np.eye(dimension + 1)
        for index in range(itk_transform.GetNumberOfTransforms()):
            decoded = _matrix_offset_direct(
                itk_transform.GetNthTransform(index), dimension
            )
            if decoded is None:
                return None
            total = total @ _homogeneous(*decoded)
        return total[:dimension, :dimension], total[:dimension, dimension]

    if hasattr(itk_transform, "GetMatrix") and hasattr(itk_transform, "GetOffset"):
        return (
            np.asarray(itk.array_from_matrix(itk_transform.GetMatrix()), dtype=float),
            np.asarray(itk_transform.GetOffset(), dtype=float),
        )
    if hasattr(itk_transform, "GetOffset"):
        # itk.TranslationTransform, which has an offset and no matrix.
        return np.eye(dimension), np.asarray(itk_transform.GetOffset(), dtype=float)
    return None


def _matrix_offset_from_itk(itk_transform, dimension: int):
    """Reduce a native ``itk.Transform`` to a single matrix and offset."""
    for name in ("GetInputSpaceDimension", "GetOutputSpaceDimension"):
        # Left to ITK a mismatch surfaces as "Expecting an itkPointD2, an int,
        # a float, ...", which names neither the transform nor `dims`, or as a
        # shape error from comparing points of two different lengths.
        declared = getattr(itk_transform, name, None)
        if declared is not None and int(declared()) != dimension:
            spaces = (
                f"{itk_transform.GetInputSpaceDimension()}D to "
                f"{itk_transform.GetOutputSpaceDimension()}D"
            )
            msg = (
                f"{type(itk_transform).__name__} maps {spaces}, but the "
                f"coordinate system has {dimension} spatial axes"
            )
            raise ValueError(msg)

    decoded = _matrix_offset_direct(itk_transform, dimension)
    if decoded is None:
        return _matrix_offset_by_probing(itk_transform, dimension)
    matrix, offset = decoded
    _reject_non_finite(itk_transform, matrix, offset)
    _check_affine_model(itk_transform, matrix, offset, (matrix,))
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

    # A transform of the wrong dimensionality must not be decoded. Slicing a
    # 3D parameter vector down to 2D silently projects the transform (or, for
    # an affine, scrambles rows into a singular matrix); reading a 2D one as
    # 3D pads it with garbage. The parameter count is the ground truth here,
    # not ``transformType.inputDimension``: ``itkwasm.TransformType`` defaults
    # that field to 3, so a hand-built 2D entry that omitted it would be
    # wrongly refused on the declared number alone.
    expected = _direct_parameter_count(parameterization, dimension)
    if expected is not None:
        if parameters.size != expected:
            msg = (
                f"ITK-Wasm '{parameterization}' transform carries "
                f"{parameters.size} parameters, but a coordinate system with "
                f"{dimension} spatial axes requires exactly {expected}. The "
                "transform's dimensionality does not match `dims`."
            )
            raise ValueError(msg)
        # Only the directly decoded parameterizations read fixedParameters as
        # a center; elsewhere (a displacement field's grid, say) it holds
        # other metadata and is none of this branch's business.
        if fixed.size not in (0, dimension):
            msg = (
                f"ITK-Wasm '{parameterization}' transform carries "
                f"{fixed.size} fixed parameters (the center), but a "
                f"coordinate system with {dimension} spatial axes requires "
                f"0 or {dimension}"
            )
            raise ValueError(msg)
    center = fixed if fixed.size == dimension else np.zeros(dimension)

    if parameterization == "Identity":
        return np.eye(dimension), np.zeros(dimension)
    if parameterization == "Translation":
        return np.eye(dimension), parameters.copy()
    if parameterization == "Scale":
        # itk.ScaleTransform scales about its center, like the affine below.
        matrix = np.diag(parameters)
        return matrix, center - matrix @ center
    if parameterization == "Affine":
        matrix = parameters[: dimension * dimension].reshape(dimension, dimension)
        translation = parameters[dimension * dimension :]
        # ITK applies the matrix about the center, so fold it into the offset.
        return matrix, translation + center - matrix @ center

    if parameterization == "DisplacementField":
        msg = (
            "a displacement field has no affine equivalent; convert it with "
            "itk_displacement_field_to_ngff_transform, which returns the "
            "'displacements' transform and the field to write next to the image"
        )
        raise NotImplementedError(msg)
    if parameterization in _NON_LINEAR_PARAMETERIZATIONS:
        msg = (
            "only linear ITK transforms can be expressed as an RFC-5 affine; "
            f"'{parameterization}' describes a deformation. RFC-5 represents "
            "those with a 'displacements' or 'coordinates' field instead."
        )
        raise NotImplementedError(msg)

    # An angle or quaternion parameterization packs a type-specific number of
    # parameters, so the count says nothing about dimensionality here. These
    # entries come from serializers (itk.dict_from_transform, the ITK-Wasm
    # pipeline), which do fill in the declared dimension, so it is trustworthy
    # on this path in a way it is not for the hand-built matrix entries above.
    try:
        declared = int(transform_type.inputDimension)
    except (TypeError, ValueError):
        declared = dimension  # unusable declaration: let the itk rebuild judge
    if declared != dimension:
        msg = (
            f"ITK-Wasm '{parameterization}' transform is {declared}D, but the "
            f"coordinate system has {dimension} spatial axes"
        )
        raise ValueError(msg)

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
    # The name check above only covers the parameterizations known at the time
    # of writing; ITK's own answer covers the rest.
    _reject_non_linear(rebuilt)
    return _matrix_offset_from_itk(rebuilt, dimension)


def _itk_matrix_offset(transform, dimension: int):
    """Reduce any supported ITK transform input to a single matrix and offset."""
    from itkwasm import Transform as ItkWasmTransform

    # A native itk.Transform, including the CompositeTransform Elastix returns.
    if hasattr(transform, "TransformPoint"):
        if hasattr(transform, "GetDisplacementField"):
            msg = (
                "a displacement field has no affine equivalent; convert it with "
                "itk_displacement_field_to_ngff_transform, which returns the "
                "'displacements' transform and the field to write next to the "
                "image"
            )
            raise NotImplementedError(msg)
        _reject_non_linear(transform)
        return _matrix_offset_from_itk(transform, dimension)

    if isinstance(transform, ItkWasmTransform):
        entries = [transform]
    elif isinstance(transform, Sequence) and not isinstance(transform, (str, bytes)):
        entries = list(transform)
    else:
        msg = (
            f"unsupported transform input {type(transform).__name__}. Expected an "
            "itk.Transform, an itkwasm.Transform, or an ITK-Wasm TransformList."
        )
        raise TypeError(msg)
    if not entries:
        msg = "transform list is empty"
        raise ValueError(msg)

    # An ITK transform list applies its last entry first, so the homogeneous
    # matrices multiply left to right in list order.
    total = np.eye(dimension + 1)
    for entry in entries:
        if not isinstance(entry, ItkWasmTransform):
            # itk.transformread returns a list of native transforms, which
            # looks like a TransformList and decodes like nothing at all.
            msg = (
                f"transform list entry is a {type(entry).__name__}, not an "
                "itkwasm.Transform. A list of native itk.Transform objects is "
                "not an ITK-Wasm transform list: compose them into an "
                "itk.CompositeTransform, or pass a single itk.Transform."
            )
            raise TypeError(msg)
        if _parameterization_name(entry.transformType) == "Composite":
            # A parameterless 'Composite' entry is ambiguous. The ITK-Wasm
            # pipeline writes one as a grouping header before the children,
            # but itk.dict_from_transform never writes a header at all: there
            # it is a *nested* composite whose children the serialization
            # dropped (InsightSoftwareConsortium/ITK#6792), at any position
            # including the first. Decoding past one would silently compose
            # the wrong mapping, so refuse the entry wherever it appears and
            # name the ways out.
            msg = (
                "a 'Composite' entry in an ITK-Wasm transform list cannot be "
                "decoded: itk.dict_from_transform drops a nested composite's "
                "children, leaving this entry indistinguishable from a "
                "pipeline grouping header. Pass the native itk.Transform "
                "instead, or, for a pipeline-serialized list, drop the "
                "leading header entry and pass the children."
            )
            raise NotImplementedError(msg)
        total = total @ _homogeneous(*_matrix_offset_from_itkwasm(entry, dimension))
    return total[:dimension, :dimension], total[:dimension, dimension]


def _itk_axis_order(spatial):
    """The spatial dims in ITK component order: x first, then y, then z.

    ITK orders points fastest-axis-first by *name*, not by reversing however
    the caller spelled ``dims``: reversing is only right for the canonical
    (z, y, x).
    """
    return [dim for dim in _SPATIAL_DIMS if dim in spatial]


def _permutation_from_itk(spatial) -> np.ndarray:
    """The matrix sending an ITK-order vector to the ``dims``-order vector."""
    itk_dims = _itk_axis_order(spatial)
    permutation = np.zeros((len(spatial), len(spatial)))
    for row, dim in enumerate(spatial):
        permutation[row, itk_dims.index(dim)] = 1.0
    return permutation


def _inverse_direction(direction: np.ndarray) -> np.ndarray:
    """The inverse of an RFC-4 direction matrix.

    Every column names one LPS axis with a sign, so the matrix is a signed
    permutation: orthogonal, and its transpose is its exact inverse. Going
    through ``np.linalg.inv`` would round where the transpose does not, and
    would answer a direction that somehow came through singular with a
    ``LinAlgError`` rather than with geometry.
    """
    return direction.T


class _FrameGeometry(NamedTuple):
    """The direction matrices and origins of an image pair, in ITK order.

    A tuple, so it still unpacks straight into :func:`_change_of_frame`, whose
    arguments it names in order.
    """

    direction_in: np.ndarray
    direction_out: np.ndarray
    origin_in: np.ndarray
    origin_out: np.ndarray


def _frame_geometry(fixed, moving, itk_dims) -> _FrameGeometry:
    """The geometry ``ngff_image_to_itk_image`` gives the two images."""
    from .resample_bounding_box import _itk_direction

    return _FrameGeometry(
        _itk_direction(fixed, itk_dims),
        _itk_direction(moving, itk_dims),
        np.array([float(fixed.translation[d]) for d in itk_dims]),
        np.array([float(moving.translation[d]) for d in itk_dims]),
    )


def _change_of_frame(
    matrix, offset, direction_in, direction_out, origin_in, origin_out
):
    """Re-express ``y = A x + t`` through a change of frame on each side.

    ``ngff_image_to_itk_image`` builds each image with ``origin = translation``
    and the RFC-4 direction matrix ``D``, so an intrinsic point ``p`` sits at
    the physical point ``phi(p) = D (p - o) + o``. Given a mapping ``T``
    between two frames, this returns ``phi_out^-1 . T . phi_in`` for the
    directions and origins supplied:

        M = D_out^-1 A D_in
        b = D_out^-1 (A (I - D_in) o_in + t - o_out) + o_out

    ``phi^-1`` has the same shape as ``phi`` with ``D^-1`` in place of ``D``,
    so the same formula serves both conversions: ITK to RFC-5 passes the
    images' directions, RFC-5 to ITK passes their inverses. The scales cancel
    on both sides; the translations do not, which is why orientations alone
    would not be enough. With no anatomical orientation every direction is
    the identity and the mapping comes back unchanged.
    """
    inverse_out = _inverse_direction(direction_out)
    identity = np.eye(len(origin_in))
    conjugated = inverse_out @ matrix @ direction_in
    shifted = (
        inverse_out
        @ (matrix @ (identity - direction_in) @ origin_in + offset - origin_out)
        + origin_out
    )
    return conjugated, shifted


def _check_frame_images(fixed, moving, spatial):
    """Validate the fixed/moving pair handed to a converter."""
    if (fixed is None) != (moving is None):
        msg = "pass both fixed and moving, or neither"
        raise ValueError(msg)
    if fixed is None:
        return False
    for label, image in (("fixed", fixed), ("moving", moving)):
        missing = [d for d in spatial if d not in image.translation]
        if missing:
            msg = (
                f"the {label} image has no translation entry for spatial "
                f"dimension(s) {missing}; its dims do not cover `dims`"
            )
            raise ValueError(msg)
    return True


def itk_transform_to_ngff_matrix(
    transform,
    dims: Sequence[str],
    *,
    fixed=None,
    moving=None,
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

    if _check_frame_images(fixed, moving, spatial):
        # ITK physical space -> the intrinsic systems: phi_m^-1 . T . phi_f.
        geometry = _frame_geometry(fixed, moving, _itk_axis_order(spatial))
        matrix, offset = _change_of_frame(matrix, offset, *geometry)

    # ITK (fastest-axis-first) order -> the order the axes appear in `dims`.
    permutation = _permutation_from_itk(spatial)
    return permutation @ matrix @ permutation.T, permutation @ offset


def itk_transform_to_ngff_transform(
    transform,
    dims: Sequence[str],
    simplify: bool = True,
    *,
    fixed=None,
    moving=None,
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
        RFC-5 recommends this. A mirror never simplifies to a ``scale``, since
        RFC-5 requires strictly positive scale factors. Note that
        ``multiscales > datasets`` accepts only a single ``scale``, a single
        ``identity``, or a two-element ``sequence`` of scale and translation,
        so a bare ``translation`` or an ``affine`` belongs in the
        multiscales-level ``coordinateTransformations`` instead. Set to
        ``False`` to always get an ``affine``.
    :type  simplify: bool

    :return: An RFC-5 coordinate transformation over ``dims``.
    :rtype: Transform

    :param fixed: The fixed and moving images the ITK transform was produced
        on. An ITK transform acts on ITK physical space, which includes the
        direction matrix derived from RFC-4 anatomical orientation; an RFC-5
        transformation acts on the intrinsic coordinate systems. Passing both
        images lets the conversion change frames exactly. Omitting them is
        exact only when neither image carries an anatomical orientation.
    :type  fixed: NgffImage, optional

    :param moving: See ``fixed``. Pass both or neither.
    :type  moving: NgffImage, optional

    :raises NotImplementedError: If the transform is not linear. A non-linear
        registration has no affine equivalent.
    """
    dims = tuple(dims)
    matrix, offset = itk_transform_to_ngff_matrix(
        transform, dims, fixed=fixed, moving=moving
    )

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
    diagonal = np.diag(matrix)
    is_identity = np.array_equal(matrix, np.eye(ndim))
    is_diagonal = np.array_equal(matrix, np.diag(diagonal))
    no_offset = not offset.any()
    # RFC-5 requires every scale factor to be strictly positive, so a mirror
    # or a flip is not a `scale` however diagonal its matrix looks. Emitting
    # one anyway writes metadata the schema rejects, and an LPS to RAS flip
    # (diag(-1, -1, 1)) is an ordinary registration result. Fall through to
    # `affine`, which carries the sign.
    is_scale = is_diagonal and bool((diagonal > 0).all())

    if is_identity and no_offset:
        return Identity()
    if is_identity:
        return Translation(translation=offset.tolist())
    if is_scale and no_offset:
        return Scale(scale=diagonal.tolist())
    if is_scale:
        # Scale first, then translate: y = scale * x + translation, which is
        # the form `multiscales > datasets` accepts.
        return TransformSequence(
            transformations=[
                Scale(scale=diagonal.tolist()),
                Translation(translation=offset.tolist()),
            ]
        )
    return None
