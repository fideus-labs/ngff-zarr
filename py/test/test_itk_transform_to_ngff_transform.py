# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for converting ITK transforms back to RFC-5 transformations.

The strongest check available here is the round trip: converting an RFC-5
transformation to ITK and back must preserve the mapping exactly. Anything
wrong with the axis reversal or the center-of-rotation algebra breaks it.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import (
    itk_transform_to_ngff_matrix,
    itk_transform_to_ngff_transform,
    ngff_transform_to_itk_transform,
)
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Identity,
    Rotation,
    Scale,
    TransformSequence,
    Translation,
)
from packaging import version

ROUND_TRIP_CASES = [
    ("identity", Identity(), ("y", "x")),
    ("translation", Translation(translation=[5.0, 10.0]), ("y", "x")),
    ("scale", Scale(scale=[2.0, 3.0, 4.0]), ("z", "y", "x")),
    ("affine_2d", Affine(affine=[[0.8, -0.6, 10.0], [0.6, 0.8, -4.0]]), ("y", "x")),
    (
        "sheared_affine_3d",
        Affine(
            affine=[
                [1.0, 0.2, 0.0, 4.0],
                [0.0, 2.0, 0.3, -6.0],
                [0.5, 0.0, 1.0, 11.0],
            ]
        ),
        ("z", "y", "x"),
    ),
    (
        "sequence",
        TransformSequence(
            transformations=[
                Translation(translation=[0.0, 10.0]),
                Scale(scale=[1.0, 2.0]),
            ]
        ),
        ("y", "x"),
    ),
    (
        "affine_with_non_spatial_axes",
        Affine(
            affine=[
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.1, 3.0],
                [0.0, 0.0, 0.2, 1.1, -2.0],
            ]
        ),
        ("t", "c", "y", "x"),
    ),
]


def _apply(transform, point):
    """Evaluate an RFC-5 transformation, independently of the module.

    The round trip below compares mappings point by point rather than reading
    both sides through ``_ngff_transform_to_itk_matrix``, so it does not lean on
    the helper whose conventions it is meant to exercise.

    Note what a round trip can and cannot show. It establishes that the two
    directions are mutual inverses; by construction it cannot see a convention
    the two of them get wrong *together*, since that error cancels. Reverse the
    axes in neither direction and every case here still passes. The absolute
    anchors live elsewhere: the documented worked example, ``_oracle_region``,
    and the comparisons against ``itk``'s own ``TransformPoint``.
    """
    point = np.asarray(point, dtype=float)
    if isinstance(transform, Identity):
        return point
    if isinstance(transform, Scale):
        return point * np.asarray(transform.scale, dtype=float)
    if isinstance(transform, Translation):
        return point + np.asarray(transform.translation, dtype=float)
    if isinstance(transform, Affine):
        affine = np.asarray(transform.affine, dtype=float)
        return affine[:, :-1] @ point + affine[:, -1]
    if isinstance(transform, TransformSequence):
        # RFC-5 applies the first entry first.
        for sub_transform in transform.transformations:
            point = _apply(sub_transform, point)
        return point
    raise AssertionError(f"the oracle does not handle {type(transform).__name__}")


def _sample_points(ndim):
    """Points that pin every column of the matrix and the offset."""
    rng = np.random.default_rng(20260806)
    return [np.zeros(ndim), *rng.normal(scale=10.0, size=(4, ndim))]


@pytest.mark.parametrize(
    "transform, dims",
    [(case[1], case[2]) for case in ROUND_TRIP_CASES],
    ids=[case[0] for case in ROUND_TRIP_CASES],
)
def test_round_trip_preserves_the_mapping(transform, dims):
    """RFC-5 -> ITK -> RFC-5 must describe the same function."""
    converted = itk_transform_to_ngff_transform(
        ngff_transform_to_itk_transform(transform, dims), dims
    )

    for point in _sample_points(len(dims)):
        assert np.allclose(_apply(converted, point), _apply(transform, point))


@pytest.mark.parametrize(
    "transform, expected_type",
    [
        (Identity(), Identity),
        (Translation(translation=[5.0, 10.0]), Translation),
        (Scale(scale=[2.0, 3.0]), Scale),
        (
            TransformSequence(
                transformations=[
                    Scale(scale=[2.0, 3.0]),
                    Translation(translation=[1.0, 2.0]),
                ]
            ),
            TransformSequence,
        ),
        (Affine(affine=[[0.8, -0.6, 0.0], [0.6, 0.8, 0.0]]), Affine),
    ],
)
def test_simplify_returns_the_least_expressive_form(transform, expected_type):
    """RFC-5 recommends the simplest form, and datasets only accept those."""
    dims = ("y", "x")
    converted = itk_transform_to_ngff_transform(
        ngff_transform_to_itk_transform(transform, dims), dims
    )
    assert isinstance(converted, expected_type)


def test_a_mirror_does_not_simplify_to_a_scale():
    """RFC-5 requires strictly positive scale factors.

    A flip is diagonal, so a sign-blind simplifier calls it a ``scale`` and
    writes metadata the 0.6 schema rejects. An LPS to RAS flip is an ordinary
    registration result, so this is not an exotic input.
    """
    itk = pytest.importorskip("itk")

    flip = itk.ScaleTransform[itk.D, 2].New()
    flip.SetScale([-1.0, 1.0])

    converted = itk_transform_to_ngff_transform(flip, ("y", "x"))

    assert isinstance(converted, Affine)
    assert np.allclose(converted.affine, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

    # A mirror with a translation must not become a scale/translation sequence
    # either, for the same reason.
    mirror = itk.AffineTransform[itk.D, 2].New()
    mirror.SetMatrix(itk.matrix_from_array(np.diag([-2.0, 3.0])))
    mirror.SetTranslation([5.0, 7.0])
    mirror.SetCenter([0.0, 0.0])
    assert isinstance(itk_transform_to_ngff_transform(mirror, ("y", "x")), Affine)

    # ... while a strictly positive diagonal still simplifies.
    positive = itk.ScaleTransform[itk.D, 2].New()
    positive.SetScale([2.0, 3.0])
    assert isinstance(itk_transform_to_ngff_transform(positive, ("y", "x")), Scale)


@pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="writing OME-Zarr 0.6 requires zarr-python 3",
)
def test_a_mirror_writes_a_store_that_validates(tmp_path):
    """The point of the sign rule: the metadata has to survive the validator."""
    itk = pytest.importorskip("itk")

    import dask.array as da
    from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
    from ngff_zarr.v06.zarr_metadata import (
        CoordinateSystem,
        CoordinateSystemIdentifier,
    )

    flip = itk.ScaleTransform[itk.D, 2].New()
    flip.SetScale([-1.0, 1.0])
    converted = itk_transform_to_ngff_transform(flip, ("y", "x"))

    image = to_ngff_image(
        da.zeros((32, 32), dtype=np.uint8),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    multiscales = to_multiscales(image, scale_factors=[])
    intrinsic = multiscales.metadata.coordinateSystems[0]
    registered = CoordinateSystem(name="registered", axes=list(intrinsic.axes))
    multiscales.metadata.coordinateSystems.append(registered)
    converted.input = CoordinateSystemIdentifier(name=intrinsic.name)
    converted.output = CoordinateSystemIdentifier(name=registered.name)
    multiscales.metadata.coordinateTransformations = [converted]

    store = tmp_path / "mirror.ome.zarr"
    to_ome_zarr(str(store), multiscales, version="0.6")
    from_ome_zarr(str(store), validate=True)


def test_simplify_can_be_disabled():
    dims = ("y", "x")
    converted = itk_transform_to_ngff_transform(
        ngff_transform_to_itk_transform(Identity(), dims), dims, simplify=False
    )
    assert isinstance(converted, Affine)
    assert converted.affine == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_itk_transform_with_angle_parameters_is_converted():
    """Euler stores an angle, not a matrix, so reading its parameters would
    need type-specific arithmetic. Every ``MatrixOffsetTransformBase`` answers
    ``GetMatrix``/``GetOffset`` alike, whatever it stores underneath."""
    itk = pytest.importorskip("itk")

    euler = itk.Euler2DTransform[itk.D].New()
    euler.SetAngle(0.3)
    euler.SetTranslation([5.0, -2.0])

    matrix, offset = itk_transform_to_ngff_matrix(euler, ("y", "x"))

    cosine, sine = np.cos(0.3), np.sin(0.3)
    reversal = np.eye(2)[::-1]
    expected = reversal @ np.array([[cosine, -sine], [sine, cosine]]) @ reversal
    assert np.allclose(matrix, expected)
    # ITK order (x, y) reverses to NGFF order (y, x).
    assert np.allclose(offset, [-2.0, 5.0])


def test_itk_composite_transform_agrees_with_transform_point():
    """The Elastix path: a composite must convert to the same mapping."""
    itk = pytest.importorskip("itk")

    translation = itk.TranslationTransform[itk.D, 2].New()
    translation.SetOffset([10.0, 0.0])
    scaling = itk.AffineTransform[itk.D, 2].New()
    scaling.SetMatrix(itk.matrix_from_array(np.diag([2.0, 2.0])))
    scaling.SetTranslation([0.0, 0.0])
    scaling.SetCenter([0.0, 0.0])
    composite = itk.CompositeTransform[itk.D, 2].New()
    composite.AddTransform(translation)
    composite.AddTransform(scaling)

    matrix, offset = itk_transform_to_ngff_matrix(composite, ("y", "x"))

    rng = np.random.default_rng(12345)
    for _ in range(5):
        point = rng.normal(size=2)  # (y, x)
        expected = np.asarray(composite.TransformPoint(point[::-1].tolist()))[::-1]
        assert np.allclose(matrix @ point + offset, expected)


def test_itk_center_of_rotation_is_folded_into_the_offset():
    """An RFC-5 affine has no center, so ITK's must be absorbed."""
    itk = pytest.importorskip("itk")

    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    center = np.array([9.5, 4.5])
    translation = np.array([10.0, 10.0])
    affine = itk.AffineTransform[itk.D, 2].New()
    affine.SetMatrix(itk.matrix_from_array(rotation))
    affine.SetTranslation(translation.tolist())
    affine.SetCenter(center.tolist())

    _, offset = itk_transform_to_ngff_matrix(affine, ("y", "x"))

    expected = translation + center - rotation @ center
    assert np.allclose(offset, expected[::-1])


def test_non_linear_itk_transform_is_rejected():
    itk = pytest.importorskip("itk")

    bspline = itk.BSplineTransform[itk.D, 2, 3].New()
    with pytest.raises(NotImplementedError, match="not linear"):
        itk_transform_to_ngff_transform(bspline, ("y", "x"))


def _itkwasm_entry(parameterization, parameters, fixed, dimension=2):
    """A bare ITK-Wasm transform entry of any parameterization."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    return Transform(
        transformType=TransformType(
            transformParameterization=getattr(
                TransformParameterizations, parameterization
            ),
            parametersValueType=FloatTypes.Float64,
            inputDimension=dimension,
            outputDimension=dimension,
        ),
        numberOfFixedParameters=len(fixed),
        numberOfParameters=len(parameters),
        fixedParameters=np.asarray(fixed, dtype=np.float64),
        parameters=np.asarray(parameters, dtype=np.float64),
    )


def _itkwasm_affine(matrix_row_major, translation, center, parameterization=None):
    """An ITK-Wasm affine, by default built with the enum member."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    dimension = len(translation)
    transform_type = TransformType(
        transformParameterization=(
            TransformParameterizations.Affine
            if parameterization is None
            else parameterization
        ),
        parametersValueType=FloatTypes.Float64,
        inputDimension=dimension,
        outputDimension=dimension,
    )
    parameters = np.asarray(
        list(matrix_row_major) + list(translation), dtype=np.float64
    )
    return Transform(
        transformType=transform_type,
        numberOfFixedParameters=len(center),
        numberOfParameters=len(parameters),
        fixedParameters=np.asarray(center, dtype=np.float64),
        parameters=parameters,
    )


def _displacement_field_transform(itk, size=8, spacing=8.0, seed=0):
    """A random deformation: nothing about it is affine."""
    field = itk.Image[itk.Vector[itk.D, 2], 2].New()
    region = itk.ImageRegion[2]()
    extent = itk.Size[2]()
    extent[0], extent[1] = size, size
    region.SetSize(extent)
    field.SetRegions(region)
    field.SetSpacing([spacing, spacing])
    field.Allocate()
    rng = np.random.default_rng(seed)
    itk.array_view_from_image(field)[:] = rng.normal(scale=5.0, size=(size, size, 2))

    transform = itk.DisplacementFieldTransform[itk.D, 2].New()
    transform.SetDisplacementField(field)
    return transform


def test_non_linear_transform_is_rejected_as_an_itkwasm_entry():
    """The ITK-Wasm path must refuse what the native path refuses.

    Probing a deformation succeeds: three points always determine an affine.
    The result is a plausible transform that is simply wrong everywhere else,
    and it would be written into the store without a word.
    """
    itk = pytest.importorskip("itk")
    from ngff_zarr.resample_bounding_box import _as_itk_transform_list

    displacement = _displacement_field_transform(itk)
    # A displacement field is refused with a pointer to its own conversion,
    # on both paths alike.
    with pytest.raises(
        NotImplementedError, match="itk_displacement_field_to_ngff_transform"
    ):
        itk_transform_to_ngff_transform(displacement, ("y", "x"))

    entries = _as_itk_transform_list(displacement)
    with pytest.raises(
        NotImplementedError, match="itk_displacement_field_to_ngff_transform"
    ):
        itk_transform_to_ngff_transform(entries, ("y", "x"))

    # Any other deformation gets the generic refusal on both paths.
    bspline = _itkwasm_entry("BSpline", np.zeros(32), np.zeros(8), dimension=2)
    with pytest.raises(NotImplementedError, match="only linear"):
        itk_transform_to_ngff_transform([bspline], ("y", "x"))


def test_non_linear_transform_is_rejected_without_itk(monkeypatch):
    """Refusing a deformation must not depend on the optional ``itk`` extra."""
    itk = pytest.importorskip("itk")
    from ngff_zarr.resample_bounding_box import _as_itk_transform_list

    entries = _as_itk_transform_list(_displacement_field_transform(itk))

    import builtins

    real_import = builtins.__import__

    def no_itk(name, *args, **kwargs):
        if name == "itk":
            raise ImportError("itk is unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_itk)

    with pytest.raises(
        NotImplementedError, match="itk_displacement_field_to_ngff_transform"
    ):
        itk_transform_to_ngff_matrix(entries, ("y", "x"))
    bspline = _itkwasm_entry("BSpline", np.zeros(32), np.zeros(8), dimension=2)
    with pytest.raises(NotImplementedError, match="describes a deformation"):
        itk_transform_to_ngff_matrix([bspline], ("y", "x"))


def test_probing_refuses_a_transform_that_is_not_affine():
    """The last line of defence, for a type ``IsLinear()`` does not catch."""
    from ngff_zarr.itk_transform_to_ngff_transform import _matrix_offset_by_probing

    class Quadratic:
        """Linear at the origin and along each axis, quadratic in between."""

        def TransformPoint(self, point):
            x, y = point
            return [x + 3.0 * x * y, y]

    with pytest.raises(NotImplementedError, match="only linear"):
        _matrix_offset_by_probing(Quadratic(), 2)


def _float32_affine(matrix, translation):
    """An ``itk.AffineTransform[itk.F, N]``, built through its parameters.

    ``SetMatrix`` wants an ``itk.Matrix`` of the same value type, so the
    parameter vector is the least fiddly way in.
    """
    itk = pytest.importorskip("itk")

    matrix = np.asarray(matrix, dtype=float)
    dimension = matrix.shape[0]
    transform = itk.AffineTransform[itk.F, dimension].New()
    parameters = itk.OptimizerParameters[itk.F](dimension * dimension + dimension)
    for index, value in enumerate([*matrix.ravel(), *translation]):
        parameters.SetElement(index, float(value))
    transform.SetParameters(parameters)
    fixed = itk.OptimizerParameters[itk.D](dimension)
    for index in range(dimension):
        fixed.SetElement(index, 0.0)
    transform.SetFixedParameters(fixed)
    return transform


# Stored as the nearest doubles; every recovery below is compared against
# those doubles rather than against the decimals.
_ROTATION_3D = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.6], [0.0, -0.6, 0.8]]


def _affine_3d(matrix, offset):
    itk = pytest.importorskip("itk")

    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetMatrix(itk.matrix_from_array(np.asarray(matrix, dtype=float)))
    transform.SetTranslation([float(offset)] * 3)
    transform.SetCenter([0.0] * 3)
    return transform


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e8, 1e12, 1e15])
@pytest.mark.parametrize("coefficient", [1.0, 1e-20])
def test_an_offset_that_dwarfs_the_matrix_does_not_swallow_it(offset, coefficient):
    """The matrix must survive an offset of any magnitude beside it.

    Nanometre coordinates put real electron-microscopy data in the 1e15 range,
    where a matrix read back from ``T(e_j) - T(0)`` would be pure cancellation
    noise: at a coefficient of 1e-20 the difference rounds to zero outright,
    and a transform that inverts is persisted as one that does not. Reading
    ``GetMatrix``/``GetOffset`` has no such limit, so equality is exact.
    """
    rotation = np.asarray(_ROTATION_3D) * coefficient
    matrix, _ = itk_transform_to_ngff_matrix(
        _affine_3d(rotation, offset), ("z", "y", "x")
    )

    reversal = np.eye(3)[::-1]
    assert np.array_equal(matrix, reversal @ rotation @ reversal)


def test_probing_follows_the_offset_scale():
    """The fallback path, for a transform that carries no matrix of its own.

    Probing is a difference of two nearly equal points, so a step fixed at 1
    keeps no significant digit once the offset reaches 1e15 and the recovered
    matrix collapses towards zero. Following the offset's own magnitude keeps
    the two terms comparable; the check point has to grow with it too, since
    one fixed near the origin cannot tell a zeroed matrix from a correct one.
    """
    from ngff_zarr.itk_transform_to_ngff_transform import _matrix_offset_by_probing

    expected = np.asarray(_ROTATION_3D)
    for offset in (0.0, 1e3, 1e8, 1e12, 1e15):
        matrix, recovered = _matrix_offset_by_probing(_affine_3d(expected, offset), 3)
        assert np.allclose(matrix, expected, rtol=0, atol=1e-12)
        assert np.allclose(recovered, [offset] * 3)


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e8])
def test_single_precision_transforms_are_not_rejected_as_non_linear(offset):
    """A float32 transform carries about seven digits, not sixteen.

    Holding it to a double-precision tolerance refuses every single-precision
    rotation, at any magnitude including none, even though ``IsLinear()`` says
    it is linear and it is.
    """
    transform = _float32_affine(_ROTATION_3D, [offset] * 3)
    assert transform.IsLinear()

    matrix, _ = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))

    reversal = np.eye(3)[::-1]
    expected = reversal @ np.asarray(_ROTATION_3D) @ reversal
    assert np.allclose(matrix, expected, rtol=0, atol=1e-6)


def test_a_composite_with_hidden_large_intermediates_converts():
    """A composite may pass through internal frames that dwarf its outputs.

    Two translations out to a global frame at 1e9 and back leave T(x) - x
    exactly constant, but the rounding of those hidden intermediates lands in
    every evaluation. A tolerance measured against the *output* rejects such
    transforms as non-linear; measured against the working scale it must not.
    Composing the children keeps the intermediates out of the result entirely,
    so both routes are checked here.
    """
    itk = pytest.importorskip("itk")
    from ngff_zarr.itk_transform_to_ngff_transform import _matrix_offset_by_probing

    big = 1e9
    composite = itk.CompositeTransform[itk.D, 3].New()
    outward = itk.TranslationTransform[itk.D, 3].New()
    outward.SetOffset([big, -big, big])
    backward = itk.TranslationTransform[itk.D, 3].New()
    backward.SetOffset([-big + 5.25, big - 3.125, -big + 1.5])
    composite.AddTransform(outward)
    composite.AddTransform(backward)

    matrix, offset = itk_transform_to_ngff_matrix(composite, ("z", "y", "x"))
    assert np.array_equal(matrix, np.eye(3))
    assert np.array_equal(offset, [1.5, -3.125, 5.25])

    probed, probed_offset = _matrix_offset_by_probing(composite, 3)
    assert np.allclose(probed, np.eye(3), rtol=0, atol=1e-12)
    assert np.allclose(probed_offset, [5.25, -3.125, 1.5])


def test_the_affine_check_still_refuses_a_near_affine_transform():
    """Loosening the tolerance for float32 must not blunt the check.

    This is linear at the origin and along every axis and quadratic in
    between, so only the check point can tell, and only if it is far enough
    from the probes.
    """
    from ngff_zarr.itk_transform_to_ngff_transform import _matrix_offset_by_probing

    class NearlyAffine:
        def __init__(self, coefficient, offset):
            self.coefficient = coefficient
            self.offset = offset

        def TransformPoint(self, point):
            x, y, z = point
            return [x + self.coefficient * x * y + self.offset, y + self.offset, z]

    for coefficient in (1e-1, 1e-3, 1e-6):
        for offset in (0.0, 1e8):
            with pytest.raises(NotImplementedError, match="only linear"):
                _matrix_offset_by_probing(NearlyAffine(coefficient, offset), 3)


@pytest.mark.parametrize("magnitude", [1e-6, 1.0, 1e6])
def test_a_genuinely_linear_transform_is_accepted(magnitude):
    """The affine check must not fire on ordinary float error.

    It compares two arithmetic paths -- the recovered model against the
    transform's own evaluation -- rather than a value against itself, so it
    has to stay quiet across the range of magnitudes a registration produces,
    whichever route recovered the model.
    """
    itk = pytest.importorskip("itk")
    from ngff_zarr.itk_transform_to_ngff_transform import _matrix_offset_by_probing

    euler = itk.Euler3DTransform[itk.D].New()
    euler.SetRotation(0.3, -0.7, 1.1)
    euler.SetTranslation([magnitude, -2 * magnitude, 3 * magnitude])
    euler.SetCenter([0.5 * magnitude, magnitude, -1.5 * magnitude])

    matrix, offset = itk_transform_to_ngff_matrix(euler, ("z", "y", "x"))
    _matrix_offset_by_probing(euler, 3)

    point = np.array([12.0, -34.0, 56.0]) * magnitude  # (z, y, x)
    expected = np.asarray(euler.TransformPoint(point[::-1].tolist()))[::-1]
    assert np.allclose(matrix @ point + offset, expected)


def test_a_transform_whose_matrix_contradicts_its_mapping_is_refused():
    """``IsLinear()`` and the stored matrix can both be wrong at once.

    ``AzimuthElevationToCartesianTransform`` inherits ``AffineTransform``, so
    it carries an identity matrix and offset and reports itself linear, while
    overriding ``TransformPoint`` with a spherical-to-Cartesian mapping
    (InsightSoftwareConsortium/ITK#6791). Nothing but confronting the model
    with an evaluation catches it.
    """
    itk = pytest.importorskip("itk")
    template = getattr(itk, "AzimuthElevationToCartesianTransform", None)
    if template is None:  # not every ITK build wraps it
        pytest.skip("this ITK build does not wrap AzimuthElevationToCartesian")

    transform = template[itk.D, 3].New()
    assert transform.IsLinear()  # the upstream defect this guards against
    assert np.array_equal(itk.array_from_matrix(transform.GetMatrix()), np.eye(3))

    with pytest.raises(NotImplementedError, match="only linear"):
        itk_transform_to_ngff_transform(transform, ("z", "y", "x"))


def test_a_transform_of_the_wrong_dimensionality_is_named():
    """Left to ITK this is a SWIG TypeError naming an ``itkPointD2``."""
    itk = pytest.importorskip("itk")

    with pytest.raises(ValueError, match="maps 2D to 2D, but the coordinate system"):
        itk_transform_to_ngff_matrix(
            itk.AffineTransform[itk.D, 2].New(), ("z", "y", "x")
        )


@pytest.mark.parametrize("transform", [5, "affine", None, {"affine": 1}])
def test_an_input_that_is_no_transform_at_all_is_named(transform):
    """Not 'int object is not iterable' from inside a list() call."""
    with pytest.raises(TypeError, match="unsupported transform input"):
        itk_transform_to_ngff_matrix(transform, ("y", "x"))


def _itkwasm_scale(scale, center):
    """An ITK-Wasm scale, with the center of scaling as its fixed parameters."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    dimension = len(scale)
    return Transform(
        transformType=TransformType(
            transformParameterization=TransformParameterizations.Scale,
            parametersValueType=FloatTypes.Float64,
            inputDimension=dimension,
            outputDimension=dimension,
        ),
        numberOfFixedParameters=len(center),
        numberOfParameters=dimension,
        fixedParameters=np.asarray(center, dtype=np.float64),
        parameters=np.asarray(scale, dtype=np.float64),
    )


def test_itkwasm_scale_center_is_folded_into_the_offset():
    """``itk.ScaleTransform`` scales about its center, like an affine does.

    Reading its parameters and ignoring the fixed ones yields a plausible
    transform that puts the image in the wrong place, with no error.
    """
    transform = _itkwasm_scale([2.0, 3.0], [10.0, 20.0])

    matrix, offset = itk_transform_to_ngff_matrix(transform, ("y", "x"))

    # ITK order (x, y): c - S c = (10 - 2*10, 20 - 3*20) = (-10, -40).
    assert np.array_equal(matrix, [[3.0, 0.0], [0.0, 2.0]])
    assert np.array_equal(offset, [-40.0, -10.0])


def test_itkwasm_scale_agrees_with_the_native_itk_transform():
    """The two input paths must not disagree about the same transform.

    A native ``itk.Transform`` is probed; the same transform handed over as an
    ITK-Wasm entry is decoded from its parameters.
    """
    itk = pytest.importorskip("itk")
    from ngff_zarr.resample_bounding_box import _as_itk_transform_list

    scaling = itk.ScaleTransform[itk.D, 2].New()
    scaling.SetScale([2.0, 3.0])
    scaling.SetCenter([10.0, 20.0])

    probed = itk_transform_to_ngff_matrix(scaling, ("y", "x"))
    decoded = itk_transform_to_ngff_matrix(_as_itk_transform_list(scaling), ("y", "x"))

    assert np.allclose(probed[0], decoded[0])
    assert np.allclose(probed[1], decoded[1])


def test_parameterization_is_read_by_value_not_by_str():
    """``str()`` on an itkwasm parameterization is not its name.

    ``TransformParameterizations`` mixes in ``str``, so since Python 3.11
    ``str(member)`` gives ``"TransformParameterizations.Affine"``. Matching on
    that would silently miss every enum-built transform.
    """
    from itkwasm import TransformParameterizations
    from ngff_zarr.itk_transform_to_ngff_transform import _parameterization_name

    member = TransformParameterizations.Affine
    assert str(member) != "Affine"
    assert (
        _parameterization_name(type("T", (), {"transformParameterization": member}))
        == "Affine"
    )
    # A plain string must work identically.
    assert (
        _parameterization_name(type("T", (), {"transformParameterization": "Affine"}))
        == "Affine"
    )


def test_affine_decoding_is_exact_for_an_enum_built_transform():
    """Decoding must read the parameters, not re-derive them.

    Falling back to probing the mapping would land within a few ULP rather than
    reproducing the input exactly, so this asserts bit equality.
    """
    matrix = [0.90, 0.12, -0.30, -0.22, 1.05, 0.17, 0.08, -0.41, 0.95]
    translation = [3.5, -1.25, 2.0]
    transform = _itkwasm_affine(matrix, translation, [0.0, 0.0, 0.0])

    got_matrix, got_offset = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))

    reversal = np.eye(3)[::-1]
    expected_matrix = reversal @ np.asarray(matrix).reshape(3, 3) @ reversal
    expected_offset = reversal @ np.asarray(translation)
    assert np.array_equal(got_matrix, expected_matrix)
    assert np.array_equal(got_offset, expected_offset)


def test_matrix_parameterizations_decode_without_itk(monkeypatch):
    """The matrix-carrying types must not need ``itk`` to decode.

    They were reaching the ``itk`` fallback, which both required an optional
    dependency and reported a misleading reason for an affine.
    """
    import builtins

    real_import = builtins.__import__

    def no_itk(name, *args, **kwargs):
        if name == "itk":
            raise ImportError("itk is unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_itk)

    transform = _itkwasm_affine(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]
    )
    _, offset = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))
    assert np.array_equal(offset, [3.0, 2.0, 1.0])


def test_this_ports_own_output_decodes_without_itk(monkeypatch):
    """``ngff_transform_to_itk_transform`` emits the enum, so the pair must agree."""
    import builtins

    real_import = builtins.__import__

    def no_itk(name, *args, **kwargs):
        if name == "itk":
            raise ImportError("itk is unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_itk)

    dims = ("y", "x")
    emitted = ngff_transform_to_itk_transform(
        Affine(affine=[[0.8, -0.6, 10.0], [0.6, 0.8, -4.0]]), dims
    )
    converted = itk_transform_to_ngff_transform(emitted, dims)
    assert isinstance(converted, Affine)
    assert np.allclose(converted.affine, [[0.8, -0.6, 10.0], [0.6, 0.8, -4.0]])


@pytest.mark.parametrize(
    ("parameterization", "parameters", "fixed", "dims"),
    [
        # 3D entries against a 2D coordinate system: slicing would silently
        # project, or scramble an affine's rows into a singular matrix.
        ("Affine", [2, 0, 0, 0, 3, 0, 0, 0, 4, 10, 20, 30], [0, 0, 0], ("y", "x")),
        ("Translation", [10, 20, 30], [], ("y", "x")),
        # 2D entries against a 3D coordinate system: reading past the end.
        ("Affine", [2, 0, 0, 3, 10, 20], [0, 0], ("z", "y", "x")),
        ("Scale", [2, 3], [], ("z", "y", "x")),
        # A single parameter would broadcast across every axis.
        ("Translation", [5], [], ("y", "x")),
        # No parameters at all (a default-constructed entry).
        ("Affine", [], [], ("y", "x")),
    ],
)
def test_mismatched_itkwasm_dimensionality_is_rejected(
    parameterization, parameters, fixed, dims
):
    """The parameter count is the ground truth for an entry's dimensionality.

    Without this check a 3D affine decoded against 2D dims yields a singular
    matrix that ``_simplify`` dresses up as a zero ``scale``, and a 2D affine
    against 3D dims pads itself with garbage -- both written into store
    metadata without a word.
    """
    entry = _itkwasm_entry(parameterization, parameters, fixed)
    with pytest.raises(ValueError, match="does not match"):
        itk_transform_to_ngff_matrix(entry, dims)


def test_mismatched_angle_parameterization_dimensionality_is_rejected():
    """Angle entries carry a type-specific count, so the declared dimension
    is checked instead -- and before the ``itk`` import, so the error does not
    depend on the optional extra."""
    entry = _itkwasm_entry(
        "Euler3D", [0.1, 0.2, 0.3, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0], dimension=3
    )
    with pytest.raises(ValueError, match="3D, but the coordinate system has 2"):
        itk_transform_to_ngff_matrix(entry, ("y", "x"))


@pytest.mark.parametrize("position", [0, 1])
def test_a_composite_entry_is_refused_at_any_position(position):
    """A parameterless 'Composite' entry is ambiguous, so it is refused.

    The ITK-Wasm pipeline writes one as a grouping header before the
    children, but ``itk.dict_from_transform`` never writes a header at all:
    there it is a *nested* composite whose children the serialization
    dropped, at any position including the first. Skipping a leading one --
    the obvious 'header' reading -- silently returns the mapping minus the
    lost children, which is exactly the failure mode this module exists to
    avoid.
    """
    entries = [
        _itkwasm_entry("Translation", [10.0, 0.0], []),
        _itkwasm_entry("Affine", [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0]),
    ]
    entries.insert(position, _itkwasm_entry("Composite", [], []))

    with pytest.raises(NotImplementedError, match="'Composite' entry"):
        itk_transform_to_ngff_matrix(entries, ("y", "x"))


def test_a_nested_native_composite_converts_exactly():
    """The safe route for any composite, nested included: the native transform
    still holds its children, so no serialization can drop them."""
    itk = pytest.importorskip("itk")

    inner = itk.CompositeTransform[itk.D, 2].New()
    shift = itk.TranslationTransform[itk.D, 2].New()
    shift.SetOffset([1.0, 2.0])
    inner.AddTransform(shift)
    outer = itk.CompositeTransform[itk.D, 2].New()
    outer.AddTransform(inner)
    second = itk.TranslationTransform[itk.D, 2].New()
    second.SetOffset([100.0, 0.0])
    outer.AddTransform(second)

    matrix, offset = itk_transform_to_ngff_matrix(outer, ("y", "x"))

    assert np.array_equal(matrix, np.eye(2))
    assert np.allclose(offset, [2.0, 101.0])


@pytest.mark.parametrize(
    ("transform", "match"),
    [
        (Affine(affine=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), "affine transformation is 6"),
        (Rotation(rotation=[0.0, -1.0, 1.0, 0.0]), "rotation transformation is 4"),
    ],
)
def test_a_flat_parameter_array_gets_the_intended_error(transform, match):
    """A flat array is an easy hand-construction slip, since the docs
    describe the affine as 'the upper M x (N+1) block'. The shape in the
    message is formatted from the whole shape tuple, so a 1-D array gets the
    intended ValueError rather than an IndexError raised while formatting
    it."""
    from ngff_zarr.ngff_transform_to_itk_transform import (
        _ngff_transform_to_itk_matrix,
    )

    with pytest.raises(ValueError, match=match):
        _ngff_transform_to_itk_matrix(transform, ("y", "x"))


def test_an_itkwasm_identity_entry_decodes_to_identity():
    """The Identity decode branch, exercised directly in each port."""
    matrix, offset = itk_transform_to_ngff_matrix(
        _itkwasm_entry("Identity", [], []), ("y", "x")
    )
    assert np.array_equal(matrix, np.eye(2))
    assert np.array_equal(offset, np.zeros(2))


def test_an_empty_transform_list_is_rejected():
    with pytest.raises(ValueError, match="transform list is empty"):
        itk_transform_to_ngff_matrix([], ("y", "x"))


def test_dims_without_spatial_axes_are_rejected():
    with pytest.raises(ValueError, match="no spatial axes"):
        itk_transform_to_ngff_matrix(_itkwasm_entry("Identity", [], []), ("t", "c"))


def test_an_itkwasm_affine_center_is_folded_into_the_offset():
    """The decode branch's own center algebra, not the probing path's.

    Every other Affine entry in this file carries a zero center, so only
    this test observes the fold ``b = t + c - A c`` in the decode branch.
    """
    matrix = [[0.8, -0.6], [0.6, 0.8]]
    center = [9.5, 4.5]
    translation = [10.0, 10.0]
    entry = _itkwasm_entry("Affine", [*matrix[0], *matrix[1], *translation], center)

    _, offset = itk_transform_to_ngff_matrix(entry, ("y", "x"))

    expected_itk = (
        np.asarray(translation)
        + np.asarray(center)
        - np.asarray(matrix) @ np.asarray(center)
    )
    assert np.allclose(offset, expected_itk[::-1])


def test_non_canonical_spatial_order_converts_by_name():
    """The converters bind ITK components to axes by name, like the bounding
    box: an ITK translation (x=7, y=5, z=1) lands on the axes so named
    whatever order ``dims`` spells them in."""
    entry = _itkwasm_entry("Translation", [7.0, 5.0, 1.0], [], dimension=3)

    for dims in [("z", "y", "x"), ("z", "x", "y"), ("x", "y", "z")]:
        _, offset = itk_transform_to_ngff_matrix(entry, dims)
        named = dict(zip(dims, offset))
        assert named == {"x": 7.0, "y": 5.0, "z": 1.0}, dims

    # And the reverse direction inverts it exactly, in every spelling.
    from ngff_zarr.ngff_transform_to_itk_transform import (
        _ngff_transform_to_itk_matrix,
    )

    for dims in [("z", "y", "x"), ("z", "x", "y"), ("x", "y", "z")]:
        translation = Translation(
            translation=[{"x": 7.0, "y": 5.0, "z": 1.0}[d] for d in dims]
        )
        _, itk_offset = _ngff_transform_to_itk_matrix(translation, dims)
        assert np.allclose(itk_offset, [7.0, 5.0, 1.0]), dims


def test_itkwasm_transform_list_composes_last_entry_first():
    """An ITK transform list applies its last entry first."""
    from itkwasm import (
        FloatTypes,
        Transform,
        TransformParameterizations,
        TransformType,
    )

    def entry(parameterization, parameters, fixed=()):
        transform_type = TransformType(
            transformParameterization=parameterization,
            parametersValueType=FloatTypes.Float64,
            inputDimension=2,
            outputDimension=2,
        )
        return Transform(
            transformType=transform_type,
            numberOfFixedParameters=len(fixed),
            numberOfParameters=len(parameters),
            fixedParameters=np.asarray(fixed, dtype=np.float64),
            parameters=np.asarray(parameters, dtype=np.float64),
        )

    # ITK order (x, y): translate x by 10, and scale by 2.
    shift = entry(TransformParameterizations.Translation, [10.0, 0.0])
    scaling = entry(
        TransformParameterizations.Affine, [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0]
    )

    # [shift, scaling] means shift(scaling(p)) = 2p + 10.
    matrix, offset = itk_transform_to_ngff_matrix([shift, scaling], ("y", "x"))
    assert np.allclose(matrix, np.diag([2.0, 2.0]))
    assert np.allclose(offset, [0.0, 10.0])  # reversed to (y, x)

    # The other order means scaling(shift(p)) = 2(p + 10) = 2p + 20.
    _, reversed_offset = itk_transform_to_ngff_matrix([scaling, shift], ("y", "x"))
    assert np.allclose(reversed_offset, [0.0, 20.0])


def _oriented_image(shape, scale, translation, orientations="RAS"):
    """A geometry-only 3D NgffImage.

    ``"RAS"`` gives the diagonal direction ``diag(-1, -1, 1)``, which is its
    own inverse. ``"cycle"`` maps z, y, x onto LPS x, z, y: a 3-cycle whose
    direction matrix is not symmetric, so it tells ``D`` from ``D^-1``.
    """
    import dask.array as da
    from ngff_zarr import NgffImage
    from ngff_zarr.rfc4 import RAS, AnatomicalOrientation, AnatomicalOrientationValues

    if orientations == "RAS":
        axes_orientations = RAS
    elif orientations == "cycle":
        axes_orientations = {
            "z": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
            "y": AnatomicalOrientation(
                value=AnatomicalOrientationValues.inferior_to_superior
            ),
            "x": AnatomicalOrientation(
                value=AnatomicalOrientationValues.posterior_to_anterior
            ),
        }
    else:
        axes_orientations = None
    return NgffImage(
        data=da.zeros(shape, dtype=np.uint8),
        dims=["z", "y", "x"],
        scale=scale,
        translation=translation,
        axes_orientations=axes_orientations,
    )


def _sheared_itk_affine(itk):
    """An affine with rotation, shear and translation: every frame error shows."""
    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetMatrix(
        itk.matrix_from_array(
            np.array([[0.9, 0.1, 0.0], [-0.1, 1.1, 0.0], [0.0, 0.0, 1.0]])
        )
    )
    transform.SetTranslation([2.0, -3.0, 4.0])
    transform.SetCenter([0.0, 0.0, 0.0])
    return transform


def test_conversion_with_frames_matches_the_itk_path_on_oriented_images():
    """The acceptance test for the change of frame.

    An ITK transform acts on physical space, direction matrix included; the
    RFC-5 branch of the bounding box acts on the intrinsic systems. Convert
    with ``fixed=``/``moving=`` and the two paths must select the same region
    of the same RAS-oriented images. Without the frames they are 25 voxels
    apart in y on this geometry.
    """
    itk = pytest.importorskip("itk")

    from ngff_zarr import resample_bounding_box

    # Fractional translations keep every corner away from an integer, so the
    # comparison cannot ride a floor/ceil knife edge.
    fixed = _oriented_image(
        (8, 8, 8),
        {"z": 1.0, "y": 2.0, "x": 3.0},
        {"z": 1.3, "y": -2.7, "x": 5.1},
    )
    moving = _oriented_image(
        (256, 256, 256),
        {"z": 1.0, "y": 1.0, "x": 1.0},
        {"z": -4.2, "y": 7.6, "x": 3.4},
    )
    transform = _sheared_itk_affine(itk)

    via_itk = resample_bounding_box(transform, fixed, moving, padding=0)
    converted = itk_transform_to_ngff_transform(
        transform, ("z", "y", "x"), fixed=fixed, moving=moving
    )
    via_rfc5 = resample_bounding_box(converted, fixed, moving, padding=0)

    assert via_rfc5.start_index == via_itk.start_index
    assert via_rfc5.size == via_itk.size


def test_conversion_with_frames_preserves_the_rotation_sign():
    """Conjugation by diag(-1, -1, 1) transposes a rotation that mixes a
    flipped axis with an unflipped one: a +30 degree registration about the
    ITK x axis is persisted as -30 degrees unless the frames are given.

    (A rotation about z would not show it: both flipped axes lie in its
    plane, so the conjugation is the identity there.)
    """
    itk = pytest.importorskip("itk")

    angle = np.pi / 6
    # About the ITK x axis: mixes y (flipped by RAS) with z (not flipped).
    rotation_itk = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetMatrix(itk.matrix_from_array(rotation_itk))
    transform.SetTranslation([0.0, 0.0, 0.0])
    transform.SetCenter([0.0, 0.0, 0.0])

    fixed = _oriented_image(
        (8, 8, 8), dict.fromkeys("zyx", 1.0), dict.fromkeys("zyx", 0.0)
    )
    moving = _oriented_image(
        (8, 8, 8), dict.fromkeys("zyx", 1.0), dict.fromkeys("zyx", 0.0)
    )

    with_frames, _ = itk_transform_to_ngff_matrix(
        transform, ("z", "y", "x"), fixed=fixed, moving=moving
    )
    frameless, _ = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))

    reversal = np.eye(3)[::-1]
    flip = np.diag([-1.0, -1.0, 1.0])
    conjugated = reversal @ (flip @ rotation_itk @ flip) @ reversal
    assert np.allclose(with_frames, conjugated)
    # Without the frames the numbers are copied unconjugated, which for this
    # rotation is exactly the transpose: the opposite angle.
    assert np.allclose(frameless, reversal @ rotation_itk @ reversal)
    assert np.allclose(with_frames, frameless.T)
    assert not np.allclose(with_frames, frameless)


def test_frames_round_trip_through_both_converters():
    """RFC-5 -> ITK -> RFC-5 with frames must return the identical mapping.

    On a 3-cycle orientation: a diagonal direction such as RAS is its own
    inverse, so it cannot tell the reverse conversion's ``D^-1`` from ``D``.
    """
    pytest.importorskip("itk")

    from ngff_zarr import ngff_transform_to_itk_transform

    fixed = _oriented_image(
        (8, 8, 8),
        {"z": 1.0, "y": 2.0, "x": 3.0},
        {"z": 1.3, "y": -2.7, "x": 5.1},
        "cycle",
    )
    moving = _oriented_image(
        (16, 16, 16),
        {"z": 1.0, "y": 1.0, "x": 1.0},
        {"z": -4.2, "y": 7.6, "x": 3.4},
        "cycle",
    )
    original = Affine(
        affine=[
            [1.0, 0.2, 0.0, 4.1],
            [0.0, 2.0, 0.3, -6.2],
            [0.5, 0.0, 1.0, 11.3],
        ]
    )

    itk_list = ngff_transform_to_itk_transform(
        original, ("z", "y", "x"), fixed=fixed, moving=moving
    )
    back = itk_transform_to_ngff_transform(
        itk_list, ("z", "y", "x"), simplify=False, fixed=fixed, moving=moving
    )

    assert np.allclose(back.affine, original.affine)


def test_an_out_of_plane_orientation_falls_back_to_identity_direction():
    """A 2D image whose axis points along LPS z would truncate to a singular
    direction matrix; the all-or-nothing rule keeps the identity instead, so
    frame conversion behaves as for an unoriented image rather than raising
    LinAlgError from the inverse."""
    itk = pytest.importorskip("itk")

    import dask.array as da
    from ngff_zarr import NgffImage
    from ngff_zarr.rfc4 import AnatomicalOrientation, AnatomicalOrientationValues

    oriented = NgffImage(
        data=da.zeros((8, 8), dtype=np.uint8),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
        axes_orientations={
            "y": AnatomicalOrientation(
                value=AnatomicalOrientationValues.superior_to_inferior
            ),
            "x": AnatomicalOrientation(value=AnatomicalOrientationValues.left_to_right),
        },
    )
    transform = itk.TranslationTransform[itk.D, 2].New()
    transform.SetOffset([3.0, 5.0])

    with_frames = itk_transform_to_ngff_matrix(
        transform, ("y", "x"), fixed=oriented, moving=oriented
    )
    frameless = itk_transform_to_ngff_matrix(transform, ("y", "x"))

    assert np.allclose(with_frames[0], frameless[0])
    assert np.allclose(with_frames[1], frameless[1])


def test_frames_are_all_or_nothing_and_a_noop_when_unoriented():
    itk = pytest.importorskip("itk")

    transform = _sheared_itk_affine(itk)
    plain = _oriented_image(
        (8, 8, 8), dict.fromkeys("zyx", 1.0), dict.fromkeys("zyx", 0.5), None
    )

    with pytest.raises(ValueError, match="both fixed and moving"):
        itk_transform_to_ngff_matrix(transform, ("z", "y", "x"), fixed=plain)

    with_frames = itk_transform_to_ngff_matrix(
        transform, ("z", "y", "x"), fixed=plain, moving=plain
    )
    without = itk_transform_to_ngff_matrix(transform, ("z", "y", "x"))
    assert np.allclose(with_frames[0], without[0])
    assert np.allclose(with_frames[1], without[1])


def test_registration_result_can_be_attached_to_multiscales():
    """The point of this direction: persist a registration into the store."""
    itk = pytest.importorskip("itk")
    import dask.array as da
    import ngff_zarr as nz

    image = nz.to_ngff_image(da.zeros((32, 32), dtype=np.uint8), dims=["y", "x"])
    multiscales = nz.to_multiscales(image, scale_factors=[2], chunks=16)

    euler = itk.Euler2DTransform[itk.D].New()
    euler.SetAngle(0.2)
    euler.SetTranslation([3.0, -1.0])

    transform = itk_transform_to_ngff_transform(euler, ["y", "x"])

    assert isinstance(transform, Affine)
    # An RFC-5 affine is M rows of N+1 columns, translation last.
    assert len(transform.affine) == 2
    assert all(len(row) == 3 for row in transform.affine)
    assert transform.type == "affine"
    # It is shaped to go straight onto the multiscales metadata.
    assert multiscales.metadata.coordinateTransformations is None
