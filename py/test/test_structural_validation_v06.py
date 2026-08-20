# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Structural validation of the v0.6 ``coordinateSystems`` metadata model.

At v0.6 (RFC-5) the axes live in ``coordinateSystems`` and each dataset carries
a single transformation -- an ``identity``, a ``scale``, or a ``sequence`` of a
``scale`` and a ``translation`` -- mapping its array to the intrinsic
coordinate system. :func:`~ngff_zarr.validate_structural` reduces that model to
the flat v0.4 shape its rules are written against, so the same rules apply.

This matters for every version: ``from_ngff_zarr`` and ``to_multiscales``
normalize to the v0.6 model whatever the store holds, so the metadata handed to
callers is the coordinate-system one even for a v0.4 store.
"""

from dataclasses import asdict

import numpy as np
import pytest
import zarr
from ngff_zarr import (
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
    validate_structural,
)
from ngff_zarr.structural_validation import SpecRule, ValidationError
from ngff_zarr.v06.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Dataset,
    Identity,
    Metadata,
    Scale,
    TransformSequence,
    Translation,
)
from packaging import version as packaging_version

#: Writing OME-Zarr 0.5 and later needs a Zarr v3 hierarchy.
needs_zarr_v3 = pytest.mark.skipif(
    packaging_version.parse(zarr.__version__) < packaging_version.parse("3.0.0b2"),
    reason="zarr version >= 3.0.0b2 required for OME-Zarr version >= 0.5",
)

INTRINSIC = "intrinsic"


def _axes(names):
    return [Axis(name=name, type="space") for name in names]


def _sequence(path, scale, translation, output=INTRINSIC):
    """The dataset transform ngff-zarr writes at v0.6: scale, then translation."""
    return TransformSequence(
        transformations=[Scale(scale=scale), Translation(translation=translation)],
        input=CoordinateSystemIdentifier(path=path),
        output=CoordinateSystemIdentifier(name=output),
        name=f"{path}_to_{output}",
    )


def _metadata(
    datasets, axes=None, coordinate_systems=(), coordinate_transformations=None
):
    """v0.6 metadata over an intrinsic ``(z, y, x)`` system, unless overridden."""
    intrinsic = CoordinateSystem(name=INTRINSIC, axes=axes or _axes("zyx"))
    return Metadata(
        coordinateSystems=[intrinsic, *coordinate_systems],
        datasets=datasets,
        coordinateTransformations=coordinate_transformations,
    )


def _two_level_metadata(axes=None, scales=((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))):
    zeros = [0.0] * len(scales[0])
    return _metadata(
        [
            Dataset(
                path=str(index),
                coordinateTransformations=[
                    _sequence(str(index), list(scale), list(zeros))
                ],
            )
            for index, scale in enumerate(scales)
        ],
        axes=axes,
    )


# ── the model is accepted ──────────────────────────────────────────────────


def test_validate_structural_accepts_the_v06_model():
    # The regression under test: the axis rules read the intrinsic coordinate
    # system's axes instead of raising AttributeError on a missing `axes`.
    validate_structural(_two_level_metadata())


def test_identity_dataset_transform_is_accepted():
    # v0.6 permits a bare identity where v0.4 requires a scale; the reduction
    # makes the unit scale and zero translation it implies explicit, so the
    # per-dataset scale-count rule sees exactly one scale.
    metadata = _metadata(
        [
            Dataset(
                path="0",
                coordinateTransformations=[
                    Identity(
                        input=CoordinateSystemIdentifier(path="0"),
                        output=CoordinateSystemIdentifier(name=INTRINSIC),
                    )
                ],
            )
        ]
    )
    validate_structural(metadata)


def test_bare_scale_dataset_transform_is_accepted():
    # The other single-transform shape the v0.6 schema permits.
    metadata = _metadata(
        [
            Dataset(
                path="0",
                coordinateTransformations=[
                    Scale(
                        scale=[1.0, 1.0, 1.0],
                        input=CoordinateSystemIdentifier(path="0"),
                        output=CoordinateSystemIdentifier(name=INTRINSIC),
                    )
                ],
            )
        ]
    )
    validate_structural(metadata)


def _custom_sequence(path, transformations, output=INTRINSIC):
    """A dataset sequence carrying whatever transformations are given."""
    return TransformSequence(
        transformations=list(transformations),
        input=CoordinateSystemIdentifier(path=path),
        output=CoordinateSystemIdentifier(name=output),
        name=f"{path}_to_{output}",
    )


@pytest.mark.parametrize(
    "transformations, rule",
    [
        pytest.param(
            [Translation(translation=[0.0, 0.0, 0.0])],
            SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,
            id="no-scale",
        ),
        pytest.param(
            [Scale(scale=[1.0, 1.0, 1.0]), Scale(scale=[2.0, 2.0, 2.0])],
            SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,
            id="two-scales",
        ),
        pytest.param(
            [
                Translation(translation=[0.0, 0.0, 0.0]),
                Scale(scale=[1.0, 1.0, 1.0]),
            ],
            SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,
            id="translation-before-scale",
        ),
    ],
)
def test_malformed_dataset_sequences_are_rejected(transformations, rule):
    """The reduction renders the source sequence, it does not repair it.

    A dataset sequence with no scale, with two, or with its translation ahead
    of its scale is what ``validate_per_dataset_scale_count`` and
    ``validate_transform_order`` exist to catch. Synthesizing a well-formed
    scale-then-translation pair would make both inert at v0.6.
    """
    metadata = _metadata(
        [
            Dataset(
                path="0",
                coordinateTransformations=[_custom_sequence("0", transformations)],
            )
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_structural(metadata)
    assert exc_info.value.rule == rule


def test_axes_property_is_not_a_field():
    # `axes` must stay a property: a field would be serialized into the
    # multiscales entry, which carries `coordinateSystems` at v0.6.
    metadata = _two_level_metadata()

    assert "axes" not in asdict(metadata)
    assert metadata.axes == metadata.coordinateSystems[0].axes


# ── the rules still fire through the reduction ─────────────────────────────


def test_scale_length_is_checked_inside_the_sequence():
    metadata = _two_level_metadata(scales=((1.0, 1.0, 1.0), (2.0, 2.0)))

    with pytest.raises(ValidationError) as excinfo:
        validate_structural(metadata)

    assert excinfo.value.rule == SpecRule.SCALE_LENGTH_MISMATCH
    assert (
        excinfo.value.location
        == "multiscales[0].datasets[1].coordinateTransformations[0]"
    )


def test_dataset_order_is_checked_inside_the_sequence():
    metadata = _two_level_metadata(scales=((4.0, 4.0, 4.0), (1.0, 1.0, 1.0)))

    with pytest.raises(ValidationError) as excinfo:
        validate_structural(metadata)

    assert excinfo.value.rule == SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST
    assert excinfo.value.location == "multiscales[0].datasets[1]"


def test_axis_count_is_read_from_the_intrinsic_system():
    metadata = _two_level_metadata(
        axes=_axes("abcdef"),
        scales=((1.0,) * 6, (2.0,) * 6),
    )

    with pytest.raises(ValidationError) as excinfo:
        validate_structural(metadata)

    assert excinfo.value.rule == SpecRule.AXIS_COUNT
    assert excinfo.value.location == "multiscales[0].axes"


def test_top_level_inter_system_transform_is_not_length_checked():
    # A v0.6 multiscale-level transform maps between named coordinate systems,
    # so its vectors are sized by *those* systems -- measuring them against the
    # intrinsic axis count would be a false positive.
    other = CoordinateSystem(name="other", axes=_axes("yx"))
    metadata = _metadata(
        [
            Dataset(
                path="0",
                coordinateTransformations=[_sequence("0", [1.0] * 3, [0.0] * 3)],
            )
        ],
        coordinate_systems=(other,),
        coordinate_transformations=[
            Scale(
                scale=[2.0, 2.0],
                input=CoordinateSystemIdentifier(name="other"),
                output=CoordinateSystemIdentifier(name="other"),
            )
        ],
    )

    validate_structural(metadata)


# ── end to end, through a store ────────────────────────────────────────────


@pytest.mark.parametrize(
    "version",
    [
        "0.4",
        pytest.param("0.5", marks=needs_zarr_v3),
        pytest.param("0.6", marks=needs_zarr_v3),
    ],
)
def test_validate_structural_after_round_trip(tmp_path, version):
    # Every version reads back as the v0.6 model, so this is the same code path
    # each time; before the reduction it raised AttributeError.
    image = to_ngff_image(np.zeros((4, 8, 8), dtype=np.float32), dims=["z", "y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2])
    store = tmp_path / f"image-{version}.ome.zarr"
    to_ngff_zarr(str(store), multiscales, version=version)

    metadata = from_ngff_zarr(str(store)).metadata

    validate_structural(metadata)
