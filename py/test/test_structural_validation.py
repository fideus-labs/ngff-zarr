# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`ngff_zarr.structural_validation`.

One paired pass/fail test per structural rule: a shared, fully valid
``[c, y, x]`` two-level baseline (:func:`build_valid_metadata`) is mutated in
exactly one field to trip exactly one rule, asserting the raised
:class:`~ngff_zarr.structural_validation.ValidationError` carries the expected
:class:`~ngff_zarr.structural_validation.SpecRule` and a populated ``location``.
Each rule function is exercised directly so the violation is isolated to that
rule. Additional tests cover the public surface (enum values, option defaults,
error formatting) and the orchestrator's fail-fast ordering and the
``SCHEMA_ONLY`` short-circuit.
"""

import pytest
from ngff_zarr.structural_validation import (
    SpecRule,
    ValidateOptions,
    ValidationError,
    ValidationLevel,
    validate_axis_count,
    validate_axis_order,
    validate_axis_type,
    validate_dataset_order,
    validate_omero_color_hex,
    validate_per_dataset_scale_count,
    validate_scale_length,
    validate_spatial_axis_order,
    validate_structural,
    validate_transform_order,
)
from ngff_zarr.v04.zarr_metadata import (
    Axis,
    Dataset,
    Metadata,
    Omero,
    OmeroChannel,
    OmeroWindow,
    Scale,
    Translation,
)


def build_valid_metadata() -> Metadata:
    """Return a fully valid v0.4 ``[c, y, x]`` two-level multiscales metadata.

    Three axes (one channel, two space); two datasets, each with a single
    length-3 ``scale``; the second level is coarser than the first. This
    baseline passes every structural rule, so each test mutates exactly one
    field of a fresh copy to trip exactly one rule.
    """
    return Metadata(
        axes=[
            Axis(name="c", type="channel"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[Scale([1.0, 1.0, 1.0])],
            ),
            Dataset(
                path="1",
                coordinateTransformations=[Scale([1.0, 2.0, 2.0])],
            ),
        ],
        coordinateTransformations=None,
    )


@pytest.fixture
def valid_metadata() -> Metadata:
    """A fresh, fully valid baseline metadata for each test."""
    return build_valid_metadata()


def _omero(color: str) -> Omero:
    """Build a single-channel OMERO block with the given channel ``color``."""
    return Omero(
        channels=[
            OmeroChannel(
                color=color,
                window=OmeroWindow(min=0.0, max=255.0, start=0.0, end=255.0),
            )
        ]
    )


# ---------------------------------------------------------------------------
# Per-rule paired pass/fail tests
# ---------------------------------------------------------------------------


def test_axis_count_valid(valid_metadata):
    validate_axis_count(valid_metadata)


@pytest.mark.parametrize("count", [1, 6])
def test_axis_count_invalid(valid_metadata, count):
    valid_metadata.axes = [Axis(name="x", type="space") for _ in range(count)]
    with pytest.raises(ValidationError) as exc_info:
        validate_axis_count(valid_metadata)
    assert exc_info.value.rule == SpecRule.AXIS_COUNT
    assert exc_info.value.location == "multiscales[0].axes"


def test_axis_type_valid(valid_metadata):
    validate_axis_type(valid_metadata)


@pytest.mark.parametrize("axis_type", ["time", "channel"])
def test_axis_type_invalid(valid_metadata, axis_type):
    # Two axes of the same singleton type (time or channel) is not allowed.
    name = "t" if axis_type == "time" else "c"
    valid_metadata.axes = [
        Axis(name=name, type=axis_type),
        Axis(name=name, type=axis_type),
        Axis(name="y", type="space"),
        Axis(name="x", type="space"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        validate_axis_type(valid_metadata)
    assert exc_info.value.rule == SpecRule.AXIS_TYPE
    assert exc_info.value.location == "multiscales[0].axes"


def test_axis_order_class_valid(valid_metadata):
    validate_axis_order(valid_metadata)


def test_axis_order_class_invalid(valid_metadata):
    # A 'time' axis after a 'channel' axis inverts the time < channel ranking.
    valid_metadata.axes = [
        Axis(name="c", type="channel"),
        Axis(name="t", type="time"),
        Axis(name="y", type="space"),
        Axis(name="x", type="space"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        validate_axis_order(valid_metadata)
    assert exc_info.value.rule == SpecRule.AXIS_ORDER
    assert exc_info.value.location == "multiscales[0].axes[1]"


def test_spatial_axis_order_valid(valid_metadata):
    validate_spatial_axis_order(valid_metadata)


@pytest.mark.parametrize(
    "case, location",
    [
        ("suffix-mismatch", "multiscales[0].axes[1]"),
        ("too-many-spatial", "multiscales[0].axes"),
    ],
)
def test_spatial_axis_order_invalid(valid_metadata, case, location):
    if case == "suffix-mismatch":
        # (x, y) is not the (y, x) suffix of (z, y, x).
        valid_metadata.axes = [
            Axis(name="c", type="channel"),
            Axis(name="x", type="space"),
            Axis(name="y", type="space"),
        ]
    else:
        # Four space axes exceed the v0.4 maximum of three.
        valid_metadata.axes = [
            Axis(name="z", type="space"),
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ]
    with pytest.raises(ValidationError) as exc_info:
        validate_spatial_axis_order(valid_metadata)
    assert exc_info.value.rule == SpecRule.AXIS_ORDER
    assert exc_info.value.location == location


def test_per_dataset_scale_count_valid(valid_metadata):
    validate_per_dataset_scale_count(valid_metadata)


@pytest.mark.parametrize("case", ["zero-scales", "two-scales"])
def test_per_dataset_scale_count_invalid(valid_metadata, case):
    if case == "zero-scales":
        valid_metadata.datasets[0].coordinateTransformations = [
            Translation([0.0, 0.0, 0.0]),
        ]
    else:
        valid_metadata.datasets[0].coordinateTransformations = [
            Scale([1.0, 1.0, 1.0]),
            Scale([1.0, 1.0, 1.0]),
        ]
    with pytest.raises(ValidationError) as exc_info:
        validate_per_dataset_scale_count(valid_metadata)
    assert exc_info.value.rule == SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL
    assert (
        exc_info.value.location
        == "multiscales[0].datasets[0].coordinateTransformations"
    )


def test_scale_length_valid(valid_metadata):
    validate_scale_length(valid_metadata)
    # A correctly sized global scale (length == axis count) is also accepted.
    valid_metadata.coordinateTransformations = [Scale([1.0, 1.0, 1.0])]
    validate_scale_length(valid_metadata)


def test_scale_length_invalid_per_dataset(valid_metadata):
    # A length-2 dataset scale against 3 axes.
    valid_metadata.datasets[0].coordinateTransformations = [Scale([1.0, 1.0])]
    with pytest.raises(ValidationError) as exc_info:
        validate_scale_length(valid_metadata)
    assert exc_info.value.rule == SpecRule.SCALE_LENGTH_MISMATCH
    assert (
        exc_info.value.location
        == "multiscales[0].datasets[0].coordinateTransformations[0]"
    )


def test_scale_length_invalid_global(valid_metadata):
    # A length-2 global scale against 3 axes.
    valid_metadata.coordinateTransformations = [Scale([1.0, 1.0])]
    with pytest.raises(ValidationError) as exc_info:
        validate_scale_length(valid_metadata)
    assert exc_info.value.rule == SpecRule.SCALE_LENGTH_MISMATCH
    assert exc_info.value.location == "multiscales[0].coordinateTransformations[0]"


def test_transform_order_valid(valid_metadata):
    validate_transform_order(valid_metadata)
    # A translation following its scale is valid.
    valid_metadata.datasets[0].coordinateTransformations = [
        Scale([1.0, 1.0, 1.0]),
        Translation([0.0, 0.0, 0.0]),
    ]
    validate_transform_order(valid_metadata)


def test_transform_order_invalid(valid_metadata):
    # A scale after a translation: a translation must follow its scale.
    valid_metadata.datasets[0].coordinateTransformations = [
        Translation([0.0, 0.0, 0.0]),
        Scale([1.0, 1.0, 1.0]),
    ]
    with pytest.raises(ValidationError) as exc_info:
        validate_transform_order(valid_metadata)
    assert exc_info.value.rule == SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL
    assert (
        exc_info.value.location
        == "multiscales[0].datasets[0].coordinateTransformations[1]"
    )


def test_dataset_order_valid(valid_metadata):
    validate_dataset_order(valid_metadata)


def test_dataset_order_invalid(valid_metadata):
    # The second (coarser) level has a finer spatial scale than the first.
    valid_metadata.datasets[0].coordinateTransformations = [Scale([1.0, 2.0, 2.0])]
    valid_metadata.datasets[1].coordinateTransformations = [Scale([1.0, 1.0, 1.0])]
    with pytest.raises(ValidationError) as exc_info:
        validate_dataset_order(valid_metadata)
    assert exc_info.value.rule == SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST
    assert exc_info.value.location == "multiscales[0].datasets[1]"
    # Float scales render as `1.0`/`2.0` (not `1`/`2`); the TypeScript port
    # matches this byte-for-byte via its pyFloat() helper (locks parity).
    assert exc_info.value.message == (
        "Datasets must be ordered finest to coarsest; dataset 1 has spatial "
        "scale 1.0 on axis 'y', smaller than dataset 0's 2.0."
    )


def test_omero_color_valid(valid_metadata):
    validate_omero_color_hex(valid_metadata)  # omero is None
    valid_metadata.omero = _omero("00FF88")
    validate_omero_color_hex(valid_metadata)


def test_omero_color_invalid(valid_metadata):
    valid_metadata.omero = _omero("xyz")
    with pytest.raises(ValidationError) as exc_info:
        validate_omero_color_hex(valid_metadata)
    assert exc_info.value.rule == SpecRule.OMERO_CHANNEL_COLOR_FORMAT
    assert exc_info.value.location == "multiscales[0].omero.channels[0].color"


# ---------------------------------------------------------------------------
# Public-surface tests
# ---------------------------------------------------------------------------


def test_spec_rule_values():
    assert SpecRule.AXIS_COUNT.value == "axis-count"
    assert SpecRule.AXIS_TYPE.value == "axis-type"
    assert SpecRule.AXIS_ORDER.value == "axis-order"
    assert SpecRule.SCALE_LENGTH_MISMATCH.value == "scale-length-mismatch"
    assert (
        SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL.value
        == "global-coord-transform-after-per-level"
    )
    assert (
        SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST.value
        == "dataset-order-highest-to-lowest"
    )
    assert SpecRule.OMERO_CHANNEL_COLOR_FORMAT.value == "omero-channel-color-format"
    # Inheriting from str makes the kebab identifier the value for comparison.
    assert SpecRule.AXIS_COUNT == "axis-count"


def test_validation_level_default_is_strict():
    assert ValidateOptions().level == ValidationLevel.STRICT
    assert ValidationLevel.STRICT.value == "strict"
    assert ValidationLevel.SCHEMA_ONLY.value == "schema_only"


def test_validate_options_defaults():
    options = ValidateOptions()
    assert options.level == ValidationLevel.STRICT
    assert options.allow_unknown_fields is True


def test_validation_error_str():
    err = ValidationError(SpecRule.AXIS_COUNT, "found 6 axes", "multiscales[0].axes")
    assert str(err) == "Spec rule [axis-count] violated: found 6 axes"
    assert err.rule == SpecRule.AXIS_COUNT
    assert err.message == "found 6 axes"
    assert err.location == "multiscales[0].axes"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def test_validate_structural_accepts_valid(valid_metadata):
    # Default options (STRICT) and an explicit STRICT both accept the baseline.
    validate_structural(valid_metadata)
    validate_structural(valid_metadata, ValidateOptions(level=ValidationLevel.STRICT))


def test_validate_structural_fail_fast_order(valid_metadata):
    # Two independent violations: dataset-order (rule 8) and omero-color (9).
    valid_metadata.datasets[0].coordinateTransformations = [Scale([1.0, 2.0, 2.0])]
    valid_metadata.datasets[1].coordinateTransformations = [Scale([1.0, 1.0, 1.0])]
    valid_metadata.omero = _omero("xyz")
    with pytest.raises(ValidationError) as exc_info:
        validate_structural(valid_metadata)
    # Fail-fast reports the earlier-ordered rule, not the OMERO color.
    assert exc_info.value.rule == SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST


def test_validate_structural_schema_only_skips(valid_metadata):
    # A single axis is a clear axis-count violation under STRICT...
    valid_metadata.axes = [Axis(name="x", type="space")]
    with pytest.raises(ValidationError):
        validate_structural(valid_metadata)
    # ...but SCHEMA_ONLY runs no structural rule, so the same metadata passes.
    validate_structural(
        valid_metadata, ValidateOptions(level=ValidationLevel.SCHEMA_ONLY)
    )
