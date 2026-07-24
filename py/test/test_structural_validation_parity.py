# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Parity/manifest test locking the observable structural-validation surface.

This test pins the *cross-language contract* shared by the Python and
TypeScript ports: the exact set and ordering of :class:`SpecRule` identifiers,
their lower-kebab-case format, the fail-fast evaluation order of the
image/multiscales orchestrator (:func:`validate_structural`), and the
:class:`ValidationLevel` values and default. The mirror Deno test asserts the
same facts against the identical literal identifier list, so the two suites are
line-for-line comparable and any future drift between the ports fails CI.

Adding, removing, renaming, or reordering a rule -- in either language -- must
therefore be a deliberate edit to :data:`CANONICAL_SPEC_RULE_IDS` (and its
TypeScript twin), not an accident.
"""

import re

import pytest
from ngff_zarr import (
    SpecRule,
    ValidateOptions,
    ValidationError,
    ValidationLevel,
    validate_structural,
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

# The locked rule manifest: every active OME-Zarr structural rule, in canonical
# declaration order. This identical literal list appears in the Deno mirror test
# so the two are directly comparable. The first eleven entries are the
# image/multiscales rules dispatched by validate_structural -- the first nine
# are the v0.4 rules and the next two (zarr-format, ome-namespace) are the v0.5
# namespacing rules, inert for v0.4; the final two are the HCS plate/well rules
# dispatched by validate_plate / validate_well.
CANONICAL_SPEC_RULE_IDS = [
    "axis-count",
    "axis-type",
    "axis-order",
    "scale-length-mismatch",
    "global-coord-transform-after-per-level",
    "dataset-order-highest-to-lowest",
    "omero-channel-color-format",
    "axis-orientation-anatomical-type",
    "axis-orientation-on-non-space",
    "axis-orientation-unique-axis",
    "zarr-format",
    "ome-namespace",
    "plate-row-index-consistency",
    "well-acquisition-missing",
]

# The canonical fail-fast evaluation order of the image/multiscales
# orchestrator (validate_structural). Each entry is the SpecRule the
# orchestrator must raise when that rule -- and every rule after it -- is
# violated at once. Two SpecRule values appear twice because two rule functions
# share each: validate_axis_order and validate_spatial_axis_order both surface
# AXIS_ORDER (positions 3-4), and validate_per_dataset_scale_count and
# validate_transform_order both surface GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL
# (positions 5 and 7). The two HCS rules are absent: they are not part of the
# image/multiscales orchestrator. The two v0.5 namespacing rules (zarr-format,
# ome-namespace) run last in the orchestrator but are inert for the v0.4
# metadata exercised here, so they never appear in the observed order.
EXPECTED_EVALUATION_ORDER = [
    SpecRule.AXIS_COUNT,
    SpecRule.AXIS_TYPE,
    SpecRule.AXIS_ORDER,  # validate_axis_order (class ordering)
    SpecRule.AXIS_ORDER,  # validate_spatial_axis_order (spatial suffix)
    SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,  # per-dataset scale count
    SpecRule.SCALE_LENGTH_MISMATCH,
    SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,  # transform order
    SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST,
    SpecRule.OMERO_CHANNEL_COLOR_FORMAT,
    SpecRule.AXIS_ORIENTATION_ANATOMICAL_TYPE,
]


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


def _orientation(orientation_type: str, value: str) -> dict:
    """Render an anatomical orientation as the raw dict an image read produces."""
    return {"type": orientation_type, "value": value}


def _valid_axes_with_inconsistent_orientation() -> list[Axis]:
    """Return valid ``[c, z, y, x]`` axes whose orientations mix RFC 4 types.

    Every axis rule passes (channel-then-space class order, ``(z, y, x)``
    spatial suffix, count 4), but the ``y`` axis declares orientation type
    ``"other"`` -- a type RFC 4 does not define -- tripping
    :attr:`SpecRule.AXIS_ORIENTATION_ANATOMICAL_TYPE`, the last orchestrator
    rule, when every earlier rule is satisfied.
    """
    return [
        Axis(name="c", type="channel"),
        Axis(
            name="z",
            type="space",
            orientation=_orientation("anatomical", "inferior-to-superior"),
        ),
        Axis(
            name="y",
            type="space",
            orientation=_orientation("other", "anterior-to-posterior"),
        ),
        Axis(
            name="x",
            type="space",
            orientation=_orientation("anatomical", "right-to-left"),
        ),
    ]


def _first_violated_rule(metadata: Metadata) -> SpecRule:
    """Run the orchestrator and return the single :class:`SpecRule` it raises."""
    with pytest.raises(ValidationError) as exc_info:
        validate_structural(metadata)
    return exc_info.value.rule


# ---------------------------------------------------------------------------
# Manifest: the locked SpecRule set, ordering, and format
# ---------------------------------------------------------------------------


def test_spec_rule_manifest_is_locked():
    # The exact set must equal the canonical manifest: adding, removing, or
    # renaming a rule must be a deliberate edit to CANONICAL_SPEC_RULE_IDS.
    assert {rule.value for rule in SpecRule} == set(CANONICAL_SPEC_RULE_IDS)
    # Declaration order is part of the cross-language contract, so lock it too;
    # this is the order the Deno SpecRule object must also declare.
    assert [rule.value for rule in SpecRule] == CANONICAL_SPEC_RULE_IDS
    # No two members may share a value (which would create a silent alias).
    assert len({rule.value for rule in SpecRule}) == len(CANONICAL_SPEC_RULE_IDS)


def test_spec_rule_ids_are_lower_kebab_case():
    # Each identifier is one or more lowercase ASCII words joined by hyphens.
    for rule in SpecRule:
        assert re.fullmatch(r"[a-z]+(-[a-z]+)*", rule.value), rule.value


def test_validation_level_manifest_and_default():
    # Exactly two levels, with the documented string values...
    assert {level.value for level in ValidationLevel} == {"schema_only", "strict"}
    assert ValidationLevel.SCHEMA_ONLY.value == "schema_only"
    assert ValidationLevel.STRICT.value == "strict"
    # ...and STRICT is the default applied when no options are passed.
    assert ValidateOptions().level == ValidationLevel.STRICT


# ---------------------------------------------------------------------------
# Orchestrator evaluation order (fail-fast, canonical spec-MUST order)
# ---------------------------------------------------------------------------


def test_orchestrator_evaluation_order():
    """Lock the fail-fast evaluation order of ``validate_structural``.

    Starts from metadata that violates the earliest rule *and* every later
    rule, then repairs exactly one rule at a time. Because each repair leaves
    all later violations intact, the rule raised at each step proves that rule
    is evaluated strictly before every rule after it. The observed sequence
    must equal :data:`EXPECTED_EVALUATION_ORDER`.
    """
    observed: list[SpecRule] = []
    bad_omero = _omero("xyz")  # OMERO violation (rule 9), present until repaired

    # Stage 1 -> AXIS_COUNT. Six axes simultaneously trip axis-type (two
    # channels), axis-order (a space precedes a channel), and spatial-order
    # (names are not the (z, y, x) suffix); axis-count is evaluated first.
    metadata = Metadata(
        axes=[
            Axis(name="x", type="space"),
            Axis(name="c", type="channel"),
            Axis(name="c2", type="channel"),
            Axis(name="t", type="time"),
            Axis(name="z", type="space"),
            Axis(name="w", type="space"),
        ],
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[
                    Scale([1.0, 1.0, 1.0]),
                    Scale([1.0, 1.0, 1.0]),
                ],
            ),
            Dataset(
                path="1",
                coordinateTransformations=[Scale([1.0, 2.0, 2.0, 2.0])],
            ),
        ],
        coordinateTransformations=None,
    )
    metadata.omero = bad_omero
    observed.append(_first_violated_rule(metadata))

    # Stage 2 -> AXIS_TYPE. Count fixed (5 axes); two channels remain, and a
    # space still precedes a channel and the spatial names are still wrong.
    metadata.axes = [
        Axis(name="x", type="space"),
        Axis(name="c", type="channel"),
        Axis(name="c2", type="channel"),
        Axis(name="z", type="space"),
        Axis(name="w", type="space"),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 3 -> AXIS_ORDER (class ordering). One channel now, but a space axis
    # still precedes it; the spatial names are still not the (z, y, x) suffix.
    metadata.axes = [
        Axis(name="x", type="space"),
        Axis(name="c", type="channel"),
        Axis(name="z", type="space"),
        Axis(name="w", type="space"),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 4 -> AXIS_ORDER (spatial suffix). Class order fixed (channel first),
    # but spatial names (x, z, w) are not the length-3 suffix of (z, y, x).
    metadata.axes = [
        Axis(name="c", type="channel"),
        Axis(name="x", type="space"),
        Axis(name="z", type="space"),
        Axis(name="w", type="space"),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 5 -> per-dataset scale count. Axes are now fully valid [c, z, y, x]
    # (with an inconsistent-orientation violation lurking as rule 10). Dataset
    # 0 still has two scales, so the per-dataset-scale-count rule fires.
    metadata.axes = _valid_axes_with_inconsistent_orientation()
    observed.append(_first_violated_rule(metadata))

    # Stage 6 -> SCALE_LENGTH_MISMATCH. Dataset 0 now has exactly one scale, but
    # its length is 3 against 4 axes; a scale-after-translation (rule 7) lurks.
    metadata.datasets[0].coordinateTransformations = [
        Translation([0.0, 0.0, 0.0]),
        Scale([1.0, 1.0, 1.0]),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 7 -> transform order. Lengths fixed to 4; dataset 0's scale still
    # follows a translation. Dataset 1 is made coarser-but-smaller so the
    # dataset-order rule (8) lurks behind the transform-order violation.
    metadata.datasets[0].coordinateTransformations = [
        Translation([0.0, 0.0, 0.0, 0.0]),
        Scale([1.0, 1.0, 1.0, 1.0]),
    ]
    metadata.datasets[1].coordinateTransformations = [Scale([1.0, 0.5, 0.5, 0.5])]
    observed.append(_first_violated_rule(metadata))

    # Stage 8 -> dataset order. Transform order fixed (scale before
    # translation); dataset 1 is still coarser-but-smaller than dataset 0.
    metadata.datasets[0].coordinateTransformations = [
        Scale([1.0, 1.0, 1.0, 1.0]),
        Translation([0.0, 0.0, 0.0, 0.0]),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 9 -> OMERO color. Dataset order fixed (level 1 coarser-larger);
    # only the bad OMERO color and the orientation violation remain.
    metadata.datasets[1].coordinateTransformations = [Scale([1.0, 2.0, 2.0, 2.0])]
    observed.append(_first_violated_rule(metadata))

    # Stage 10 -> orientation. OMERO color fixed; the y axis still declares a
    # different orientation type than its spatial siblings.
    metadata.omero = _omero("00FF88")
    observed.append(_first_violated_rule(metadata))

    assert observed == EXPECTED_EVALUATION_ORDER

    # Fully repaired: give the y axis a consistent orientation type and the
    # orchestrator accepts the metadata with no violation.
    metadata.axes[2] = Axis(
        name="y",
        type="space",
        orientation=_orientation("anatomical", "anterior-to-posterior"),
    )
    validate_structural(metadata)
