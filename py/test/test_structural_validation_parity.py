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
from pathlib import Path

import pytest
from ngff_zarr import (
    SUPPORTED_VERSIONS,
    SpecRule,
    ValidateOptions,
    ValidationError,
    ValidationLevel,
    structural_validation,
    validate_collection,
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
# so the two are directly comparable. The first thirteen entries are the
# image/multiscales rules dispatched by validate_structural -- the first eleven
# are the v0.4 rules and the next two (zarr-format, ome-namespace) are the v0.5
# namespacing rules, inert for v0.4; the next two are the HCS plate/well rules
# dispatched by validate_plate / validate_well; the final seven are the RFC-8
# node/collection rules dispatched by validate_collection.
CANONICAL_SPEC_RULE_IDS = [
    "axis-count",
    "axis-type",
    "axis-order",
    "axis-names-unique",
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
    "node-type-required",
    "node-name-required",
    "node-name-unique",
    "node-id-format",
    "node-id-unique",
    "node-nodes-xor-path",
    "path-type-known",
]


# The locked RFC-3 version manifest: the versions whose axis model is
# unrestricted, so that validate_axis_count / validate_axis_type /
# validate_axis_order / validate_spatial_axis_order are inert for them. This
# identical literal list appears in the Deno mirror test. axis-names-unique is
# deliberately absent: RFC-3 adds that rule rather than lifting it, so it is
# never inert (see docs/validation/rule-reference.md).
CANONICAL_RFC3_VERSIONS = [
    "0.9.dev1",
    "0.9.dev3",
]

# Every other supported version, plus the no-version default, must enforce the
# axis rules. Read off SUPPORTED_VERSIONS so a newly supported version has to
# be classified here rather than silently defaulting to "restricted".
NON_RFC3_VERSIONS = [
    version.value
    for version in SUPPORTED_VERSIONS
    if version.value not in CANONICAL_RFC3_VERSIONS
] + [None]

# The locked RFC-4 orientation version manifest: the versions at which the
# three axis-orientation rules are normative (ome/ngff-spec#190 folds RFC-4
# into 0.9.dev1). This identical literal list appears in the Deno mirror test.
# The gate points the opposite way from CANONICAL_RFC3_VERSIONS: RFC-3 *lifts*
# the axis restrictions at 0.9.dev1 while RFC-4 *adds* the orientation
# requirements, so the rules are inert below these versions. No version at all
# keeps them on, as a strictness choice (like axis-names-unique).
CANONICAL_RFC4_VERSIONS = [
    "0.9.dev1",
    "0.9.dev3",
]

# Every other supported version must leave the orientation rules inert. Read
# off SUPPORTED_VERSIONS so a newly supported version has to be classified
# here rather than silently defaulting to either side.
PRE_RFC4_VERSIONS = [
    version.value
    for version in SUPPORTED_VERSIONS
    if version.value not in CANONICAL_RFC4_VERSIONS
]

# The locked RFC-8 version manifest: the versions that store the node model
# under ``ome``, at which the seven node/collection rules are normative. This
# identical literal list appears in the Deno mirror test. Like the RFC-4
# rules, the collection rules stay on when no version is declared.
CANONICAL_RFC8_VERSIONS = [
    "0.9.dev3",
]

# Every other supported version must leave the collection rules inert. Read
# off SUPPORTED_VERSIONS so a newly supported version has to be classified
# here rather than silently defaulting to either side.
PRE_RFC8_VERSIONS = [
    version.value
    for version in SUPPORTED_VERSIONS
    if version.value not in CANONICAL_RFC8_VERSIONS
]

# The canonical fail-fast evaluation order of the image/multiscales
# orchestrator (validate_structural). Each entry is the SpecRule the
# orchestrator must raise when that rule -- and every rule after it -- is
# violated at once. Two SpecRule values appear twice because two rule functions
# share each: validate_axis_order and validate_spatial_axis_order both surface
# AXIS_ORDER (positions 3-4), and validate_per_dataset_scale_count and
# validate_transform_order both surface GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL
# (positions 6 and 8). The two HCS rules are absent: they are not part of the
# image/multiscales orchestrator. The two v0.5 namespacing rules (zarr-format,
# ome-namespace) run last in the orchestrator but are inert for the v0.4
# metadata exercised here, so they never appear in the observed order.
EXPECTED_EVALUATION_ORDER = [
    SpecRule.AXIS_COUNT,
    SpecRule.AXIS_TYPE,
    SpecRule.AXIS_ORDER,  # validate_axis_order (class ordering)
    SpecRule.AXIS_ORDER,  # validate_spatial_axis_order (spatial suffix)
    SpecRule.AXIS_NAMES_UNIQUE,
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


def _axes_with_repeated_name() -> list[Axis]:
    """Return valid ``[c, z, y, x]`` axes with the channel axis renamed ``z``.

    Every rule before :attr:`SpecRule.AXIS_NAMES_UNIQUE` still passes: the
    class order is channel-then-space, the spatial names remain the ``(z, y,
    x)`` suffix, and the count is 4. Only the repeated ``z`` is left for the
    orchestrator to catch, which pins the rule's position in the cascade.
    """
    axes = _valid_axes_with_inconsistent_orientation()
    axes[0].name = "z"
    return axes


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
    # channels), axis-order (a space precedes a channel), spatial-order (names
    # are not the (z, y, x) suffix) and axis-names-unique (a repeated "z");
    # axis-count is evaluated first.
    metadata = Metadata(
        axes=[
            Axis(name="x", type="space"),
            Axis(name="c", type="channel"),
            Axis(name="c2", type="channel"),
            Axis(name="t", type="time"),
            Axis(name="z", type="space"),
            Axis(name="z", type="space"),
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

    # Stage 2 -> AXIS_TYPE. Count fixed (5 axes); two channels remain, a space
    # still precedes a channel, the spatial names are still wrong and "z" is
    # still repeated.
    metadata.axes = [
        Axis(name="x", type="space"),
        Axis(name="c", type="channel"),
        Axis(name="c2", type="channel"),
        Axis(name="z", type="space"),
        Axis(name="z", type="space"),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 3 -> AXIS_ORDER (class ordering). One channel now, but a space axis
    # still precedes it; the spatial names are still not the (z, y, x) suffix
    # and "z" is still repeated.
    metadata.axes = [
        Axis(name="x", type="space"),
        Axis(name="c", type="channel"),
        Axis(name="z", type="space"),
        Axis(name="z", type="space"),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 4 -> AXIS_ORDER (spatial suffix). Class order fixed (channel first),
    # but spatial names (x, z, z) are not the length-3 suffix of (z, y, x).
    metadata.axes = [
        Axis(name="c", type="channel"),
        Axis(name="x", type="space"),
        Axis(name="z", type="space"),
        Axis(name="z", type="space"),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 5 -> AXIS_NAMES_UNIQUE. Every axis rule before it now passes, but
    # the channel axis is named "z" like the first space axis.
    metadata.axes = _axes_with_repeated_name()
    observed.append(_first_violated_rule(metadata))

    # Stage 6 -> per-dataset scale count. Axes are now fully valid [c, z, y, x]
    # (with an inconsistent-orientation violation lurking as rule 10). Dataset
    # 0 still has two scales, so the per-dataset-scale-count rule fires.
    metadata.axes = _valid_axes_with_inconsistent_orientation()
    observed.append(_first_violated_rule(metadata))

    # Stage 7 -> SCALE_LENGTH_MISMATCH. Dataset 0 now has exactly one scale, but
    # its length is 3 against 4 axes; a scale-after-translation (rule 7) lurks.
    metadata.datasets[0].coordinateTransformations = [
        Translation([0.0, 0.0, 0.0]),
        Scale([1.0, 1.0, 1.0]),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 8 -> transform order. Lengths fixed to 4; dataset 0's scale still
    # follows a translation. Dataset 1 is made coarser-but-smaller so the
    # dataset-order rule (8) lurks behind the transform-order violation.
    metadata.datasets[0].coordinateTransformations = [
        Translation([0.0, 0.0, 0.0, 0.0]),
        Scale([1.0, 1.0, 1.0, 1.0]),
    ]
    metadata.datasets[1].coordinateTransformations = [Scale([1.0, 0.5, 0.5, 0.5])]
    observed.append(_first_violated_rule(metadata))

    # Stage 9 -> dataset order. Transform order fixed (scale before
    # translation); dataset 1 is still coarser-but-smaller than dataset 0.
    metadata.datasets[0].coordinateTransformations = [
        Scale([1.0, 1.0, 1.0, 1.0]),
        Translation([0.0, 0.0, 0.0, 0.0]),
    ]
    observed.append(_first_violated_rule(metadata))

    # Stage 10 -> OMERO color. Dataset order fixed (level 1 coarser-larger);
    # only the bad OMERO color and the orientation violation remain.
    metadata.datasets[1].coordinateTransformations = [Scale([1.0, 2.0, 2.0, 2.0])]
    observed.append(_first_violated_rule(metadata))

    # Stage 11 -> orientation. OMERO color fixed; the y axis still declares a
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


# ---------------------------------------------------------------------------
# Manifest: the locked RFC-3 version set
# ---------------------------------------------------------------------------


def _rfc3_axis_metadata() -> Metadata:
    """Six same-type axes: legal under RFC-3, illegal at every other version.

    Violates axis-count (6 > 5) and the spatial-axis rules (6 > 3 ``space``
    axes) at once, and nothing else, so the orchestrator accepts it exactly
    when the axis rules are inert.
    """
    names = ["a", "b", "c", "d", "e", "f"]
    return Metadata(
        axes=[Axis(name=name, type="space") for name in names],
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[
                    Scale([1.0] * len(names)),
                    Translation([0.0] * len(names)),
                ],
            )
        ],
        coordinateTransformations=None,
    )


def test_rfc3_version_manifest_is_locked():
    # Inert at exactly the manifest versions...
    for version in CANONICAL_RFC3_VERSIONS:
        validate_structural(_rfc3_axis_metadata(), version=version)

    # ...and enforced at every other supported version, and by default.
    for version in NON_RFC3_VERSIONS:
        with pytest.raises(ValidationError) as exc_info:
            validate_structural(_rfc3_axis_metadata(), version=version)
        assert exc_info.value.rule == SpecRule.AXIS_COUNT, version


def _repeated_name_metadata() -> Metadata:
    """A repeated axis name that no *other* axis rule can catch.

    ``(time "x", space "y", space "x")`` satisfies all four restricted axis
    rules: 3 axes, one ``time`` and two ``space``, ordered time then space, and
    the spatial names are the ``(y, x)`` suffix. ``axis-names-unique`` is
    therefore the only rule that can fire, at every version.
    """
    return Metadata(
        axes=[
            Axis(name="x", type="time"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[
                    Scale([1.0, 1.0, 1.0]),
                    Translation([0.0, 0.0, 0.0]),
                ],
            )
        ],
        coordinateTransformations=None,
    )


def test_axis_names_unique_is_never_inert():
    # RFC-3 *adds* this rule rather than lifting one, so unlike the other four
    # axis rules it fires at the RFC-3 versions too -- and at every other.
    for version in CANONICAL_RFC3_VERSIONS + NON_RFC3_VERSIONS:
        with pytest.raises(ValidationError) as exc_info:
            validate_structural(_repeated_name_metadata(), version=version)
        assert exc_info.value.rule == SpecRule.AXIS_NAMES_UNIQUE, version


# ---------------------------------------------------------------------------
# Manifest: the locked RFC-4 orientation version set
# ---------------------------------------------------------------------------


def _orientation_metadata(axes: list[Axis]) -> Metadata:
    """Wrap ``axes`` in metadata that satisfies every non-orientation rule.

    The axis lists below are legal under both the restricted axis model and
    RFC-3, so at every version the orientation rules alone decide the verdict.
    """
    return Metadata(
        axes=axes,
        datasets=[
            Dataset(
                path="0",
                coordinateTransformations=[
                    Scale([1.0] * len(axes)),
                    Translation([0.0] * len(axes)),
                ],
            )
        ],
        coordinateTransformations=None,
    )


def _orientation_on_non_space_axes() -> list[Axis]:
    """Orientation on the non-spatial time axis, and nowhere else.

    Only a stray orientation violates RFC 4 here, and it sits on the one axis
    that may not carry it -- exercising the non-space arm the orchestrator
    reaches even when no spatial axis is oriented.
    """
    return [
        Axis(
            name="t",
            type="time",
            orientation=_orientation("anatomical", "inferior-to-superior"),
        ),
        Axis(name="y", type="space"),
        Axis(name="x", type="space"),
    ]


def _duplicate_anatomical_axis_axes() -> list[Axis]:
    """Two spatial axes on the one left-right anatomical axis."""
    return [
        Axis(
            name="y",
            type="space",
            orientation=_orientation("anatomical", "left-to-right"),
        ),
        Axis(
            name="x",
            type="space",
            orientation=_orientation("anatomical", "right-to-left"),
        ),
    ]


# One violating document per orientation rule. Each satisfies every other rule
# at every version, so the orientation rule alone decides accept or reject.
_RFC4_ORIENTATION_CASES = [
    (
        _valid_axes_with_inconsistent_orientation,
        SpecRule.AXIS_ORIENTATION_ANATOMICAL_TYPE,
    ),
    (_orientation_on_non_space_axes, SpecRule.AXIS_ORIENTATION_ON_NON_SPACE),
    (_duplicate_anatomical_axis_axes, SpecRule.AXIS_ORIENTATION_UNIQUE_AXIS),
]


def test_rfc4_orientation_version_manifest_is_locked():
    for build_axes, rule in _RFC4_ORIENTATION_CASES:
        # Enforced at the manifest versions, and when no version is given...
        for version in [*CANONICAL_RFC4_VERSIONS, None]:
            with pytest.raises(ValidationError) as exc_info:
                validate_structural(
                    _orientation_metadata(build_axes()), version=version
                )
            assert exc_info.value.rule == rule, (rule, version)
        # ...and inert at every earlier supported version.
        for version in PRE_RFC4_VERSIONS:
            validate_structural(_orientation_metadata(build_axes()), version=version)


# ---------------------------------------------------------------------------
# Manifest: the locked RFC-8 collection version set
# ---------------------------------------------------------------------------


def _invalid_collection_document() -> dict:
    """A collection whose sibling nodes share a name, and nothing else wrong.

    Trips :attr:`SpecRule.NODE_NAME_UNIQUE` exactly when the RFC-8 rules run,
    so the orchestrator accepts it exactly when they are inert.
    """
    return {
        "type": "collection",
        "name": "c",
        "nodes": [
            {"type": "multiscale", "name": "img"},
            {"type": "multiscale", "name": "img"},
        ],
    }


def test_rfc8_collection_version_manifest_is_locked():
    # Enforced at the manifest versions, and when no version is given...
    for version in [*CANONICAL_RFC8_VERSIONS, None]:
        with pytest.raises(ValidationError) as exc_info:
            validate_collection(_invalid_collection_document(), version=version)
        assert exc_info.value.rule == SpecRule.NODE_NAME_UNIQUE, version
    # ...and inert at every other supported version, 0.9.dev1 included:
    # RFC-8 lands at 0.9.dev3.
    for version in PRE_RFC8_VERSIONS:
        validate_collection(_invalid_collection_document(), version=version)


def _commented_rule_ids() -> list[str]:
    """The rule ids listed in the canonical table at the top of the module.

    The table is a comment block, so nothing but this test keeps it in step
    with :class:`SpecRule`; it lost ``axis-names-unique`` once already.
    """
    source = Path(structural_validation.__file__).read_text()
    table = source.split("# Canonical rule table", 1)[1].split("\nfrom ", 1)[0]
    return re.findall(r"^#   ([a-z0-9-]+)$", table, re.M)


def _documented_rule_ids() -> list[str]:
    """The rule ids listed in the rule-reference table."""
    reference = (
        Path(__file__).parents[2] / "docs" / "validation" / "rule-reference.md"
    ).read_text()
    return re.findall(r"^\| *\d+ *\| *`([a-z0-9-]+)`", reference, re.M)


def test_module_rule_table_matches_the_manifest():
    assert _commented_rule_ids() == CANONICAL_SPEC_RULE_IDS


def test_rule_reference_doc_matches_the_manifest():
    assert _documented_rule_ids() == CANONICAL_SPEC_RULE_IDS
