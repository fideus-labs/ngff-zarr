# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Unit tests for the OME-Zarr v0.5 structural-validation rules.

The two v0.5 namespacing rules -- :func:`validate_zarr_format_for_version`
(``zarr-format``) and :func:`validate_ome_namespace` (``ome-namespace``) -- fire
only for v0.5 metadata and are inert for v0.4. They inspect the leaked-key
``extra`` passthrough the reader fills when a malformed or double-namespaced
document puts a group-level key inside a multiscale entry. Each rule is exercised
directly (inert-on-v0.4, accepts-clean-v0.5, rejects-malformed-v0.5) and once
end-to-end through the :func:`validate_structural` orchestrator.
"""

import pytest
from ngff_zarr.structural_validation import (
    SpecRule,
    ValidationError,
    validate_ome_namespace,
    validate_structural,
    validate_zarr_format_for_version,
)
from ngff_zarr.v04.zarr_metadata import Axis, Dataset, Metadata, Scale


def _valid_v04_metadata() -> Metadata:
    """A fully valid v0.4 ``[c, y, x]`` two-level multiscales metadata."""
    return Metadata(
        axes=[
            Axis(name="c", type="channel"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
        datasets=[
            Dataset(path="0", coordinateTransformations=[Scale([1.0, 1.0, 1.0])]),
            Dataset(path="1", coordinateTransformations=[Scale([1.0, 2.0, 2.0])]),
        ],
        coordinateTransformations=None,
    )


def _valid_v05_metadata() -> Metadata:
    """The valid baseline relabeled to v0.5 with an empty ``extra`` passthrough.

    This is the cleanly-unwrapped form a v0.5 read produces: ``version`` hoisted
    out of ``ome.version`` and no group-level wrapper key surviving in the entry.
    """
    metadata = _valid_v04_metadata()
    metadata.version = "0.5"
    return metadata


# ── zarr-format ────────────────────────────────────────────────────────────


def test_zarr_format_rule_is_inert_on_v04_even_with_stray_key():
    # Gated strictly on the v0.5 version: a v0.4 dataset is Zarr v2, so the rule
    # must not fire even if a zarr_format key is (anomalously) present in extra.
    metadata = _valid_v04_metadata()
    metadata.extra = {"zarr_format": 2}
    validate_zarr_format_for_version(metadata)


def test_zarr_format_rule_accepts_clean_v05():
    # A clean v0.5 entry carries no zarr_format in extra; nothing to contradict
    # Zarr v3, so the rule is inert.
    validate_zarr_format_for_version(_valid_v05_metadata())


def test_zarr_format_rule_accepts_v05_with_explicit_v3():
    # A zarr_format of exactly 3 (the v0.5 requirement) is accepted.
    metadata = _valid_v05_metadata()
    metadata.extra = {"zarr_format": 3}
    validate_zarr_format_for_version(metadata)


def test_zarr_format_rule_rejects_v05_non_v3():
    # A v0.5 entry that declares a non-3 zarr_format is incompatible with the
    # v0.5 => Zarr v3 requirement and must be rejected.
    metadata = _valid_v05_metadata()
    metadata.extra = {"zarr_format": 2}
    with pytest.raises(ValidationError) as exc_info:
        validate_zarr_format_for_version(metadata)
    assert exc_info.value.rule == SpecRule.ZARR_FORMAT
    assert exc_info.value.location == "multiscales[0]"


# ── ome-namespace ──────────────────────────────────────────────────────────


def test_ome_namespace_rule_is_inert_on_v04_even_with_residual_wrapper():
    # Gated strictly on the v0.5 version: a v0.4 store has no ome wrapper, so the
    # rule must not fire on v0.4 metadata regardless of extra.
    metadata = _valid_v04_metadata()
    metadata.extra = {"ome": {"version": "0.5"}}
    validate_ome_namespace(metadata)


def test_ome_namespace_rule_accepts_clean_v05():
    # A cleanly-unwrapped v0.5 entry retains no group-level wrapper key.
    validate_ome_namespace(_valid_v05_metadata())


def test_ome_namespace_rule_rejects_residual_ome_wrapper():
    # A v0.5 entry that still carries the group-level ome key is
    # double-namespaced / malformed.
    metadata = _valid_v05_metadata()
    metadata.extra = {"ome": {"version": "0.5", "multiscales": []}}
    with pytest.raises(ValidationError) as exc_info:
        validate_ome_namespace(metadata)
    assert exc_info.value.rule == SpecRule.OME_NAMESPACE
    assert exc_info.value.location == "multiscales[0]"


def test_ome_namespace_rule_rejects_residual_multiscales_wrapper():
    # The multiscales array is also a group-level key; it must not appear inside
    # a multiscale entry.
    metadata = _valid_v05_metadata()
    metadata.extra = {"multiscales": []}
    with pytest.raises(ValidationError) as exc_info:
        validate_ome_namespace(metadata)
    assert exc_info.value.rule == SpecRule.OME_NAMESPACE
    assert exc_info.value.location == "multiscales[0]"


# ── end-to-end through validate_structural ─────────────────────────────────


def test_validate_structural_accepts_clean_v05():
    # End-to-end through the orchestrator: a clean v0.5 metadata passes every
    # rule (the new v0.5 namespacing rules included).
    validate_structural(_valid_v05_metadata())


def test_validate_structural_flags_malformed_v05_namespace():
    # The active ome-namespace rule fires through the orchestrator.
    metadata = _valid_v05_metadata()
    metadata.extra = {"ome": {"version": "0.5"}}
    with pytest.raises(ValidationError) as exc_info:
        validate_structural(metadata)
    assert exc_info.value.rule == SpecRule.OME_NAMESPACE


def test_validate_structural_flags_malformed_v05_zarr_format():
    # The active zarr-format rule fires through the orchestrator.
    metadata = _valid_v05_metadata()
    metadata.extra = {"zarr_format": 2}
    with pytest.raises(ValidationError) as exc_info:
        validate_structural(metadata)
    assert exc_info.value.rule == SpecRule.ZARR_FORMAT
