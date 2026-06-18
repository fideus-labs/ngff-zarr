# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Structural validation for OME-Zarr v0.4 image/multiscales metadata.

This module implements the OME-Zarr v0.4 specification MUSTs that a JSON
Schema cannot express, layered conceptually on top of the schema validation
performed by :mod:`ngff_zarr.validate` (which checks the raw attribute dict).
Where schema validation answers "is this shaped like OME-Zarr?", the rules
here answer "does this obey the spec's structural invariants?" -- e.g. axis
counts, axis ordering, coordinate-transformation arity, finest-to-coarsest
dataset ordering, and OMERO channel color format.

The rules are pure Python (standard library only) and operate on the already
parsed :class:`~ngff_zarr.v04.zarr_metadata.Metadata` dataclass, so they work
even when the optional ``[validate]`` extra (``jsonschema``) is not installed.

Each rule is identified by a stable, kebab-case :class:`SpecRule` value that
is part of the observable surface (reused verbatim in logs, tests, and the
TypeScript port). Rules fail fast: the orchestrator
(:func:`validate_structural`) raises a :class:`ValidationError` on the first
violation, in canonical spec-MUST order.
"""

# Canonical rule table (maintainers: keep in sync with SpecRule and the
# orchestrator's call order). Location strings use dotted-segment,
# JSON-Pointer-style intent, identifying the offending metadata node.
#
#   axis-count
#       axis count within 2..=5
#       e.g. multiscales[0].axes
#   axis-type
#       <=1 time axis and <=1 channel axis
#       e.g. multiscales[0].axes
#   axis-order
#       time before channel before space; space names a suffix of (z, y, x)
#       e.g. multiscales[0].axes[1]
#   scale-length-mismatch
#       every scale/translation length == axis count (global + per-dataset)
#       e.g. multiscales[0].datasets[2].coordinateTransformations[0]
#   global-coord-transform-after-per-level
#       exactly one scale per dataset; a translation must follow its scale
#       e.g. multiscales[0].datasets[1].coordinateTransformations
#   dataset-order-highest-to-lowest
#       datasets ordered finest->coarsest (spatial scale must not decrease)
#       e.g. multiscales[0].datasets[2]
#   omero-channel-color-format
#       each OMERO channel color is exactly 6 hex digits
#       e.g. multiscales[0].omero.channels[0].color
#   axis-orientation-consistent-type
#       all spatial-axis RFC 4 orientations share one type
#       e.g. multiscales[0].axes
#   axis-orientation-completeness
#       if any spatial axis has orientation, all spatial axes do
#       e.g. multiscales[0].axes
#   zarr-format
#       a v0.5 entry implies a Zarr v3 store (a leaked zarr_format must be 3)
#       e.g. multiscales[0]
#   ome-namespace
#       a v0.5 entry must not retain a group-level ome/multiscales wrapper key
#       e.g. multiscales[0]
#   plate-row-index-consistency
#       each well's rowIndex/columnIndex matches the row/column named in its path
#       e.g. plate.wells[3]
#   well-acquisition-missing
#       each well image references an acquisition when the plate declares many
#       e.g. well.images[0].acquisition

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .v04.zarr_metadata import Axis, Dataset, Metadata, Plate, Transform, Well


class SpecRule(str, Enum):
    """Stable, kebab-case identifiers for the structural spec rules.

    Inheriting from :class:`str` makes the kebab-case identifier the member's
    value, so it can be compared and logged directly
    (``SpecRule.AXIS_COUNT == "axis-count"``) for stable assertions. Members
    are listed in canonical spec-MUST evaluation order.
    """

    AXIS_COUNT = "axis-count"
    AXIS_TYPE = "axis-type"
    AXIS_ORDER = "axis-order"
    SCALE_LENGTH_MISMATCH = "scale-length-mismatch"
    GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL = "global-coord-transform-after-per-level"
    DATASET_ORDER_HIGHEST_TO_LOWEST = "dataset-order-highest-to-lowest"
    OMERO_CHANNEL_COLOR_FORMAT = "omero-channel-color-format"
    AXIS_ORIENTATION_CONSISTENT_TYPE = "axis-orientation-consistent-type"
    AXIS_ORIENTATION_COMPLETENESS = "axis-orientation-completeness"
    ZARR_FORMAT = "zarr-format"
    OME_NAMESPACE = "ome-namespace"
    PLATE_ROW_INDEX_CONSISTENCY = "plate-row-index-consistency"
    WELL_ACQUISITION_MISSING = "well-acquisition-missing"


class ValidationLevel(str, Enum):
    """How much validation :func:`validate_structural` performs.

    ``STRICT`` is the default and runs every structural image/multiscales rule
    in this module. ``SCHEMA_ONLY`` skips them (schema validation is handled
    separately by :mod:`ngff_zarr.validate` on the raw attribute dict). Deep,
    store-reading checks (e.g. confirming each dataset array exists with the
    expected shape) are intentionally out of scope for both levels.
    """

    SCHEMA_ONLY = "schema_only"
    STRICT = "strict"


@dataclass
class ValidateOptions:
    """Options controlling structural validation.

    Attributes
    ----------
    level:
        Validation level to run. Defaults to :attr:`ValidationLevel.STRICT`.
    allow_unknown_fields:
        Whether unknown metadata fields are tolerated. Defaults to ``True`` to
        mirror the parser, which warns on (rather than rejects) unknown fields.
    """

    level: ValidationLevel = ValidationLevel.STRICT
    allow_unknown_fields: bool = True


class ValidationError(Exception):
    """Raised when a structural spec rule is violated.

    Carries the offending :class:`SpecRule`, a human-readable ``message``, and
    an optional ``location`` identifying the offending metadata node in
    dotted-segment (JSON-Pointer-style) form, e.g.
    ``multiscales[0].datasets[2].coordinateTransformations[0]``.
    """

    def __init__(
        self, rule: SpecRule, message: str, location: str | None = None
    ) -> None:
        self.rule = rule
        self.message = message
        self.location = location
        super().__init__(message)

    def __str__(self) -> str:
        return f"Spec rule [{self.rule.value}] violated: {self.message}"


def _transform_len_mismatch(
    transform: Transform, axes_len: int
) -> tuple[str, int] | None:
    """Check one coordinate transform's vector length against the axis count.

    Centralizes the length invariant shared by the global and per-dataset
    coordinate-transformation checks (see
    :attr:`SpecRule.SCALE_LENGTH_MISMATCH`), so the rule lives in one place.

    Returns ``("scale", actual_len)`` or ``("translation", actual_len)`` when
    the transform's vector length disagrees with ``axes_len``; otherwise
    returns ``None``. Transforms that carry no length-bearing vector (e.g.
    ``identity``) are ignored and return ``None``.
    """
    if transform.type == "scale":
        actual_len = len(transform.scale)
        if actual_len != axes_len:
            return ("scale", actual_len)
    elif transform.type == "translation":
        actual_len = len(transform.translation)
        if actual_len != axes_len:
            return ("translation", actual_len)
    return None


def _first_scale_vector(dataset: Dataset) -> list[float] | None:
    """Return the first ``scale`` vector in a dataset, or ``None`` if absent.

    Used by :func:`validate_dataset_order` to compare adjacent multiscale
    levels without re-deriving the per-dataset ``scale`` lookup.
    """
    for transform in dataset.coordinateTransformations:
        if transform.type == "scale":
            return transform.scale
    return None


# Class-ordering rank for axis types: a ``time`` axis must precede a
# ``channel`` axis, which must precede any ``space`` axis. Lower rank must
# never follow higher rank (see ``validate_axis_order``).
_AXIS_TYPE_RANK = {"time": 0, "channel": 1, "space": 2}

# Spatial axis names in canonical order. A valid set of ``space`` axis names
# is the matching-length suffix of this tuple: 1 -> ("x",), 2 -> ("y", "x"),
# 3 -> ("z", "y", "x") (see ``validate_spatial_axis_order``).
_SPATIAL_AXIS_NAMES = ("z", "y", "x")


def validate_axis_count(metadata: Metadata) -> None:
    """Validate that the axis count is within the v0.4-permitted range.

    OME-Zarr v0.4 requires between 2 and 5 axes, inclusive.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.AXIS_COUNT` when ``len(metadata.axes)`` lies
        outside ``2..=5``; location ``multiscales[0].axes``.
    """
    count = len(metadata.axes)
    if not (2 <= count <= 5):
        raise ValidationError(
            SpecRule.AXIS_COUNT,
            f"OME-Zarr v0.4 requires between 2 and 5 axes, inclusive; found {count}.",
            "multiscales[0].axes",
        )


def validate_axis_type(metadata: Metadata) -> None:
    """Validate axis-type multiplicity.

    At most one ``time`` axis and at most one ``channel`` axis may be present.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.AXIS_TYPE` when more than one ``time`` axis or
        more than one ``channel`` axis is present; location
        ``multiscales[0].axes``.
    """
    time_count = sum(1 for ax in metadata.axes if ax.type == "time")
    if time_count > 1:
        raise ValidationError(
            SpecRule.AXIS_TYPE,
            f"At most one 'time' axis is permitted; found {time_count}.",
            "multiscales[0].axes",
        )
    channel_count = sum(1 for ax in metadata.axes if ax.type == "channel")
    if channel_count > 1:
        raise ValidationError(
            SpecRule.AXIS_TYPE,
            f"At most one 'channel' axis is permitted; found {channel_count}.",
            "multiscales[0].axes",
        )


def validate_axis_order(metadata: Metadata) -> None:
    """Validate the class ordering of axes.

    Axes are ranked by type (``time`` < ``channel`` < ``space``) and must be
    listed in non-decreasing rank order: every ``time`` axis precedes every
    ``channel`` axis, which precedes every ``space`` axis. The first adjacent
    pair that inverts this ranking is reported.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.AXIS_ORDER` for the first adjacent pair where a
        lower-ranked axis type follows a higher-ranked one; location
        ``multiscales[0].axes[i+1]``.
    """
    axes = metadata.axes
    for i in range(len(axes) - 1):
        current = axes[i]
        following = axes[i + 1]
        current_rank = _AXIS_TYPE_RANK.get(current.type)
        following_rank = _AXIS_TYPE_RANK.get(following.type)
        if current_rank is None or following_rank is None:
            continue
        if following_rank < current_rank:
            raise ValidationError(
                SpecRule.AXIS_ORDER,
                f"Axis '{following.name}' of type '{following.type}' must not "
                f"follow axis '{current.name}' of type '{current.type}'; axes "
                f"must be ordered time, then channel, then space.",
                f"multiscales[0].axes[{i + 1}]",
            )


def validate_spatial_axis_order(metadata: Metadata) -> None:
    """Validate the count and names of spatial axes.

    The ``space`` axes, taken in order, must be the matching-length suffix of
    ``(z, y, x)``: one spatial axis must be ``(x,)``, two must be
    ``(y, x)``, and three must be ``(z, y, x)``. At most three ``space`` axes
    are permitted.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.AXIS_ORDER` when there are more than three
        ``space`` axes, or when their names are not the expected suffix of
        ``(z, y, x)``.
    """
    space_indices = [i for i, ax in enumerate(metadata.axes) if ax.type == "space"]
    count = len(space_indices)
    if count > 3:
        raise ValidationError(
            SpecRule.AXIS_ORDER,
            f"OME-Zarr v0.4 permits at most 3 'space' axes; found {count}.",
            "multiscales[0].axes",
        )
    expected = _SPATIAL_AXIS_NAMES[len(_SPATIAL_AXIS_NAMES) - count :]
    actual = tuple(metadata.axes[i].name for i in space_indices)
    if actual != expected:
        mismatch = next((j for j in range(count) if actual[j] != expected[j]), 0)
        raise ValidationError(
            SpecRule.AXIS_ORDER,
            f"Spatial axis names {list(actual)} must be the length-{count} "
            f"suffix of (z, y, x): {list(expected)}.",
            f"multiscales[0].axes[{space_indices[mismatch]}]",
        )


def validate_per_dataset_scale_count(metadata: Metadata) -> None:
    """Validate that each dataset defines exactly one ``scale`` transform.

    Every dataset's ``coordinateTransformations`` must contain exactly one
    ``scale`` (the per-level voxel size). This shares the per-dataset
    coordinate-transform-shape rule identifier
    (:attr:`SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL`); there is no
    dedicated identifier for the scale count.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL` when a
        dataset has zero or more than one ``scale`` transform; location
        ``multiscales[0].datasets[i].coordinateTransformations``.
    """
    for i, dataset in enumerate(metadata.datasets):
        scale_count = sum(
            1 for t in dataset.coordinateTransformations if t.type == "scale"
        )
        if scale_count != 1:
            raise ValidationError(
                SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,
                f"Each dataset must define exactly one 'scale' coordinate "
                f"transformation; dataset {i} has {scale_count}.",
                f"multiscales[0].datasets[{i}].coordinateTransformations",
            )


def validate_scale_length(metadata: Metadata) -> None:
    """Validate coordinate-transform vector lengths against the axis count.

    Every ``scale`` and ``translation`` vector -- in the global
    ``coordinateTransformations`` (if present) and in each dataset -- must have
    exactly one entry per axis. The first mismatch, scanned global-then-
    per-dataset, is reported. The length invariant itself lives in
    :func:`_transform_len_mismatch`.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.SCALE_LENGTH_MISMATCH` for the first transform
        whose vector length disagrees with ``len(metadata.axes)``; location
        identifies the offending transform.
    """
    axes_len = len(metadata.axes)
    if metadata.coordinateTransformations:
        for j, transform in enumerate(metadata.coordinateTransformations):
            mismatch = _transform_len_mismatch(transform, axes_len)
            if mismatch is not None:
                kind, actual_len = mismatch
                raise ValidationError(
                    SpecRule.SCALE_LENGTH_MISMATCH,
                    f"Global '{kind}' transform has length {actual_len}, but "
                    f"there are {axes_len} axes; lengths must match.",
                    f"multiscales[0].coordinateTransformations[{j}]",
                )
    for i, dataset in enumerate(metadata.datasets):
        for j, transform in enumerate(dataset.coordinateTransformations):
            mismatch = _transform_len_mismatch(transform, axes_len)
            if mismatch is not None:
                kind, actual_len = mismatch
                raise ValidationError(
                    SpecRule.SCALE_LENGTH_MISMATCH,
                    f"Dataset {i} '{kind}' transform has length {actual_len}, "
                    f"but there are {axes_len} axes; lengths must match.",
                    f"multiscales[0].datasets[{i}].coordinateTransformations[{j}]",
                )


def validate_transform_order(metadata: Metadata) -> None:
    """Validate coordinate-transform ordering within each dataset.

    Within a dataset's ``coordinateTransformations``, a ``translation`` (if
    present) must follow its ``scale`` -- a ``scale`` must never appear after a
    ``translation``.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL` for the
        first dataset where a ``scale`` follows a ``translation``; location
        identifies the offending ``scale``.
    """
    for i, dataset in enumerate(metadata.datasets):
        seen_translation = False
        for j, transform in enumerate(dataset.coordinateTransformations):
            if transform.type == "translation":
                seen_translation = True
            elif transform.type == "scale" and seen_translation:
                raise ValidationError(
                    SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,
                    f"In dataset {i}, a 'scale' transform must not follow a "
                    f"'translation'; a translation must follow its scale.",
                    f"multiscales[0].datasets[{i}].coordinateTransformations[{j}]",
                )


def validate_dataset_order(metadata: Metadata) -> None:
    """Validate that datasets are ordered finest to coarsest.

    Multiscale datasets must be listed from highest resolution (finest, i.e.
    smallest spatial ``scale``) to lowest (coarsest). For each adjacent pair,
    the later level's ``scale`` must not be smaller than the earlier level's on
    any ``space`` axis. Pairs whose ``scale`` vectors are missing or too short
    to index a spatial axis are skipped -- those shapes are the concern of
    :func:`validate_per_dataset_scale_count` and :func:`validate_scale_length`
    -- so this rule never raises ``IndexError``.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST` for the first
        adjacent pair whose later (coarser) level has a smaller spatial scale;
        location ``multiscales[0].datasets[i+1]``.
    """
    space_indices = [i for i, ax in enumerate(metadata.axes) if ax.type == "space"]
    datasets = metadata.datasets
    for i in range(len(datasets) - 1):
        finer = _first_scale_vector(datasets[i])
        coarser = _first_scale_vector(datasets[i + 1])
        if finer is None or coarser is None:
            continue
        for axis_index in space_indices:
            if axis_index >= len(finer) or axis_index >= len(coarser):
                # Scale too short to compare this axis: another rule's
                # concern. Skip so we never raise IndexError here.
                continue
            if coarser[axis_index] < finer[axis_index]:
                axis_name = metadata.axes[axis_index].name
                raise ValidationError(
                    SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST,
                    f"Datasets must be ordered finest to coarsest; dataset "
                    f"{i + 1} has spatial scale {coarser[axis_index]} on axis "
                    f"'{axis_name}', smaller than dataset {i}'s "
                    f"{finer[axis_index]}.",
                    f"multiscales[0].datasets[{i + 1}]",
                )


def validate_omero_color_hex(metadata: Metadata) -> None:
    """Validate the color format of each OMERO channel.

    When OMERO rendering metadata is present, every channel ``color`` must be
    exactly six hexadecimal digits (RGB). This reuses the existing
    :meth:`~ngff_zarr.v04.zarr_metadata.OmeroChannel.validate_color` predicate
    rather than re-implementing the format check.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.OMERO_CHANNEL_COLOR_FORMAT` for the first channel
        whose ``color`` is not six hex digits; location
        ``multiscales[0].omero.channels[i].color``.
    """
    if metadata.omero is None:
        return
    for i, channel in enumerate(metadata.omero.channels):
        try:
            channel.validate_color()
        except ValueError as exc:
            raise ValidationError(
                SpecRule.OMERO_CHANNEL_COLOR_FORMAT,
                str(exc),
                f"multiscales[0].omero.channels[{i}].color",
            ) from exc


def _axis_to_validation_dict(axis: Axis) -> dict[str, Any]:
    """Render a parsed :class:`Axis` back to the dict form RFC 4 consumes.

    Emits only the fields the RFC 4 helpers inspect -- ``name``, ``type``, and
    (for a spatial axis that declares it) ``orientation`` as a
    ``{"type", "value"}`` dict. A stored ``orientation`` may be a raw dict (the
    image read path builds axes via ``Axis(**axis_dict)``, so no coercion
    happens) or an :class:`~ngff_zarr.rfc4.AnatomicalOrientation` dataclass; an
    enum ``value`` is reduced to its string, mirroring the package's
    ``hasattr(..., "value")`` convention.
    """
    axis_dict: dict[str, Any] = {"name": axis.name, "type": axis.type}
    orientation = axis.orientation
    if orientation is not None:
        if isinstance(orientation, dict):
            orientation_type = orientation.get("type")
            orientation_value = orientation.get("value")
        else:
            orientation_type = getattr(orientation, "type", None)
            orientation_value = getattr(orientation, "value", None)
        if hasattr(orientation_value, "value"):
            orientation_value = orientation_value.value
        axis_dict["orientation"] = {
            "type": orientation_type,
            "value": orientation_value,
        }
    return axis_dict


def validate_axis_orientation(metadata: Metadata) -> None:
    """Validate RFC 4 anatomical-orientation metadata on the spatial axes.

    This rule does not reimplement RFC 4; it wraps the package's existing logic
    -- :func:`~ngff_zarr.rfc4_validation.has_rfc4_orientation_metadata` and
    :func:`~ngff_zarr.rfc4_validation.validate_rfc4_orientation` -- and surfaces
    its failures through the unified :class:`ValidationError` channel. The parsed
    :class:`~ngff_zarr.v04.zarr_metadata.Axis` objects are first rendered back to
    the axis-dict form those helpers expect (see :func:`_axis_to_validation_dict`).

    Orientation is optional in RFC 4, so when no spatial axis carries it the rule
    is a no-op and the comparatively heavy ``jsonschema`` import inside
    :func:`validate_rfc4_orientation` is never triggered.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.AXIS_ORIENTATION_CONSISTENT_TYPE` when the spatial
        axes' orientations do not all share one ``type``, or
        :attr:`SpecRule.AXIS_ORIENTATION_COMPLETENESS` when orientation is defined
        for some but not all spatial axes; location ``multiscales[0].axes``. The
        original RFC 4 message text is preserved.
    jsonschema.exceptions.ValidationError
        Propagated unchanged when an orientation value is outside the RFC 4
        vocabulary -- a schema-level concern with no dedicated structural rule.
    """
    from .rfc4_validation import (
        has_rfc4_orientation_metadata,
        validate_rfc4_orientation,
    )

    axes_dicts = [_axis_to_validation_dict(axis) for axis in metadata.axes]
    if not has_rfc4_orientation_metadata(axes_dicts):
        return
    try:
        validate_rfc4_orientation(axes_dicts)
    except ValueError as exc:
        # validate_rfc4_orientation raises exactly two ValueErrors: a spatial
        # orientation "same type" mismatch and the all-or-none completeness
        # failure. jsonschema's ValidationError (raised for an out-of-vocabulary
        # orientation value) is not a ValueError, so it -- like ImportError when
        # jsonschema is absent -- propagates untouched; no structural rule covers
        # those schema-level concerns.
        #
        # The "same type" / "all spatial axes" substrings below are a
        # load-bearing contract with validate_rfc4_orientation's message text:
        # an unrecognized ValueError falls through to the bare ``raise`` and is
        # surfaced as a plain ValueError rather than a ValidationError. The
        # message-stability test test_rfc4_orientation_messages_carry_mapping_markers
        # pins both markers so editing that wording fails CI loudly instead of
        # silently breaking this mapping.
        message = str(exc)
        if "same type" in message:
            raise ValidationError(
                SpecRule.AXIS_ORIENTATION_CONSISTENT_TYPE,
                message,
                "multiscales[0].axes",
            ) from exc
        if "all spatial axes" in message:
            raise ValidationError(
                SpecRule.AXIS_ORIENTATION_COMPLETENESS,
                message,
                "multiscales[0].axes",
            ) from exc
        raise


def validate_zarr_format_for_version(metadata: Metadata) -> None:
    """Validate that a v0.5 multiscales entry implies a Zarr v3 store.

    OME-Zarr v0.5 metadata MUST be backed by a Zarr v3 store
    (``zarr_format == 3``). A v0.4 dataset is always Zarr v2 and carries no
    ``zarr_format``, so this rule is inert below v0.5. The parsed
    :class:`~ngff_zarr.v04.zarr_metadata.Metadata` does not surface the group's
    ``zarr_format`` byte -- its authoritative on-disk verification is a
    store-backed concern -- but a ``zarr_format`` mis-embedded *inside* the
    multiscale entry (captured in :attr:`~ngff_zarr.v04.zarr_metadata.Metadata.extra`)
    is reachable here: it belongs on the enclosing Zarr v3 group, never on the
    entry, and any value other than ``3`` is incompatible with v0.5.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.ZARR_FORMAT` when v0.5 metadata carries a
        ``zarr_format`` other than ``3`` in its ``extra`` passthrough; location
        ``multiscales[0]``.
    """
    if metadata.version != "0.5":
        return
    zarr_format = metadata.extra.get("zarr_format")
    if zarr_format is not None and zarr_format != 3:
        raise ValidationError(
            SpecRule.ZARR_FORMAT,
            f"OME-Zarr v0.5 requires a Zarr v3 store (zarr_format == 3), but "
            f"the entry declares zarr_format = {zarr_format}.",
            "multiscales[0]",
        )


def validate_ome_namespace(metadata: Metadata) -> None:
    """Validate that a v0.5 multiscales entry is not double-namespaced.

    At v0.5 the multiscales metadata lives under the top-level ``ome`` namespace
    key of the group's ``zarr.json`` attributes, with the spec ``version``
    hoisted to ``ome.version``. The reader unwraps that namespace into this
    :class:`~ngff_zarr.v04.zarr_metadata.Metadata`, so the group-level wrapper
    keys -- ``ome`` itself and the ``multiscales`` array -- must never reappear
    *inside* a multiscale entry. A surviving wrapper key (captured in
    :attr:`~ngff_zarr.v04.zarr_metadata.Metadata.extra`) means the document is
    double-namespaced, was not unwrapped, or is otherwise malformed with respect
    to the v0.5 ``ome`` namespacing. A v0.4 store keeps multiscales at the
    ``.zattrs`` top level with no ``ome`` wrapper, so this rule is inert below
    v0.5.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.OME_NAMESPACE` when v0.5 metadata retains a
        group-level ``ome`` or ``multiscales`` key in its ``extra`` passthrough;
        location ``multiscales[0]``.
    """
    if metadata.version != "0.5":
        return
    for wrapper_key in ("ome", "multiscales"):
        if wrapper_key in metadata.extra:
            raise ValidationError(
                SpecRule.OME_NAMESPACE,
                f"v0.5 metadata entry retains the group-level '{wrapper_key}' "
                f"key; the 'ome' namespace wraps the group attributes, not each "
                f"multiscale entry.",
                "multiscales[0]",
            )


def validate_plate_well_index_consistency(plate: Plate) -> None:
    """Validate each plate well's recorded indices against its ``path``.

    Every :class:`~ngff_zarr.v04.zarr_metadata.PlateWell` records a ``path`` of
    the form ``"<row>/<column>"`` (e.g. ``"B/03"``) alongside numeric
    ``rowIndex`` and ``columnIndex`` fields. OME-Zarr v0.4 requires these to
    agree: the row segment must name an entry in ``plate.rows`` whose position
    equals ``rowIndex``, and the column segment must name an entry in
    ``plate.columns`` whose position equals ``columnIndex``. The first
    offending well, scanned in order, is reported.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.PLATE_ROW_INDEX_CONSISTENCY` for the first well
        whose ``path`` is not ``"<row>/<column>"``, whose row/column segment is
        absent from ``plate.rows``/``plate.columns``, or whose recorded
        ``rowIndex``/``columnIndex`` disagrees with that segment's position;
        location ``plate.wells[i]``.
    """
    row_index_by_name = {row.name: i for i, row in enumerate(plate.rows)}
    column_index_by_name = {col.name: i for i, col in enumerate(plate.columns)}
    for i, well in enumerate(plate.wells):
        location = f"plate.wells[{i}]"
        parts = well.path.split("/")
        if len(parts) != 2:
            raise ValidationError(
                SpecRule.PLATE_ROW_INDEX_CONSISTENCY,
                f"Well path {well.path!r} must have the form '<row>/<column>'.",
                location,
            )
        row_name, column_name = parts
        if row_name not in row_index_by_name:
            raise ValidationError(
                SpecRule.PLATE_ROW_INDEX_CONSISTENCY,
                f"Well path {well.path!r} names row {row_name!r}, which is not "
                f"declared in plate.rows.",
                location,
            )
        if column_name not in column_index_by_name:
            raise ValidationError(
                SpecRule.PLATE_ROW_INDEX_CONSISTENCY,
                f"Well path {well.path!r} names column {column_name!r}, which is "
                f"not declared in plate.columns.",
                location,
            )
        expected_row = row_index_by_name[row_name]
        if well.rowIndex != expected_row:
            raise ValidationError(
                SpecRule.PLATE_ROW_INDEX_CONSISTENCY,
                f"Well {well.path!r} has rowIndex {well.rowIndex}, but row "
                f"{row_name!r} is at index {expected_row} in plate.rows.",
                location,
            )
        expected_column = column_index_by_name[column_name]
        if well.columnIndex != expected_column:
            raise ValidationError(
                SpecRule.PLATE_ROW_INDEX_CONSISTENCY,
                f"Well {well.path!r} has columnIndex {well.columnIndex}, but "
                f"column {column_name!r} is at index {expected_column} in "
                f"plate.columns.",
                location,
            )


def validate_well_acquisition(plate: Plate, well: Well) -> None:
    """Validate that well images reference an acquisition when required.

    When a plate declares more than one acquisition, every image in each of its
    wells must carry an ``acquisition`` reference so the field can be attributed
    to a specific acquisition. Plates with zero or one acquisition impose no
    such requirement, so this rule is a no-op for them. The first image missing
    its reference, scanned in order, is reported.

    Raises
    ------
    ValidationError
        With :attr:`SpecRule.WELL_ACQUISITION_MISSING` for the first
        :class:`~ngff_zarr.v04.zarr_metadata.WellImage` whose ``acquisition`` is
        ``None`` while the plate declares multiple acquisitions; location
        ``well.images[i].acquisition``.
    """
    acquisitions = plate.acquisitions
    if acquisitions is None or len(acquisitions) <= 1:
        return
    for i, image in enumerate(well.images):
        if image.acquisition is None:
            raise ValidationError(
                SpecRule.WELL_ACQUISITION_MISSING,
                f"Plate declares {len(acquisitions)} acquisitions, so well "
                f"image {i} must reference one via 'acquisition'.",
                f"well.images[{i}].acquisition",
            )


def validate_structural(
    metadata: Metadata, options: ValidateOptions | None = None
) -> None:
    """Run the structural image/multiscales rules in canonical spec order.

    Orchestrates the per-rule ``validate_*`` functions in this module. It is
    fail-fast: the rules run in canonical spec-MUST order and the first
    :class:`ValidationError` raised propagates to the caller -- later rules do
    not run.

    The rules run in this order:

    1. :func:`validate_axis_count`
    2. :func:`validate_axis_type`
    3. :func:`validate_axis_order`
    4. :func:`validate_spatial_axis_order`
    5. :func:`validate_per_dataset_scale_count`
    6. :func:`validate_scale_length`
    7. :func:`validate_transform_order`
    8. :func:`validate_dataset_order`
    9. :func:`validate_omero_color_hex`
    10. :func:`validate_axis_orientation`
    11. :func:`validate_zarr_format_for_version`
    12. :func:`validate_ome_namespace`

    Rules 11 and 12 are the OME-Zarr v0.5 namespacing checks; they fire only for
    v0.5 metadata and are inert (a no-op) for v0.4.

    Parameters
    ----------
    metadata:
        The parsed OME-Zarr v0.4+ multiscales metadata to validate.
    options:
        Validation options. Defaults to :class:`ValidateOptions`, i.e.
        :attr:`ValidationLevel.STRICT`. Under
        :attr:`ValidationLevel.SCHEMA_ONLY` this function returns immediately
        without running any structural rule -- shape/schema validation is the
        separate concern of :mod:`ngff_zarr.validate`, which checks the raw
        attribute dict.

    Raises
    ------
    ValidationError
        For the first structural rule violated, carrying the offending
        :class:`SpecRule` and ``location``. Never raised under
        :attr:`ValidationLevel.SCHEMA_ONLY`, which runs no structural rule.

    Notes
    -----
    This orchestrator covers the image/multiscales rules, including the RFC 4
    anatomical-orientation checks (:func:`validate_axis_orientation`, a no-op
    when no axis declares orientation). The HCS plate/well structural rules
    operate on separate metadata objects and are dispatched by the companion
    :func:`validate_plate` and :func:`validate_well` entry points.
    """
    if options is None:
        options = ValidateOptions()
    if options.level == ValidationLevel.SCHEMA_ONLY:
        return
    validate_axis_count(metadata)
    validate_axis_type(metadata)
    validate_axis_order(metadata)
    validate_spatial_axis_order(metadata)
    validate_per_dataset_scale_count(metadata)
    validate_scale_length(metadata)
    validate_transform_order(metadata)
    validate_dataset_order(metadata)
    validate_omero_color_hex(metadata)
    validate_axis_orientation(metadata)
    validate_zarr_format_for_version(metadata)
    validate_ome_namespace(metadata)


def validate_plate(plate: Plate, options: ValidateOptions | None = None) -> None:
    """Run the structural HCS plate rules in canonical spec order.

    The plate-level counterpart to :func:`validate_structural`: it orchestrates
    the structural rules that operate on a parsed
    :class:`~ngff_zarr.v04.zarr_metadata.Plate`. Like its image/multiscales
    sibling it is fail-fast -- the first :class:`ValidationError` propagates to
    the caller and later rules do not run.

    The rules run in this order:

    1. :func:`validate_plate_well_index_consistency`

    Per-well image rules (e.g. acquisition references) are the separate concern
    of :func:`validate_well`, which is called where each well's own metadata is
    loaded.

    Parameters
    ----------
    plate:
        The parsed OME-Zarr v0.4 plate metadata to validate.
    options:
        Validation options. Defaults to :class:`ValidateOptions`, i.e.
        :attr:`ValidationLevel.STRICT`. Under
        :attr:`ValidationLevel.SCHEMA_ONLY` this function returns immediately
        without running any structural rule -- shape/schema validation is the
        separate concern of :mod:`ngff_zarr.validate`, which checks the raw
        attribute dict.

    Raises
    ------
    ValidationError
        For the first structural plate rule violated, carrying the offending
        :class:`SpecRule` and ``location``. Never raised under
        :attr:`ValidationLevel.SCHEMA_ONLY`, which runs no structural rule.
    """
    if options is None:
        options = ValidateOptions()
    if options.level == ValidationLevel.SCHEMA_ONLY:
        return
    validate_plate_well_index_consistency(plate)


def validate_well(
    plate: Plate, well: Well, options: ValidateOptions | None = None
) -> None:
    """Run the structural HCS well rules in canonical spec order.

    Validates a single :class:`~ngff_zarr.v04.zarr_metadata.Well` in the context
    of its parent :class:`~ngff_zarr.v04.zarr_metadata.Plate` -- the plate's
    acquisition declarations govern whether each well image must reference an
    acquisition. Fail-fast, like :func:`validate_structural` and
    :func:`validate_plate`.

    The rules run in this order:

    1. :func:`validate_well_acquisition`

    Parameters
    ----------
    plate:
        The parsed plate metadata that owns ``well``; supplies the acquisition
        context the well rules consult.
    well:
        The parsed well metadata to validate.
    options:
        Validation options. Defaults to :class:`ValidateOptions`, i.e.
        :attr:`ValidationLevel.STRICT`. Under
        :attr:`ValidationLevel.SCHEMA_ONLY` this function returns immediately
        without running any structural rule -- shape/schema validation is the
        separate concern of :mod:`ngff_zarr.validate`, which checks the raw
        attribute dict.

    Raises
    ------
    ValidationError
        For the first structural well rule violated, carrying the offending
        :class:`SpecRule` and ``location``. Never raised under
        :attr:`ValidationLevel.SCHEMA_ONLY`, which runs no structural rule.
    """
    if options is None:
        options = ValidateOptions()
    if options.level == ValidationLevel.SCHEMA_ONLY:
        return
    validate_well_acquisition(plate, well)
