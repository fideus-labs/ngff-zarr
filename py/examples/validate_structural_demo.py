# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""End-to-end demo of OME-Zarr v0.4 structural validation.

Proves :func:`ngff_zarr.structural_validation.validate_structural` works on
hand-built :class:`~ngff_zarr.v04.zarr_metadata.Metadata` objects with no
network access and no test fixtures.

It first builds a fully valid ``[c, y, x]`` two-level multiscales metadata and
confirms it passes. It then builds a deliberately broken copy for each
image-metadata rule ``validate_structural`` enforces over plain multiscales
metadata -- axis count, axis type, axis order, scale-vector length, per-level
coordinate-transform ordering, dataset ordering, and OMERO channel color --
each mutating a single field, and confirms ``validate_structural`` raises the
expected :class:`~ngff_zarr.structural_validation.SpecRule`, printing the
caught rule identifier, location, and message for each.

Out of this demo's scope: the two RFC 4 anatomical-orientation rules (which
additionally pull in the optional ``jsonschema`` dependency) and the two HCS
plate/well rules (enforced through the separate ``validate_plate`` /
``validate_well`` orchestrators).

Run from the ``py/`` directory:

    pixi run --as-is -e test python examples/validate_structural_demo.py

The process exits non-zero if the valid metadata is rejected, if any broken
copy fails to raise, or if a raised rule does not match the expected one, so
the script doubles as a self-checking smoke test.
"""

from __future__ import annotations

import sys

from ngff_zarr.structural_validation import (
    SpecRule,
    ValidationError,
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


def build_valid_metadata() -> Metadata:
    """Return a fully valid v0.4 ``[c, y, x]`` two-level multiscales metadata.

    Three axes (one channel, two space); two datasets, each with a single
    length-3 ``scale``; the second level is coarser (larger spatial scale) than
    the first, as the spec requires.
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


def build_invalid_cases() -> list[tuple[str, SpecRule, Metadata]]:
    """Build one broken copy for each image-metadata rule the demo covers.

    Each case mutates exactly one field of the valid baseline so that the named
    rule is the first (and only) one violated, given the orchestrator's
    fail-fast spec-MUST ordering. The RFC 4 orientation rules and the HCS
    plate/well rules are out of scope; see the module docstring.
    """
    cases: list[tuple[str, SpecRule, Metadata]] = []

    # Six axes: the count is outside the permitted 2..=5 range.
    metadata = build_valid_metadata()
    metadata.axes = [
        Axis(name="t", type="time"),
        Axis(name="c", type="channel"),
        Axis(name="z", type="space"),
        Axis(name="y", type="space"),
        Axis(name="x", type="space"),
        Axis(name="x", type="space"),
    ]
    cases.append(("six axes (count out of 2..=5)", SpecRule.AXIS_COUNT, metadata))

    # Two channel axes: at most one is permitted.
    metadata = build_valid_metadata()
    metadata.axes = [
        Axis(name="c", type="channel"),
        Axis(name="c", type="channel"),
        Axis(name="y", type="space"),
        Axis(name="x", type="space"),
    ]
    cases.append(("two channel axes", SpecRule.AXIS_TYPE, metadata))

    # Spatial axes in (x, y) order: names must be a suffix of (z, y, x).
    metadata = build_valid_metadata()
    metadata.axes = [
        Axis(name="c", type="channel"),
        Axis(name="x", type="space"),
        Axis(name="y", type="space"),
    ]
    cases.append(("spatial axes in x, y order", SpecRule.AXIS_ORDER, metadata))

    # A dataset scale vector of length 2 against 3 axes.
    metadata = build_valid_metadata()
    metadata.datasets[0].coordinateTransformations = [Scale([1.0, 1.0])]
    cases.append(("length-2 scale vector", SpecRule.SCALE_LENGTH_MISMATCH, metadata))

    # A translation that precedes its scale within a dataset.
    metadata = build_valid_metadata()
    metadata.datasets[1].coordinateTransformations = [
        Translation([0.0, 0.0, 0.0]),
        Scale([1.0, 2.0, 2.0]),
    ]
    cases.append(
        (
            "translation before scale",
            SpecRule.GLOBAL_COORD_TRANSFORM_AFTER_PER_LEVEL,
            metadata,
        )
    )

    # A coarser level whose spatial scale is finer than the level above it.
    metadata = build_valid_metadata()
    metadata.datasets[0].coordinateTransformations = [Scale([1.0, 2.0, 2.0])]
    metadata.datasets[1].coordinateTransformations = [Scale([1.0, 1.0, 1.0])]
    cases.append(
        (
            "coarser level with finer scale",
            SpecRule.DATASET_ORDER_HIGHEST_TO_LOWEST,
            metadata,
        )
    )

    # An OMERO channel color that is not exactly six hex digits.
    metadata = build_valid_metadata()
    metadata.omero = Omero(
        channels=[
            OmeroChannel(
                color="xyz",
                window=OmeroWindow(min=0.0, max=255.0, start=0.0, end=255.0),
            )
        ]
    )
    cases.append(
        ("omero channel color 'xyz'", SpecRule.OMERO_CHANNEL_COLOR_FORMAT, metadata)
    )

    return cases


def main() -> int:
    """Run the demo; return a process exit code (0 on full success)."""
    print("OME-Zarr v0.4 structural validation demo")
    print("=" * 64)

    valid = build_valid_metadata()
    try:
        validate_structural(valid)
    except ValidationError as exc:
        print(f"UNEXPECTED: valid metadata was rejected: {exc}")
        return 1
    print("valid [c, y, x] two-level metadata: passed validate_structural")
    print("-" * 64)

    cases = build_invalid_cases()
    caught = 0
    for label, expected_rule, metadata in cases:
        try:
            validate_structural(metadata)
        except ValidationError as exc:
            if exc.rule != expected_rule:
                print(
                    f"UNEXPECTED rule for {label!r}: got [{exc.rule.value}], "
                    f"expected [{expected_rule.value}]"
                )
                return 1
            caught += 1
            print(f"caught [{exc.rule.value}] at {exc.location}")
            print(f"    {label}: {exc.message}")
        else:
            print(f"UNEXPECTED: {label!r} did not raise ValidationError")
            return 1

    print("-" * 64)
    total = len(cases)
    print(f"caught {caught}/{total} expected violations")
    return 0 if caught == total else 1


if __name__ == "__main__":
    sys.exit(main())
