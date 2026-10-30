# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-5 sequences of linear and displacement field stages, to ITK.

A registration such as a rigid alignment refined by a deformable stage is an
RFC-5 ``sequence``: an affine, then a ``displacements`` (or ``coordinates``)
field. No single affine represents it, so it converts to an ITK transform list
of one entry per group of stages.

Order
    A ``sequence`` applies its first entry first; ITK applies the last entry
    of a transform list first. The stages are reversed accordingly.

Frames
    RFC-5 stages act on the intrinsic systems, with ``phi`` mapping an image's
    intrinsic system to physical space. Every stage is converted from the
    fixed frame to the fixed frame, ``phi_f^-1 . S_k . phi_f``, so a field's
    grid has to be oriented like the fixed image, which is where registrations
    sample it. What is left, ``phi_m^-1 . phi_f``, is folded into the last
    stage when that stage is linear, and appended as its own linear stage
    otherwise: a field displaces by nothing beyond its grid, where a frame
    change folded into it would be lost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .ngff_image import NgffImage
from .v06.zarr_metadata import (
    Coordinates,
    Displacements,
    Identity,
    Transform,
    TransformSequence,
)


def _is_linear(stage: Transform) -> bool:
    return not isinstance(stage, (Displacements, Coordinates))


def flatten_sequence(transform: Transform) -> list[Transform]:
    """A sequence's stages in the order they apply, nested sequences expanded."""
    if isinstance(transform, TransformSequence):
        return [
            stage
            for nested in transform.transformations
            for stage in flatten_sequence(nested)
        ]
    return [transform]


def has_deformation(transform: Transform) -> bool:
    """Whether a ``sequence`` holds a stage no single affine can represent."""
    return not all(_is_linear(stage) for stage in flatten_sequence(transform))


def _groups(stages: list[Transform]) -> list[list[Transform]]:
    """Consecutive linear stages as one group; every field stage alone."""
    groups: list[list[Transform]] = []
    for stage in stages:
        if _is_linear(stage) and groups and _is_linear(groups[-1][0]):
            groups[-1].append(stage)
        else:
            groups.append([stage])
    return groups


def ngff_transform_chain_to_itk_transform(
    transform: TransformSequence,
    dims: Sequence[str],
    *,
    fields: Mapping[str, object] | None = None,
    fixed: NgffImage | None = None,
    moving: NgffImage | None = None,
) -> list:
    """The ITK-Wasm transform list a ``sequence`` with field stages maps to.

    What :func:`~ngff_zarr.ngff_transform_to_itk_transform` calls for such a
    sequence. Consecutive linear stages fold into one ``Affine`` entry; each
    field stage becomes its own ``DisplacementField`` entry, built from its
    image in ``fields``.
    """
    from .ngff_transform_to_itk_transform import ngff_transform_to_itk_transform

    groups = _groups(flatten_sequence(transform))
    ends_linear = _is_linear(groups[-1][0])
    framed = fixed is not None or moving is not None
    entries: list = []
    for index, group in enumerate(groups):
        last = ends_linear and index == len(groups) - 1
        frames = {"fixed": fixed, "moving": moving if last else fixed} if framed else {}
        stage = (
            group[0] if len(group) == 1 else TransformSequence(transformations=group)
        )
        # ITK applies the last entry first, so each later stage goes in front.
        entries = (
            ngff_transform_to_itk_transform(stage, dims, fields=fields, **frames)
            + entries
        )
    if not ends_linear and framed:
        entries = (
            ngff_transform_to_itk_transform(
                Identity(), dims, fixed=fixed, moving=moving
            )
            + entries
        )
    return entries
