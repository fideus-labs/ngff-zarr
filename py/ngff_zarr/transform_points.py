# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Map points through an RFC-5 transformation.

The numbers an implementation has to agree on: where a transformation sends a
point of its input coordinate system. A conformance driver compares these
against a store's expected answers, whatever the tool that wrote the store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace

import numpy as np

from .ngff_transform_to_itk_transform import ngff_transform_to_itk_transform
from .transform_chain import flatten_sequence
from .v06.zarr_metadata import Transform

_SPATIAL_DIMS = ("x", "y", "z")


def _point_field(field):
    """``field``'s finest level in float64, its anatomical orientation set aside.

    RFC-5 places a field's samples by its scale and translation alone, so the
    orientation it carries does not move a point; ``resample`` reads a field's
    blocks the same way. Left on, it asks for the images to place the field in
    ITK physical space, which a map of intrinsic points has no use for. A
    float32 field (the compact form a registration stores) would make a float
    ITK transform, which ``itk.transform_from_dict`` cannot build.
    """
    image = field.images[0] if hasattr(field, "images") else field
    return replace(image, data=image.data.astype(np.float64), axes_orientations=None)


def transform_points(
    transform: Transform,
    points: Sequence[Sequence[float]],
    dims: Sequence[str],
    *,
    fields: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Map points of the transformation's input coordinate system to its output.

    :param transform: An RFC-5 transformation: any one
        :func:`ngff_transform_to_itk_transform` converts, or a ``sequence`` of
        them, applied in list order.
    :param points: One point per row, in the order of ``dims``.
    :param dims: The spatial axis names of the input coordinate system, in
        RFC-5 (Zarr) order.
    :param fields: The field images ``displacements`` and ``coordinates``
        transformations point at, keyed by their ``path``.
    :return: The mapped points, shape ``(n, len(dims))``.
    """
    import itk

    if any(dim not in _SPATIAL_DIMS for dim in dims):
        msg = f"transform_points maps spatial points only; got dims {tuple(dims)}"
        raise ValueError(msg)
    mapped = np.asarray(points, dtype=np.float64)
    if mapped.ndim != 2 or mapped.shape[1] != len(dims):
        msg = f"points must be rows of {len(dims)} values ({tuple(dims)}); got shape {mapped.shape}"
        raise ValueError(msg)
    fields = {path: _point_field(field) for path, field in (fields or {}).items()}
    steps = flatten_sequence(transform)
    # ITK orders a point x, y, z by name, whatever the order of dims.
    itk_axes = [list(dims).index(dim) for dim in _SPATIAL_DIMS if dim in dims]
    # Step by step: a sequence may mix linear and field stages, which no single
    # ITK-Wasm entry holds.
    for step in steps:
        (entry,) = ngff_transform_to_itk_transform(step, dims, fields=fields)
        itk_transform = itk.transform_from_dict(asdict(entry))
        moved = [
            itk_transform.TransformPoint(point.tolist())
            for point in mapped[:, itk_axes]
        ]
        mapped = mapped.copy()
        mapped[:, itk_axes] = np.asarray(moved, dtype=np.float64)
    return mapped
