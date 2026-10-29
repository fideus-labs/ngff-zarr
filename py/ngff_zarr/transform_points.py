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


def _point_field(field, spatial: Sequence[str], points: np.ndarray):
    """``field``'s finest level around ``points``, in float64, its orientation set aside.

    Only the nodes the points interpolate between are read, one more each side:
    a few points need not load a field of gigabytes. A point outside the field
    stays outside the crop, which is clipped to the field, so it moves as it
    would through the whole field.

    RFC-5 places a field's samples by its scale and translation alone, so the
    orientation it carries does not move a point; ``resample`` reads a field's
    blocks the same way. Left on, it asks for the images to place the field in
    ITK physical space, which a map of intrinsic points has no use for. A
    float32 field (the compact form a registration stores) would make a float
    ITK transform, which ``itk.transform_from_dict`` cannot build.
    """
    image = field.images[0] if hasattr(field, "images") else field
    index, translation = [], dict(image.translation)
    for dim, size in zip(image.dims, image.data.shape):
        if dim not in spatial:
            index.append(slice(None))
            continue
        at = (points[:, spatial.index(dim)] - image.translation[dim]) / image.scale[dim]
        start = int(np.clip(np.floor(at.min()) - 1, 0, size - 1))
        stop = int(np.clip(np.ceil(at.max()) + 2, start + 1, size))
        index.append(slice(start, stop))
        translation[dim] = image.translation[dim] + start * image.scale[dim]
    return replace(
        image,
        data=image.data[tuple(index)].astype(np.float64),
        translation=translation,
        axes_orientations=None,
    )


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
    :param points: One point per row, along the spatial axes of ``dims``, in
        their order.
    :param dims: The axis names of the input coordinate system, in RFC-5 (Zarr)
        order. Only the spatial ones carry the points: a time or channel axis
        is not mapped.
    :param fields: The field images ``displacements`` and ``coordinates``
        transformations point at, keyed by their ``path``.
    :return: The mapped points, shape ``(n, number of spatial axes)``.
    """
    import itk

    spatial = [dim for dim in dims if dim in _SPATIAL_DIMS]
    mapped = np.asarray(points, dtype=np.float64)
    if mapped.ndim != 2 or mapped.shape[1] != len(spatial):
        msg = f"points must be rows of {len(spatial)} values ({tuple(spatial)}); got shape {mapped.shape}"
        raise ValueError(msg)
    fields = dict(fields or {})
    steps = flatten_sequence(transform)
    # ITK orders a point x, y, z by name, whatever the order of dims.
    itk_axes = [spatial.index(dim) for dim in _SPATIAL_DIMS if dim in spatial]
    # Step by step: a sequence may mix linear and field stages, which no single
    # ITK-Wasm entry holds.
    for step in steps:
        path = getattr(step, "path", None)
        step_fields = (
            {path: _point_field(fields[path], spatial, mapped)}
            if path in fields
            else {}
        )
        (entry,) = ngff_transform_to_itk_transform(step, dims, fields=step_fields)
        itk_transform = itk.transform_from_dict(asdict(entry))
        moved = [
            itk_transform.TransformPoint(point.tolist())
            for point in mapped[:, itk_axes]
        ]
        mapped = mapped.copy()
        mapped[:, itk_axes] = np.asarray(moved, dtype=np.float64)
    return mapped
