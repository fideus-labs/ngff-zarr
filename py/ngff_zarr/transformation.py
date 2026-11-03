# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A transformation on its own (OME-Zarr 0.6).

The spec's standalone transformation examples are documents that hold
``coordinateTransformations`` and, optionally, ``coordinateSystems``, without
an image: the shape of a scene without images. A transformation store is a
group whose ``ome`` metadata is such a document with one transformation.
Parameters that live in arrays, such as a ``displacements`` field, are nodes
below the store, written first, that the transformation references by
``path``; a rotation or affine matrix the writer stores there itself.
"""

from __future__ import annotations

import posixpath
from typing import Any

from ._matrix_transform_arrays import (
    check_matrix_destinations,
    stored_matrix_paths,
    write_matrix_arrays,
)
from ._store_types import StoreLike
from ._supported_versions import V06_ONDISK_VERSION
from .scene import (
    SCENE_VERSIONS,
    _check_image_path,
    _externalize_matrices,
    _field_paths,
    _scene_from_dict,
)
from .v06.zarr_metadata import Transform, validate_transform


def _is_transformation_document(ome: object) -> bool:
    """Whether an ``ome`` value is a transformation document and nothing else."""
    return (
        isinstance(ome, dict)
        and "coordinateTransformations" in ome
        and "multiscales" not in ome
        and "scene" not in ome
    )


def _write_transformation(
    store: StoreLike, transform: Transform, version: str, overwrite: bool
) -> None:
    """Write ``transform`` as the only transformation of a store; see ``to_ome_zarr``.

    The transformation lands in the root group's
    ``ome.coordinateTransformations``, and a rotation or affine matrix in a
    float64 array below the store, at ``coordinateTransformations/<name>``
    unless the transform names a ``path`` (see
    :mod:`~ngff_zarr._matrix_transform_arrays`). A node it references by
    ``path`` has to be in the store already, written first, and the store
    written with ``overwrite=False``, which also keeps the root group's other
    attributes.
    """
    from ._zarrista_utils import (
        consolidate_metadata,
        create_zarrista_group,
        create_zarrista_subgroup,
        normalize_store,
        read_group_attributes,
    )
    from .to_ngff_zarr import _remove_none_values

    if version not in SCENE_VERSIONS:
        raise ValueError(
            "A transformation store is defined from OME-Zarr 0.6; got version "
            f"{version!r}."
        )
    validate_transform(transform, None, version)
    store_path = normalize_store(store)
    serialized = [_remove_none_values(transform.to_dict())]
    matrix_arrays, fields = _externalize_matrices([transform], serialized)
    for path in [*fields, *matrix_arrays]:
        _check_image_path(path)
    if fields and overwrite:
        raise ValueError(
            f"The transformation references the nodes {fields}; write those "
            "first with to_ome_zarr() below the store, then the transformation "
            "with overwrite=False so they are kept."
        )
    for path in fields:
        if not (store_path.joinpath(*path.split("/")) / "zarr.json").exists():
            raise ValueError(
                f"The transformation references {path!r}, which the store does "
                "not hold; write it first with to_ome_zarr() below the store, "
                "then the transformation with overwrite=False."
            )
    if not overwrite:
        # The store keeps what it holds: a matrix may only replace an array
        # the current root already names as one.
        check_matrix_destinations(
            store_path,
            matrix_arrays,
            stored_matrix_paths(read_group_attributes(store_path, zarr_format=3)),
        )
    ondisk = V06_ONDISK_VERSION.value if version == "0.6" else version
    document: dict[str, Any] = {
        "version": ondisk,
        "coordinateTransformations": serialized,
    }
    # The arrays land before the root names them, so a failed write leaves a
    # kept store readable; an overwrite clears the store when it creates the
    # root, so there they follow it.
    if not overwrite:
        write_matrix_arrays(store_path, matrix_arrays)
    create_zarrista_group(store_path, {"ome": document}, 3, overwrite=overwrite)
    if overwrite:
        write_matrix_arrays(store_path, matrix_arrays)
    for path in fields:
        parent = posixpath.dirname(path)
        if parent:
            create_zarrista_subgroup(store_path, parent, None, 3)
    consolidate_metadata(store_path, 3)


def _read_transformation(
    root_attrs: dict[str, Any], validate: bool, version: str | None, store: Any = None
) -> Transform:
    """The transformation ``root_attrs`` declares; see ``from_ome_zarr``.

    ``store`` is the opened store the document came from; a matrix stored in
    an array below it is loaded inline.
    """
    from .parse_metadata import _detect_version

    ome = root_attrs["ome"]
    if version is None:
        version = _detect_version(root_attrs).value
    _, transforms = _scene_from_dict(ome, version, store)
    if len(transforms) != 1:
        raise ValueError(
            "A transformation store holds one coordinateTransformations entry; "
            f"got {len(transforms)}."
        )
    (transform,) = transforms
    if validate:
        for path in _field_paths([transform]):
            _check_image_path(path)
    return transform
