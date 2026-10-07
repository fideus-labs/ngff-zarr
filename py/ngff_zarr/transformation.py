# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A transformation on its own (OME-Zarr 0.6).

The spec's standalone transformation examples are documents that hold
``coordinateTransformations`` and, optionally, ``coordinateSystems``, without
an image: the shape of a scene without images. A transformation store is a
group whose ``ome`` metadata is such a document with one transformation.
Parameters that live in arrays, such as a ``displacements`` field, are nodes
below the store, written first, that the transformation references by
``path``.
"""

from __future__ import annotations

import posixpath
from typing import Any

from ._store_types import StoreLike
from ._supported_versions import V06_ONDISK_VERSION
from .scene import SCENE_VERSIONS, _check_image_path, _field_paths, _scene_from_dict
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
    ``ome.coordinateTransformations``. A node it references by ``path`` has
    to be in the store already, written first, and the store written with
    ``overwrite=False``, which also keeps the root group's other attributes.
    """
    from ._zarrista_utils import (
        consolidate_metadata,
        create_zarrista_group,
        create_zarrista_subgroup,
        normalize_store,
    )
    from .to_ngff_zarr import _remove_none_values

    if version not in SCENE_VERSIONS:
        raise ValueError(
            "A transformation store is defined from OME-Zarr 0.6; got version "
            f"{version!r}."
        )
    validate_transform(transform, None, version)
    store_path = normalize_store(store)
    fields = sorted(_field_paths([transform]))
    for path in fields:
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
    ondisk = V06_ONDISK_VERSION.value if version == "0.6" else version
    document: dict[str, Any] = {
        "version": ondisk,
        "coordinateTransformations": [_remove_none_values(transform.to_dict())],
    }
    create_zarrista_group(store_path, {"ome": document}, 3, overwrite=overwrite)
    for path in fields:
        parent = posixpath.dirname(path)
        if parent:
            create_zarrista_subgroup(store_path, parent, None, 3)
    consolidate_metadata(store_path, 3)


def _read_transformation(
    root_attrs: dict[str, Any], validate: bool, version: str | None
) -> Transform:
    """The transformation ``root_attrs`` declares; see ``from_ome_zarr``."""
    from .parse_metadata import _detect_version

    ome = root_attrs["ome"]
    if version is None:
        version = _detect_version(root_attrs).value
    _, transforms = _scene_from_dict(ome, version)
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
