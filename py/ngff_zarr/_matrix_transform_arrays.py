# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Rotation and affine matrices stored as Zarr arrays.

RFC-5 lets a ``rotation`` or ``affine`` carry its matrix inline in the JSON
metadata or in a 2D Zarr array the transform names by ``path``, relative to the
multiscales group; the schema accepts one or the other, never both. The writer
always takes the array form. A JSON number is decimal text, kept only to the
precision of every tool that parses and re-serializes the document, while a
float64 array holds each entry bit for bit.

The reader loads the array back into the inline field and keeps ``path``, so
the in-memory model always carries the values and a rewrite reuses the array.
"""

import copy
import posixpath
import re
from collections.abc import Iterable, Iterator

import numpy as np

#: The group generated matrix arrays are written under, as in the
#: specification's own example.
MATRIX_ARRAY_GROUP = "coordinateTransformations"

#: The transformation types parameterized by a matrix. Each holds its matrix
#: inline under the key that is also its type.
_MATRIX_TYPES = frozenset({"rotation", "affine"})

_NODE_NAME_CHARACTERS = re.compile(r"[A-Za-z0-9._-]+")


def _iter_matrix_transforms(transforms: Iterable | None) -> Iterator[dict]:
    """Every rotation and affine in ``transforms``, depth first in document order.

    Descends into the wrapper transforms: ``sequence`` and ``byDimension``
    items, and a ``bijection``'s ``forward`` then ``inverse``.
    """
    for transform in transforms or ():
        if not isinstance(transform, dict):
            continue
        kind = transform.get("type")
        if kind in _MATRIX_TYPES:
            yield transform
        elif kind == "sequence":
            yield from _iter_matrix_transforms(transform.get("transformations"))
        elif kind == "byDimension":
            yield from _iter_matrix_transforms(
                item.get("transformation")
                for item in transform.get("transformations") or ()
                if isinstance(item, dict)
            )
        elif kind == "bijection":
            yield from _iter_matrix_transforms(
                (transform.get("forward"), transform.get("inverse"))
            )


def _describe(transform: dict) -> str:
    name = transform.get("name")
    return f"{transform['type']} transformation" + (f" '{name}'" if name else "")


def _is_node_name(name) -> bool:
    """Whether ``name`` is a portable Zarr node name.

    The Zarr v3 node name rules with the recommended character set: no ``/``,
    not only periods, and no reserved ``__`` prefix.
    """
    return (
        isinstance(name, str)
        and _NODE_NAME_CHARACTERS.fullmatch(name) is not None
        and name.strip(".") != ""
        and not name.startswith("__")
    )


def _checked_path(path: str, transform: dict) -> str:
    """``path``, refused when it could resolve outside the multiscales group."""
    if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(
            f"{_describe(transform)} names the array path {path!r}; a matrix "
            "array sits at a relative path inside the multiscales group, with "
            "no empty, '.' or '..' segments"
        )
    return path


def _is_claimed(path: str, claimed: set[str]) -> bool:
    """Whether ``path`` is taken, or is an ancestor or descendant of a taken one."""
    return any(
        other == path or other.startswith(f"{path}/") or path.startswith(f"{other}/")
        for other in claimed
    )


def _generated_path(transform: dict, claimed: set[str]) -> str:
    name = transform.get("name")
    base = name if _is_node_name(name) else transform["type"]
    path = f"{MATRIX_ARRAY_GROUP}/{base}"
    suffix = 0
    while _is_claimed(path, claimed):
        suffix += 1
        path = f"{MATRIX_ARRAY_GROUP}/{base}_{suffix}"
    return path


def _matrix(values, transform: dict) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{_describe(transform)} does not hold a rectangular matrix of "
            f"numbers: {error}"
        ) from error
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(
            f"{_describe(transform)} must hold a non-empty 2D matrix; got shape "
            f"{matrix.shape}"
        )
    return matrix


def named_matrix_paths(transforms: list | None) -> set[str]:
    """The array paths the rotations and affines in ``transforms`` already name."""
    return {
        transform["path"]
        for transform in _iter_matrix_transforms(transforms)
        if transform.get("path")
    }


def externalize_matrix_transforms(transforms: list | None) -> dict[str, np.ndarray]:
    """Move each rotation and affine matrix in ``transforms`` into an array.

    ``transforms`` is a serialized ``coordinateTransformations`` list and is
    edited in place: a transform holding its matrix inline loses that field
    and gains ``path``. Returns the matrices to write, as float64 arrays keyed
    by their path relative to the multiscales group.

    A transform keeps a ``path`` it already names, as one read back from a
    store does, so a rewrite updates that array. Otherwise the path is
    ``coordinateTransformations/<name>``, with the transform type standing in
    for a name that is absent or not a portable Zarr node name, and a numeric
    suffix added when an earlier transform claimed the path. A transform that
    names a ``path`` but holds no matrix refers to an array already in the
    store and is left as it is.
    """
    matrices = list(_iter_matrix_transforms(transforms))
    claimed = {
        _checked_path(transform["path"], transform)
        for transform in matrices
        if transform.get("path")
    }
    arrays: dict[str, np.ndarray] = {}
    for transform in matrices:
        kind = transform["type"]
        values = transform.pop(kind, None)
        if values is None or len(values) == 0:
            if not transform.get("path"):
                raise ValueError(
                    f"{_describe(transform)} holds no matrix and names no path "
                    "to an array holding one"
                )
            continue
        matrix = _matrix(values, transform)
        path = transform.get("path")
        if not path:
            path = _generated_path(transform, claimed)
            claimed.add(path)
            transform["path"] = path
        if path in arrays and not np.array_equal(arrays[path], matrix):
            raise ValueError(
                f"Two matrix transformations name the array path {path!r} but "
                "hold different matrices"
            )
        arrays[path] = matrix
    return arrays


def write_matrix_arrays(store, arrays: dict[str, np.ndarray]) -> None:
    """Write each matrix as a single-chunk, uncompressed float64 Zarr v3 array.

    ``arrays`` is keyed by path relative to ``store``, the multiscales group;
    missing parent groups are created. An existing array at a path is replaced.
    """
    from ._zarrista_utils import (
        _native_contiguous,
        create_zarrista_array,
        create_zarrista_subgroup,
    )

    for path, matrix in arrays.items():
        parent = posixpath.dirname(path)
        if parent:
            create_zarrista_subgroup(store, parent, None, 3)
        array = create_zarrista_array(
            store, path, matrix.shape, np.float64, matrix.shape, 3, compressors=[]
        )
        array[tuple(slice(0, extent) for extent in matrix.shape)] = _native_contiguous(
            matrix
        )


def resolve_matrix_transforms(
    transforms: list, store, subpath: str | None = None
) -> list:
    """``transforms`` with each array-stored rotation and affine matrix inline.

    Returns a deep copy of the serialized ``coordinateTransformations`` list in
    which every rotation and affine that names a ``path`` and holds no inline
    matrix carries the array's values, as nested lists of float, beside that
    ``path``. ``path`` resolves against the multiscales group at ``subpath``
    within ``store``, and is refused when it could reach outside that group.
    """
    from zarrista.exceptions import ZarristaError

    from ._zarrista_utils import open_lazy_array

    resolved = copy.deepcopy(transforms)
    for transform in _iter_matrix_transforms(resolved):
        kind = transform["type"]
        path = transform.get("path")
        if transform.get(kind) is not None or not isinstance(path, str):
            continue
        # Read only inside the group: a crafted path could otherwise reach
        # past the store, or past ``subpath``, which an absolute path discards.
        _checked_path(path, transform)
        location = posixpath.join(subpath, path) if subpath else path
        try:
            values = open_lazy_array(store, location).compute()
        except (OSError, KeyError, ValueError, ZarristaError) as error:
            raise ValueError(
                f"{_describe(transform)} stores its matrix in the array at "
                f"'{path}', which could not be read: {error}"
            ) from error
        transform[kind] = _matrix(values, transform).tolist()
    return resolved
