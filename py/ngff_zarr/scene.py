# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Scenes: images that share a spatial relationship (OME-Zarr 0.6).

A scene is the Zarr group above a set of multiscale images. Its ``ome.scene``
metadata declares the coordinate transformations between the images'
coordinate systems and, optionally, coordinate systems of its own, such as a
common world system the images map into. Transformation parameters stored as
arrays, such as displacement fields, live in a ``coordinateTransformations``
subgroup beside the images. The layout is specified at
https://ngff.openmicroscopy.org/0.6/#scene-md.
"""

from __future__ import annotations

import copy
import os
import posixpath
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote

from ._store_types import StoreLike
from ._supported_versions import V06_ONDISK_VERSION, NgffVersion
from .multiscales import NgffMultiscales
from .v06.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Transform,
)
from .v06.zarr_metadata import Metadata as Metadata_v06

#: The versions whose metadata model defines scenes.
SCENE_VERSIONS = (NgffVersion.V06, NgffVersion.V09dev1)


@dataclass
class NgffScene:
    """Images that share a spatial relationship, and the transformations between them.

    ``images`` maps each image's path below the scene group to its
    multiscales. ``coordinateTransformations`` relate the images' coordinate
    systems to each other and to ``coordinateSystems``, the systems the scene
    declares itself. Each end of a transformation is a
    :class:`~ngff_zarr.CoordinateSystemIdentifier`: with a ``path`` it names a
    coordinate system of that image, without one a system of the scene. The
    first scene coordinate system is the reference a viewer displays by
    default.
    """

    images: dict[str, NgffMultiscales]
    coordinateTransformations: list[Transform]
    coordinateSystems: list[CoordinateSystem] | None = None

    def to_ome_zarr(self, store: StoreLike, **kwargs: Any) -> None:
        """Write this scene and its images to an OME-Zarr store.

        Convenience method that delegates to :func:`ngff_zarr.to_ome_zarr`.
        See that function for full parameter documentation.
        """
        from .to_ngff_zarr import to_ome_zarr

        to_ome_zarr(store, self, **kwargs)

    @classmethod
    def from_ome_zarr(cls, store: StoreLike, **kwargs: Any) -> NgffScene:
        """Read an OME-Zarr scene store into an NgffScene.

        Convenience classmethod that delegates to :func:`ngff_zarr.from_ome_zarr`
        with ``kind="scene"``, which refuses a store that holds no scene.
        """
        from .from_ngff_zarr import from_ome_zarr

        return from_ome_zarr(store, **{**kwargs, "kind": "scene"})


def _check_image_path(path: object) -> None:
    """Reject an image path that is not a plain path below the scene group.

    The path is joined to the store, so a backslash counts as a separator
    (Windows does) and a percent-encoded part is read as a URL store does.
    """
    parts = re.split(r"[\\/]", path) if isinstance(path, str) else [""]
    if any(unquote(part) in ("", ".", "..") for part in parts):
        raise ValueError(
            f"Image path {path!r} must be a relative path below the scene group."
        )


def _image_systems(multiscales: NgffMultiscales) -> list[CoordinateSystem]:
    """The coordinate systems the image carries once written at 0.6."""
    return list(multiscales.metadata.to_version("0.6").coordinateSystems)


def _field_paths(transforms: list[Transform]) -> set[str]:
    """The paths of the arrays and field groups the transformations reference."""
    paths: set[str] = set()
    for transform in transforms:
        path = getattr(transform, "path", None)
        if isinstance(path, str):
            paths.add(path)
        paths |= _field_paths(
            [
                item.transformation if hasattr(item, "transformation") else item
                for item in getattr(transform, "transformations", None) or ()
            ]
        )
        paths |= _field_paths(
            [
                nested
                for nested in (
                    getattr(transform, "forward", None),
                    getattr(transform, "inverse", None),
                )
                if nested is not None
            ]
        )
    return paths


def _check_scene(scene: NgffScene, version: str = "0.6") -> None:
    """Raise ``ValueError`` unless the scene is one OME-Zarr ``version`` can express.

    Every image path stays below the scene group; the scene's coordinate
    systems carry unique names and an axis model the version allows; every
    transformation names its ends, they resolve, to a system the scene
    declares or to one of the image at its path, and the transformation holds
    for the two systems it joins; and the coordinate systems and images form
    one connected graph, as the spec requires. An image's own systems are
    connected through its multiscales already, so each image counts as one
    node.
    """
    from .to_ngff_zarr import _AxisView, _gate_axis_views, _gate_spans
    from .v06.zarr_metadata import validate_transform

    for path in scene.images:
        _check_image_path(path)
    if not scene.coordinateTransformations:
        raise ValueError("A scene declares at least one coordinate transformation.")

    local: dict[str, CoordinateSystem] = {}
    for system in scene.coordinateSystems or []:
        if not system.name or system.name in local:
            raise ValueError(
                "Scene coordinateSystems names must be non-empty and unique; "
                f"got {[s.name for s in scene.coordinateSystems]}."
            )
        local[system.name] = system
    _gate_axis_views(
        [
            (f"scene.coordinateSystems[{index}].axes", _AxisView(list(system.axes)))
            for index, system in enumerate(scene.coordinateSystems or [])
        ],
        version,
    )

    nodes = {("scene", name) for name in local}
    nodes |= {("image", path) for path in scene.images}
    parent = dict(zip(nodes, nodes))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index, transform in enumerate(scene.coordinateTransformations):
        ends = []
        systems = []
        for side in ("input", "output"):
            where = f"coordinateTransformations[{index}].{side}"
            reference = getattr(transform, side, None)
            if reference is None or reference.name is None:
                raise ValueError(
                    f"{where} must name a coordinate system: a "
                    "CoordinateSystemIdentifier with a name, and the path of the "
                    "image that declares it unless the scene does."
                )
            if reference.path is None:
                if reference.name not in local:
                    raise ValueError(
                        f"{where} names coordinate system {reference.name!r}, "
                        "which the scene does not declare. Declare it in "
                        "coordinateSystems, or give the reference the path of "
                        "the image that declares it."
                    )
                ends.append(("scene", reference.name))
                systems.append(local[reference.name])
                continue
            image = scene.images.get(reference.path)
            if image is None:
                raise ValueError(
                    f"{where} references image {reference.path!r}, which the "
                    f"scene does not contain; its images are "
                    f"{sorted(scene.images)}."
                )
            declared = {system.name: system for system in _image_systems(image)}
            if reference.name not in declared:
                raise ValueError(
                    f"{where} names coordinate system {reference.name!r} of "
                    f"image {reference.path!r}, which declares "
                    f"{sorted(declared)}."
                )
            ends.append(("image", reference.path))
            systems.append(declared[reference.name])

        where = f"coordinateTransformations[{index}]"
        _gate_spans(transform, where, {}, {len(system.axes) for system in systems})
        # The reader's own checks, against the two systems the transformation
        # joins. They resolve a reference by name, so the probe names its
        # ends apart when both refer to systems that share a name.
        names = [transform.input.name, transform.output.name]
        if names[0] == names[1] and systems[0] is not systems[1]:
            names = ["input", "output"]
        probe = copy.copy(transform)
        probe.input = CoordinateSystemIdentifier(name=names[0])
        probe.output = CoordinateSystemIdentifier(name=names[1])
        try:
            validate_transform(
                probe,
                [
                    CoordinateSystem(name=names[0], axes=systems[0].axes),
                    CoordinateSystem(name=names[1], axes=systems[1].axes),
                ],
                version,
            )
        except ValueError as invalid:
            raise ValueError(
                f"{where} ({transform.type}) would be written as a transform "
                f"this package cannot read back: {invalid}"
            ) from invalid
        parent[find(ends[0])] = find(ends[1])

    groups: dict[tuple, list[str]] = {}
    for kind, name in nodes:
        label = f"image {name!r}" if kind == "image" else f"coordinate system {name!r}"
        groups.setdefault(find((kind, name)), []).append(label)
    if len(groups) > 1:
        raise ValueError(
            "The scene's coordinate systems and images must form one connected "
            f"graph of transformations, but they fall into {len(groups)} "
            f"unconnected groups: {sorted(sorted(group) for group in groups.values())}."
        )


def _scene_to_dict(scene: NgffScene) -> dict[str, Any]:
    """Serialize the scene's metadata to its ``ome.scene`` object."""
    from .to_ngff_zarr import _remove_none_values

    document: dict[str, Any] = {}
    if scene.coordinateSystems is not None:
        document["coordinateSystems"] = [
            _remove_none_values(asdict(system)) for system in scene.coordinateSystems
        ]
    document["coordinateTransformations"] = [
        _remove_none_values(transform.to_dict())
        for transform in scene.coordinateTransformations
    ]
    return document


def _scene_from_dict(
    document: dict[str, Any], version: str | None
) -> tuple[list[CoordinateSystem] | None, list[Transform]]:
    """Parse an ``ome.scene`` object into its coordinate systems and transformations."""
    systems = None
    if document.get("coordinateSystems") is not None:
        systems = [
            CoordinateSystem(
                name=entry["name"], axes=[Axis(**axis) for axis in entry["axes"]]
            )
            for entry in document["coordinateSystems"]
        ]
    raw = document.get("coordinateTransformations") or []
    transforms = Metadata_v06._parse_transforms(raw, systems or [], version)
    # The multiscales parser keeps a reference only when its name resolves in
    # the systems it is given. A scene's references mostly name systems of its
    # images, so they are restored as written.
    for transform, entry in zip(transforms, raw):
        for side in ("input", "output"):
            reference = entry.get(side)
            if isinstance(reference, dict):
                setattr(
                    transform,
                    side,
                    CoordinateSystemIdentifier(
                        path=reference.get("path"), name=reference.get("name")
                    ),
                )
    return systems, transforms


def _child_store(store: StoreLike, path: str) -> str:
    """The store of the image at ``path`` below the scene group."""
    if not isinstance(store, (str, os.PathLike)):
        raise TypeError(
            "A scene is read from a local directory path, a remote URL or an "
            f".ozx path; got {type(store).__name__}."
        )
    return f"{os.fspath(store).rstrip('/')}/{path}"


def _write_scene(
    store: StoreLike,
    scene: NgffScene,
    version: str,
    overwrite: bool,
    consolidate_metadata: bool,
    **kwargs: Any,
) -> None:
    """Write a scene and its images to a directory store; see ``to_ome_zarr``.

    The scene metadata lands in the root group's ``ome.scene`` and each image
    is written below its path with :func:`~ngff_zarr.to_ome_zarr`, at
    ``version``, with ``kwargs``. The scene is checked against the spec
    first (see :func:`_check_scene`); nothing is written when it fails. A
    transformation that references an array or a field group by ``path``,
    such as a ``displacements`` field, needs that node written first, below
    the scene's store, and the scene written with ``overwrite=False``, which
    also keeps the root group's other attributes.
    """
    from ._zarrista_utils import consolidate_metadata as _consolidate_metadata
    from ._zarrista_utils import (
        create_zarrista_group,
        create_zarrista_subgroup,
        normalize_store,
    )
    from .to_ngff_zarr import to_ome_zarr

    if version not in SCENE_VERSIONS:
        raise ValueError(
            f"Scene metadata is defined from OME-Zarr 0.6; got version {version!r}."
        )
    _check_scene(scene, version)
    store_path = normalize_store(store)
    # An array-backed transformation points at a node of the store, which
    # the scene writer does not produce: it has to be there already, so it is
    # written first and the scene keeps it.
    fields = sorted(_field_paths(scene.coordinateTransformations))
    for path in fields:
        _check_image_path(path)
    if fields and overwrite:
        raise ValueError(
            f"The scene's transformations reference the nodes {fields}; write "
            "those first with to_ome_zarr() below the scene's store, then the "
            "scene with overwrite=False so they are kept."
        )
    for path in fields:
        if not (store_path.joinpath(*path.split("/")) / "zarr.json").exists():
            raise ValueError(
                f"The scene's transformations reference {path!r}, which the "
                "store does not hold; write it first with to_ome_zarr() below "
                "the scene's store, then the scene with overwrite=False."
            )
    ondisk = V06_ONDISK_VERSION.value if version == "0.6" else version
    root_attrs = {"ome": {"version": ondisk, "scene": _scene_to_dict(scene)}}
    create_zarrista_group(store_path, root_attrs, 3, overwrite=overwrite)
    for path in fields:
        parent = posixpath.dirname(path)
        if parent:
            create_zarrista_subgroup(store_path, parent, None, 3)
    for path, multiscales in scene.images.items():
        parent = posixpath.dirname(path)
        if parent:
            create_zarrista_subgroup(store_path, parent, None, 3)
        to_ome_zarr(
            store_path.joinpath(*path.split("/")),
            multiscales,
            version=version,
            overwrite=True,
            consolidate_metadata=False,
            **kwargs,
        )
    if consolidate_metadata:
        _consolidate_metadata(store_path, 3)


def _read_scene(
    store: StoreLike,
    root_attrs: dict[str, Any],
    validate: bool,
    version: str | None,
    storage_options: dict | None,
) -> NgffScene:
    """Read the scene ``root_attrs`` declares, with the images it references.

    ``store`` is what the caller passed to ``from_ome_zarr``: a local directory
    path, a remote URL or an ``.ozx`` path, below which each referenced image is
    read with :func:`~ngff_zarr.from_ome_zarr`. ``validate`` checks the root
    against the bundled ``scene`` schema, validates each image, and runs the
    spec checks of :func:`_check_scene` on the result.
    """
    from .from_ngff_zarr import from_ome_zarr
    from .parse_metadata import _detect_version

    document = root_attrs["ome"]["scene"]
    if not isinstance(document, dict):
        raise ValueError(f"The 'ome.scene' entry must be an object; got {document!r}.")
    if version is None:
        version = _detect_version(root_attrs).value
    if validate:
        from .validate import validate as validate_ngff

        validate_ngff(root_attrs, version=version, model="scene")

    systems, transforms = _scene_from_dict(document, version)
    images: dict[str, NgffMultiscales] = {}
    for transform in transforms:
        for reference in (transform.input, transform.output):
            path = getattr(reference, "path", None)
            if path is not None and path not in images:
                # The path comes from the store's own metadata and is joined
                # to the store, so it must not reach outside the scene group.
                _check_image_path(path)
                image = from_ome_zarr(
                    _child_store(store, path),
                    validate=validate,
                    storage_options=storage_options,
                )
                if not isinstance(image, NgffMultiscales):
                    raise ValueError(
                        f"The scene references image {path!r}, but that node "
                        "holds a scene rather than a multiscales image."
                    )
                images[path] = image
    scene = NgffScene(
        images=images,
        coordinateTransformations=transforms,
        coordinateSystems=systems,
    )
    if validate:
        _check_scene(scene, version)
    return scene
