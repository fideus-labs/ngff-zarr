# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""OME-Zarr ``0.9.dev1`` metadata model (RFC-3).

``0.9.dev1`` is OME-Zarr 0.6 plus RFC-3: an image may declare any number of
axes, with any names, any type strings, and in any order. The RFC-5
coordinate-system and transformation model of 0.6 is unchanged.

The dataclasses here carry no version condition. The axis restrictions live in
:mod:`ngff_zarr.structural_validation` and are applied against the *target*
version by the writer.

This module defines :class:`Axis`, :class:`CoordinateSystem` and
:class:`Metadata`; every other symbol is re-exported from
:mod:`ngff_zarr.v06.zarr_metadata`.
"""

import functools
import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Union

from .._store_types import StoreLike
from .._supported_versions import NgffVersion
from ..rfc4 import AnatomicalOrientation

# RFC-3 touches the axis model only; these symbols are unchanged from v0.6.
# They must be the same classes, not copies: ``v06.Metadata._to_v05``
# dispatches on ``isinstance(t, Scale)`` and ``isinstance(t,
# TransformSequence)``, so copies would fall through those branches and emit
# default scale/translation transforms instead.
from ..v04.zarr_metadata import AxisUnit, SupportedDims
from ..v06.zarr_metadata import (
    Affine,
    BaseTransform,
    Coordinates,
    CoordinateSystemIdentifier,
    Dataset,
    Displacements,
    Identity,
    MethodMetadata,
    Omero,
    Rotation,
    Scale,
    Transform,
    TransformSequence,
    Translation,
)
from ..v06.zarr_metadata import AxesType as AxesTypeV06

if TYPE_CHECKING:
    from ..ngff_image import NgffImage
    from ..v04.zarr_metadata import Metadata as Metadata_v04
    from ..v05.zarr_metadata import Metadata as Metadata_v05
    from ..v06.zarr_metadata import Metadata as Metadata_v06

logger = logging.getLogger(__name__)

#: RFC-3 axis name. Any non-empty string is legal; the union keeps the
#: v0.4 ``SupportedDims`` vocabulary so editors still complete the conventional
#: names. Mirrors the TypeScript port's ``AxisName``.
AxisName = Union[SupportedDims, str]

#: RFC-3 axis type. Any string is legal alongside the spec-defined vocabulary,
#: which the union keeps for the same reason. ``None`` is permitted: ``type``
#: is optional in every published axes schema (``required: ["name"]``).
AxesType = Union[AxesTypeV06, str, None]


@dataclass
class Axis:
    """An RFC-3 axis.

    Same fields as :class:`ngff_zarr.v06.zarr_metadata.Axis`, with ``name``,
    ``type`` and ``unit`` widened: each keeps the spec-defined vocabulary in a
    union with the free-form string RFC-3 allows.

    ``orientation`` (RFC-4) and ``discrete`` (RFC-5) are carried so a 0.6
    document round-trips without loss.
    """

    name: AxisName
    type: AxesType = None
    unit: AxisUnit | None = None
    orientation: AnatomicalOrientation | None = None
    discrete: bool | None = None


@dataclass
class CoordinateSystem:
    """A named set of RFC-3 axes."""

    name: str
    axes: list[Axis]


@functools.lru_cache(maxsize=1)
def _get_axis_fields() -> set[str]:
    """Valid field names of :class:`Axis`, cached."""
    return {f.name for f in fields(Axis)}


def _filter_axis_dict(axis_dict: dict) -> dict:
    """Filter an axis dict down to the recognized :class:`Axis` fields.

    Unknown keys are logged and dropped. The 0.6 axes schema permits keys this
    dataclass does not declare, such as ``longName``.

    Raises
    ------
    ValueError
        If the required ``name`` key is absent.
    """
    if "name" not in axis_dict:
        raise ValueError(
            f"Axis dictionary is missing required field 'name': {axis_dict}"
        )

    axis_fields = _get_axis_fields()
    unknown_fields = set(axis_dict.keys()) - axis_fields
    if unknown_fields:
        logger.warning(
            f"Ignoring unknown fields {unknown_fields} in axis "
            f"'{axis_dict['name']}'. These fields are not modelled by the "
            f"ngff-zarr OME-Zarr 0.9.dev1 axis dataclass."
        )
    filtered = {k: v for k, v in axis_dict.items() if k in axis_fields}
    # The v0.6 reader builds axes with a bare ``Axis(**axis)`` and its ``type``
    # has no default, so an omitted ``type`` must be made explicit here.
    filtered.setdefault("type", None)
    return filtered


def _axis_from(axis: object) -> Axis:
    """Re-instantiate any version's axis object as a v0.9.dev1 :class:`Axis`.

    ``discrete`` is read with :func:`getattr`: ``v06.Metadata._from_v05``
    assigns the v0.5 axis list straight into a ``CoordinateSystem``, so a v0.6
    ``Metadata`` reached from a 0.4/0.5 store holds v0.4 ``Axis`` instances,
    which have no ``discrete`` field.
    """
    return Axis(
        name=axis.name,
        type=axis.type,
        unit=axis.unit,
        orientation=axis.orientation,
        discrete=getattr(axis, "discrete", None),
    )


@dataclass
class Metadata:
    """OME-Zarr ``0.9.dev1`` multiscales metadata.

    Same fields as :class:`ngff_zarr.v06.zarr_metadata.Metadata`. It carries
    ``coordinateSystems``, not the flat ``axes`` list of v0.4/v0.5, so the
    v0.6 <-> 0.9.dev1 hop is a lossless structural copy; routing through v0.5
    would drop every rotation, affine, coordinates and displacements
    transform.

    There is no ``version`` field, as in v0.5 and v0.6. From v0.5 on the spec
    version lives in the group-level ``ome`` namespace, written by
    ``to_ngff_zarr._write_root_ome_attrs`` from the writer's ``version``
    argument. A field here would be emitted by ``asdict`` inside the
    ``multiscales[]`` entry, which the ``ome-namespace`` rule rejects.
    """

    coordinateSystems: list[CoordinateSystem]
    datasets: list[Dataset]
    coordinateTransformations: list[Transform] | None = None
    omero: Omero | None = None
    name: str = "image"
    type: str | None = None
    metadata: MethodMetadata | None = None
    #: Unrecognized keys captured on read (see
    #: :attr:`ngff_zarr.v04.zarr_metadata.Metadata.extra`). Carried across
    #: version conversion; a read-side validation aid, never serialized.
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        """On-the-fly validation not well covered by the JSON schemas."""
        _ = self.intrinsic_coordinate_system

    @property
    def intrinsic_coordinate_system(self) -> CoordinateSystem:
        output_cs = [ds.coordinateTransformations[0].output for ds in self.datasets]

        if output_cs[0] is None:
            raise ValueError(
                "No output coordinate system found in dataset coordinate transformations. "
            )

        if not all(output == output_cs[0] for output in output_cs):
            raise ValueError(
                "Multiple different outputs coordinate systems found in"
                f" coordinate transformations for multiscales: {output_cs}. "
                "This is out of spec for this ome-zarr 0.9.dev1."
            )

        for cs in self.coordinateSystems:
            if cs.name == output_cs[0].name:
                return cs
        raise ValueError(
            f"Dataset coordinate transformations reference coordinate system"
            f" {output_cs[0].name!r}, which is not declared in"
            f" coordinateSystems: {[cs.name for cs in self.coordinateSystems]}."
        )

    @property
    def axes(self) -> list[Axis]:
        """The intrinsic coordinate system's axes.

        A property, not a field, so ``dataclasses.asdict`` and
        ``dataclasses.fields`` ignore it and the serialized entry is unchanged.
        The axis rules in :mod:`ngff_zarr.structural_validation` read
        ``metadata.axes``.
        """
        return self.intrinsic_coordinate_system.axes

    @property
    def dimension_names(self) -> tuple:
        return tuple(ax.name for ax in self.intrinsic_coordinate_system.axes)

    def to_version(
        self, version: Union[str, NgffVersion]
    ) -> Union["Metadata", "Metadata_v04", "Metadata_v05", "Metadata_v06"]:
        if isinstance(version, str):
            version = NgffVersion(version)

        if version == NgffVersion.V09dev1:
            return self
        if version in (NgffVersion.V06, NgffVersion.V06dev4):
            return self._to_v06()
        if version == NgffVersion.V05:
            return self._to_v06()._to_v05()
        if version == NgffVersion.V04:
            return self._to_v06()._to_v05()._to_v04()
        raise ValueError(f"Unsupported version conversion: 0.9.dev1 -> {version}")

    @classmethod
    def from_version(
        cls,
        metadata: Union["Metadata", "Metadata_v04", "Metadata_v05", "Metadata_v06"],
    ) -> "Metadata":
        from ..v04.zarr_metadata import Metadata as Metadata_v04
        from ..v05.zarr_metadata import Metadata as Metadata_v05
        from ..v06.zarr_metadata import Metadata as Metadata_v06

        if isinstance(metadata, cls):
            return metadata
        if isinstance(metadata, Metadata_v06):
            return cls._from_v06(metadata)
        if isinstance(metadata, (Metadata_v04, Metadata_v05)):
            return cls._from_v06(Metadata_v06.from_version(metadata))
        raise ValueError(
            f"Unsupported metadata type ({type(metadata)}) for conversion to 0.9.dev1"
        )

    def _to_v06(self) -> "Metadata_v06":
        """Structurally map to v0.6, re-instantiating every axis.

        This does not enforce the v0.6 axis restrictions; the writer applies
        them against the target version. This converter is also used on the
        way to v0.5 and v0.4.
        """
        from ..v06.zarr_metadata import Axis as Axis_v06
        from ..v06.zarr_metadata import CoordinateSystem as CoordinateSystem_v06
        from ..v06.zarr_metadata import Metadata as Metadata_v06

        coordinate_systems = [
            CoordinateSystem_v06(
                name=cs.name,
                axes=[
                    Axis_v06(
                        name=ax.name,
                        type=ax.type,
                        unit=ax.unit,
                        orientation=ax.orientation,
                        discrete=ax.discrete,
                    )
                    for ax in cs.axes
                ],
            )
            for cs in self.coordinateSystems
        ]

        return Metadata_v06(
            coordinateSystems=coordinate_systems,
            datasets=list(self.datasets),
            coordinateTransformations=self.coordinateTransformations,
            omero=self.omero,
            name=self.name,
            type=self.type,
            metadata=self.metadata,
            extra=dict(self.extra),
        )

    @classmethod
    def _from_v06(cls, metadata_v06: "Metadata_v06") -> "Metadata":
        """Structurally map from v0.6, re-instantiating every axis.

        ``v06.Metadata._from_v05`` assigns the v0.5 axis list straight into
        ``CoordinateSystem.axes``, so a v0.6 ``Metadata`` reached from a
        0.4/0.5 store holds v0.4 ``Axis`` objects, in the same list object.
        Aliasing them would share mutable state with the source, drop fields
        under ``dataclasses.asdict`` (which walks the runtime class), and break
        equality, which dataclasses compare class-exact.
        """
        coordinate_systems = [
            CoordinateSystem(name=cs.name, axes=[_axis_from(ax) for ax in cs.axes])
            for cs in metadata_v06.coordinateSystems
        ]

        return cls(
            coordinateSystems=coordinate_systems,
            datasets=list(metadata_v06.datasets),
            coordinateTransformations=metadata_v06.coordinateTransformations,
            omero=metadata_v06.omero,
            name=metadata_v06.name,
            type=metadata_v06.type,
            metadata=metadata_v06.metadata,
            extra=dict(metadata_v06.extra),
        )

    @classmethod
    def _from_zarr_attrs(
        cls,
        root_attrs: dict,
        store: StoreLike,
        validate: bool = False,
        subpath: str | None = None,
    ) -> tuple["Metadata", list["NgffImage"]]:
        """Read a ``0.9.dev1`` store.

        Dataset transform parsing and ``NgffImage`` construction are delegated
        to the v0.6 reader. Handled here first:

        1. ``validate=True`` runs the schema pass against the bundled
           ``spec/0.9`` tree, which holds the schemas of the ``0.9.dev1``
           release. The delegate then runs with ``validate=False``: it would
           otherwise measure the document against the schemas of its own
           version.
        2. A 0.5-shaped entry (flat ``axes``, no ``coordinateSystems``) is
           normalized to a single ``intrinsic`` coordinate system, so either
           shape is readable.
        3. Unknown axis keys are stripped; the v0.6 reader builds axes with a
           bare ``Axis(**axis)``.
        """
        import copy

        from ..v06.zarr_metadata import Metadata as Metadata_v06

        if validate:
            # The 0.9.dev series records its version on the ``ome`` namespace,
            # as 0.6 does, and the bundled ``spec/0.9`` tree carries the
            # ``0.9.dev1`` tag its ``_version.schema`` binds.
            from ..validate import validate as validate_ngff

            # A document that carries no version string has none to select a
            # schema with; it goes to the 0.9.dev1 pass, which describes what
            # is wrong with it. Reaching for the version first turned such a
            # document into an error about which schemas are bundled.
            ome = root_attrs.get("ome")
            declared = ome.get("version") if isinstance(ome, dict) else None
            if not isinstance(declared, str) or not declared:
                declared = "0.9.dev1"
            validate_ngff(root_attrs, version=declared)

        if "ome" not in root_attrs or "multiscales" not in root_attrs.get("ome", {}):
            raise ValueError(
                "Invalid OME-Zarr 0.9.dev1 metadata: missing 'ome' or 'multiscales' field."
            )

        # Attributes are plain JSON, so a deep copy is cheap and keeps the
        # caller's dict untouched while the entry is normalized in place.
        patched = dict(root_attrs)
        patched["ome"] = copy.deepcopy(root_attrs["ome"])
        entry = patched["ome"]["multiscales"][0]

        if "coordinateSystems" not in entry and "axes" in entry:
            # A 0.5-shaped entry: flat axes, and dataset transforms that carry
            # no coordinate-system identifiers. The v0.6 reader dereferences
            # those identifiers, so read this shape with the v0.5 reader and
            # convert up.
            from ..v05.zarr_metadata import Metadata as Metadata_v05

            patched["ome"]["version"] = "0.5"
            metadata_v05, images = Metadata_v05._from_zarr_attrs(
                patched, store, validate=False, subpath=subpath
            )
            return cls.from_version(metadata_v05), images

        for cs in entry.get("coordinateSystems", []):
            cs["axes"] = [_filter_axis_dict(axis) for axis in cs.get("axes", [])]

        metadata_v06, images = Metadata_v06._from_zarr_attrs(
            patched, store, validate=False, subpath=subpath
        )
        return cls._from_v06(metadata_v06), images


#: The public, version-scoped API of this module: 0.9's own Axis/CoordinateSystem/Metadata,
#: plus the field transforms and shared types it reuses from v06.
__all__ = [
    "Affine",
    "Axis",
    "BaseTransform",
    "CoordinateSystem",
    "CoordinateSystemIdentifier",
    "Coordinates",
    "Dataset",
    "Displacements",
    "Identity",
    "Metadata",
    "MethodMetadata",
    "Omero",
    "Rotation",
    "Scale",
    "Transform",
    "TransformSequence",
    "Translation",
]
