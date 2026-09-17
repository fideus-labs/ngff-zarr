# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from .._store_types import StoreLike
from .._supported_versions import NgffVersion
from ..v04.zarr_metadata import Axis, Dataset, MethodMetadata, Omero, Transform

if TYPE_CHECKING:
    from ..v04.zarr_metadata import Metadata as Metadata_v04
    from ..v06.zarr_metadata import Metadata as Metadata_v06
    from ..v09.zarr_metadata import Metadata as Metadata_v09


@dataclass
class Metadata:
    axes: list[Axis]
    datasets: list[Dataset]
    coordinateTransformations: list[Transform] | None
    omero: Omero | None = None
    name: str = "image"
    type: str | None = None
    metadata: MethodMetadata | None = None
    #: Unrecognized keys captured on read (see
    #: :attr:`ngff_zarr.v04.zarr_metadata.Metadata.extra`). Carried across
    #: version conversion so a value read back through ``to_version`` retains
    #: the field; a read-side validation aid that is never serialized.
    extra: dict = field(default_factory=dict)

    def to_version(
        self, version: str | NgffVersion
    ) -> Union["Metadata", "Metadata_v04", "Metadata_v06", "Metadata_v09"]:
        """Convert metadata to specified NGFF version."""
        if isinstance(version, str):
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self._to_v04()
        if version == NgffVersion.V05:
            return self
        if version == NgffVersion.V06:
            return self._to_v06()
        if version in (NgffVersion.V09dev1, NgffVersion.V09dev3):
            from ..v09.zarr_metadata import Metadata as Metadata_v09

            return Metadata_v09.from_version(self)
        raise ValueError(f"Unsupported version conversion: 0.5 -> {version}")

    @classmethod
    def from_version(
        cls,
        metadata: Union["Metadata", "Metadata_v04", "Metadata_v06", "Metadata_v09"],
    ) -> "Metadata":
        """Convert metadata from specified NGFF version."""
        from ..v04.zarr_metadata import Metadata as Metadata_v04
        from ..v06.zarr_metadata import Metadata as Metadata_v06
        from ..v09.zarr_metadata import Metadata as Metadata_v09

        if isinstance(metadata, Metadata_v04):
            return cls._from_v04(metadata)
        if isinstance(metadata, Metadata_v09):
            return cls._from_v06(metadata._to_v06())
        if isinstance(metadata, Metadata_v06):
            return cls._from_v06(metadata)
        raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def _to_v06(self) -> "Metadata_v06":
        """Convert v0.5 metadata to v0.6."""
        from ..v06.zarr_metadata import Metadata as Metadata_v06

        return Metadata_v06.from_version(self)

    @classmethod
    def _from_v06(cls, metadata_v06: "Metadata_v06") -> "Metadata":
        """Convert v0.6 metadata to v0.5."""
        return metadata_v06._to_v05()

    def _to_v04(self) -> "Metadata_v04":
        from ..v04.zarr_metadata import Metadata as Metadata_v04

        metadata = Metadata_v04(
            axes=self.axes,
            datasets=self.datasets,
            coordinateTransformations=self.coordinateTransformations,
            name=self.name,
            metadata=self.metadata,
            type=self.type,
            omero=self.omero,
        )
        return metadata

    @classmethod
    def _from_v04(cls, metadata_v04: "Metadata_v04") -> "Metadata":

        metadata = cls(
            axes=metadata_v04.axes,
            datasets=metadata_v04.datasets,
            coordinateTransformations=metadata_v04.coordinateTransformations,
            name=metadata_v04.name,
            metadata=metadata_v04.metadata,
            type=metadata_v04.type,
            omero=metadata_v04.omero,
        )
        return metadata

    @classmethod
    def _from_zarr_attrs(
        cls,
        root_attrs: dict,
        store: StoreLike,
        validate: bool = False,
        subpath: str | None = None,
    ) -> tuple["Metadata", list["NgffImage"]]:  # noqa: F821
        from ..v04.zarr_metadata import Metadata as Metadata_v04

        # The v0.4 parser hoists the group-level ``version`` (here ``ome.version``)
        # so its structural pass sees the true spec version, which the v0.5
        # namespacing rules gate on. This entry point definitionally handles
        # v0.5, so floor ``ome.version`` to ``"0.5"`` when it is missing *or*
        # explicitly null (a malformed ``"version": null`` reads back as ``None``);
        # either way the store would otherwise be validated as v0.4 and skip the
        # v0.5 rules. Shallow-copy so the caller's attrs are untouched.
        ome_attrs = dict(root_attrs["ome"])
        if ome_attrs.get("version") is None:
            ome_attrs["version"] = "0.5"
        v4_metadata, images = Metadata_v04._from_zarr_attrs(
            ome_attrs, store, validate=validate, subpath=subpath
        )
        metadata = cls._from_v04(v4_metadata)
        return metadata, images

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.axes])
