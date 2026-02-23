# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import List, Optional, Union
from dataclasses import dataclass

from ..v04.zarr_metadata import Axis, Transform, Dataset, Omero, MethodMetadata
from .._supported_versions import NgffVersion
from .._zarr_types import StoreLike


@dataclass
class Metadata:
    axes: List[Axis]
    datasets: List[Dataset]
    coordinateTransformations: Optional[List[Transform]]
    omero: Optional[Omero] = None
    name: str = "image"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None

    def to_version(self, version: Union[str, NgffVersion]) -> "Metadata":
        """Convert metadata to specified NGFF version."""
        if isinstance(version, str):
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self._to_v04()
        elif version == NgffVersion.V05:
            return self
        else:
            raise ValueError(f"Unsupported version conversion: 0.5 -> {version}")

    @classmethod
    def from_version(cls, metadata: "Metadata") -> "Metadata":
        """Convert metadata from specified NGFF version."""
        from ..v04.zarr_metadata import Metadata as Metadata_v04

        if isinstance(metadata, Metadata_v04):
            return cls._from_v04(metadata)
        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def _to_v04(self) -> "Metadata":
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
    def _from_v04(cls, metadata_v04: "Metadata") -> "Metadata":
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
    ) -> tuple["Metadata", list["NgffImage"]]:  # noqa: F821
        from ..v04.zarr_metadata import Metadata as Metadata_v04

        v4_metadata, images = Metadata_v04._from_zarr_attrs(
            root_attrs["ome"], store, validate=validate
        )
        metadata = cls._from_v04(v4_metadata)
        return metadata, images

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.axes])
