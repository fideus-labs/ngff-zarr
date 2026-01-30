# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from typing_extensions import Literal
import re

# Import RFC 4 support
from ..rfc4 import AnatomicalOrientation
from .._supported_versions import NgffVersion
from .._zarr_types import StoreLike

if TYPE_CHECKING:
    from ..ngff_image import NgffImage

SupportedDims = Union[
    Literal["c"], Literal["x"], Literal["y"], Literal["z"], Literal["t"]
]

SpatialDims = Union[Literal["x"], Literal["y"], Literal["z"]]
AxesType = Union[Literal["time"], Literal["space"], Literal["channel"]]
SpaceUnits = Union[
    Literal["angstrom"],
    Literal["attometer"],
    Literal["centimeter"],
    Literal["decimeter"],
    Literal["exameter"],
    Literal["femtometer"],
    Literal["foot"],
    Literal["gigameter"],
    Literal["hectometer"],
    Literal["inch"],
    Literal["kilometer"],
    Literal["megameter"],
    Literal["meter"],
    Literal["micrometer"],
    Literal["mile"],
    Literal["millimeter"],
    Literal["nanometer"],
    Literal["parsec"],
    Literal["petameter"],
    Literal["picometer"],
    Literal["terameter"],
    Literal["yard"],
    Literal["yoctometer"],
    Literal["yottameter"],
    Literal["zeptometer"],
    Literal["zettameter"],
]
TimeUnits = Union[
    Literal["attosecond"],
    Literal["centisecond"],
    Literal["day"],
    Literal["decisecond"],
    Literal["exasecond"],
    Literal["femtosecond"],
    Literal["gigasecond"],
    Literal["hectosecond"],
    Literal["hour"],
    Literal["kilosecond"],
    Literal["megasecond"],
    Literal["microsecond"],
    Literal["millisecond"],
    Literal["minute"],
    Literal["nanosecond"],
    Literal["petasecond"],
    Literal["picosecond"],
    Literal["second"],
    Literal["terasecond"],
    Literal["yoctosecond"],
    Literal["yottasecond"],
    Literal["zeptosecond"],
    Literal["zettasecond"],
]
Units = Union[SpaceUnits, TimeUnits]

supported_dims = ["x", "y", "z", "c", "t"]

space_units = [
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]

time_units = [
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]


def is_dimension_supported(dim: str) -> bool:
    """Helper for string validation"""
    return dim in supported_dims


def is_unit_supported(unit: str) -> bool:
    """Helper for string validation"""
    return (unit in time_units) or (unit in space_units)


@dataclass
class Axis:
    name: SupportedDims
    type: AxesType
    unit: Optional[Units] = None
    orientation: Optional[AnatomicalOrientation] = None


@dataclass
class Identity:
    type: str = "identity"


@dataclass
class Scale:
    scale: List[float]
    type: str = "scale"


@dataclass
class Translation:
    translation: List[float]
    type: str = "translation"


Transform = Union[Scale, Translation]


@dataclass
class Dataset:
    path: str
    coordinateTransformations: List[Transform]


@dataclass
class OmeroWindow:
    min: float
    max: float
    start: float
    end: float


@dataclass
class OmeroChannel:
    color: str
    window: OmeroWindow
    label: Optional[str] = None

    def validate_color(self):
        if not re.fullmatch(r"[0-9A-Fa-f]{6}", self.color):
            raise ValueError(f"Invalid color '{self.color}'. Must be 6 hex digits.")


@dataclass
class Omero:
    channels: List[OmeroChannel]


@dataclass
class MethodMetadata:
    description: str
    method: str
    version: str


@dataclass
class PlateAcquisition:
    id: int
    name: Optional[str] = None
    maximumfieldcount: Optional[int] = None
    description: Optional[str] = None
    starttime: Optional[int] = None
    endtime: Optional[int] = None


@dataclass
class PlateColumn:
    name: str


@dataclass
class PlateRow:
    name: str


@dataclass
class PlateWell:
    path: str
    rowIndex: int
    columnIndex: int


@dataclass
class Plate:
    columns: List[PlateColumn]
    rows: List[PlateRow]
    wells: List[PlateWell]
    version: str = "0.4"
    acquisitions: Optional[List[PlateAcquisition]] = None
    field_count: Optional[int] = None
    name: Optional[str] = None


@dataclass
class WellImage:
    path: str
    acquisition: Optional[int] = None


@dataclass
class Well:
    images: List[WellImage]
    version: str = "0.4"


@dataclass
class Metadata:
    axes: List[Axis]
    datasets: List[Dataset]
    coordinateTransformations: Optional[List[Transform]]
    omero: Optional[Omero] = None
    name: str = "image"
    version: str = "0.4"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None

    def to_version(self, version: Union[str, NgffVersion]) -> "Metadata":
        if isinstance(version, str):
            # raise error for invalid version string
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self
        elif version == NgffVersion.V05:
            return self._to_v05()
        else:
            raise ValueError(f"Unsupported version conversion: 0.4 -> {version}")

    @classmethod
    def from_version(cls, metadata: "Metadata") -> "Metadata":
        from ..v05.zarr_metadata import Metadata as Metadata_v05

        if isinstance(metadata, Metadata_v05):
            return cls._from_v05(metadata)
        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def _to_v05(self) -> "Metadata":
        from ..v05.zarr_metadata import Metadata as Metadata_v05

        metadata = Metadata_v05(
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
    def _from_v05(cls, metadata_v05: "Metadata") -> "Metadata":
        metadata = cls(
            axes=metadata_v05.axes,
            datasets=metadata_v05.datasets,
            coordinateTransformations=metadata_v05.coordinateTransformations,
            name=metadata_v05.name,
            metadata=metadata_v05.metadata,
            type=metadata_v05.type,
            omero=metadata_v05.omero,
        )
        return metadata

    @classmethod
    def _from_zarr_attrs(
        cls,
        root_attrs: dict,
        store: StoreLike,
        validate: bool = False,
    ) -> tuple["Metadata", list["NgffImage"]]:
        """Create Metadata instance from ome-zarr metadata dictionary."""
        import sys
        import dask.array
        from ..validate import validate as validate_ngff
        from ..parse_metadata import _parse_omero
        from ..rfc4_validation import (
            validate_rfc4_orientation,
            has_rfc4_orientation_metadata,
        )
        from ..ngff_image import NgffImage

        if validate:
            validate_ngff(
                root_attrs, version=root_attrs["multiscales"][0].get("version", "0.4")
            )

            # RFC 4 validation for anatomical orientation
            if "axes" in root_attrs["multiscales"][0] and isinstance(
                root_attrs["multiscales"][0]["axes"], list
            ):
                # Type cast each axis item to dict for validation
                axes_dicts = []
                for axis in root_attrs["multiscales"][0]["axes"]:
                    if isinstance(axis, dict):
                        axes_dicts.append(axis)
                if axes_dicts and has_rfc4_orientation_metadata(axes_dicts):
                    validate_rfc4_orientation(axes_dicts)

        omero = _parse_omero(root_attrs.get("omero", None))
        root_attrs = root_attrs["multiscales"][0]

        # This handles backwards compatibility for version<=0.3
        if "axes" not in root_attrs:
            dims = tuple(reversed(supported_dims))
            axes = [
                Axis(name="t", type="time"),
                Axis(name="c", type="channel"),
                Axis(name="z", type="space"),
                Axis(name="y", type="space"),
                Axis(name="x", type="space"),
            ]
            units = {d: None for d in dims}
        else:
            dims = tuple(a["name"] if "name" in a else a for a in root_attrs["axes"])
            if "name" in root_attrs["axes"][0]:
                axes = [Axis(**axis) for axis in root_attrs["axes"]]
            else:
                # v0.3
                type_dict = {
                    "t": "time",
                    "c": "channel",
                    "z": "space",
                    "y": "space",
                    "x": "space",
                }
                axes = [
                    Axis(name=axis, type=type_dict[axis]) for axis in root_attrs["axes"]
                ]

            units = {d: None for d in dims}
            for axis in root_attrs["axes"]:
                # Only process unit information for dict-style axes that have both
                # a name and a unit (v0.4+). For v0.3 string axes, this loop is a no-op.
                if isinstance(axis, dict):
                    name = axis.get("name")
                    unit = axis.get("unit")
                    if name is not None and unit is not None:
                        units[name] = unit

        images = []
        datasets = []
        for dataset in root_attrs["datasets"]:
            data = dask.array.from_zarr(store, component=dataset["path"])
            # Convert endianness to native if needed
            if (sys.byteorder == "little" and data.dtype.byteorder == ">") or (
                sys.byteorder == "big" and data.dtype.byteorder == "<"
            ):
                data = data.astype(data.dtype.newbyteorder())

            scale = {d: 1.0 for d in dims}
            translation = {d: 0.0 for d in dims}
            coordinateTransformations = []
            if "coordinateTransformations" in dataset:
                for transformation in dataset["coordinateTransformations"]:
                    if "scale" in transformation:
                        scale = transformation["scale"]
                        scale = dict(zip(dims, scale))
                        coordinateTransformations.append(Scale(transformation["scale"]))
                    elif "translation" in transformation:
                        translation = transformation["translation"]
                        translation = dict(zip(dims, translation))
                        coordinateTransformations.append(
                            Translation(transformation["translation"])
                        )
            datasets.append(
                Dataset(
                    path=dataset["path"],
                    coordinateTransformations=coordinateTransformations,
                )
            )

            ngff_image = NgffImage(
                data=data,
                dims=dims,
                scale=scale,
                translation=translation,
                name=root_attrs.get("name", "image"),
                axes_units=units,
            )
            images.append(ngff_image)

        metadata = cls(
            axes=axes,
            datasets=datasets,
            name=root_attrs.get("name", "image"),
            version=root_attrs.get("version", "0.4"),
            omero=omero,
            coordinateTransformations=root_attrs.get("coordinateTransformations", None),
        )

        return metadata, images

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.axes])
