# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import List, Optional, Union

from typing_extensions import Literal
import re

# Import RFC 4 support
from ..rfc4 import AnatomicalOrientation
from .._supported_versions import SUPPORTED_VERSIONS, NgffVersion

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


    def to_version(self, version: str) -> "Metadata":
        if isinstance(version, str):
            # raise error for invalid version string
            version = NgffVersion(version)

        match version:
            case NgffVersion.V05:
                return self._to_v05()
            case NgffVersion.V04:
                return self
            case _:
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

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.axes])