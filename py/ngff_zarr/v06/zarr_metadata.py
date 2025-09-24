from typing import List, Optional, Union
from dataclasses import dataclass

from ..v04.zarr_metadata import Omero, MethodMetadata, AxesType, SupportedDims, Units
from ..rfc4 import AnatomicalOrientation
# from ..rfc5 import (
#     NgffBaseTransformation, NgffCoordinateSystem
# )

@dataclass
class Axis:
    name: SupportedDims
    type: AxesType
    unit: Optional[Units] = None
    orientation: Optional[AnatomicalOrientation] = None

@dataclass
class CoordinateSystem:
    name: str
    axes: List[Axis]

@dataclass
class Identity:
    type: str = "identity"
    name: str = ''
    input: Union[str, CoordinateSystem] = ''
    output: Union[str, CoordinateSystem] = ''


@dataclass
class Scale:
    scale: List[float]
    type: str = "scale"
    name: str = ''
    input: Union[str, CoordinateSystem] = ''
    output: Union[str, CoordinateSystem] = ''


@dataclass
class Translation:
    translation: List[float]
    type: str = "translation"
    name: str = ''
    input: Union[str, CoordinateSystem] = ''
    output: Union[str, CoordinateSystem] = ''

@dataclass
class TransformSequence:
    sequence: List[Union[Scale, Translation]]
    name: str = ''
    type: str = "sequence"
    input: Union[str, CoordinateSystem] = ''
    output: Union[str, CoordinateSystem] = ''

Transform = Union[Scale, Translation, TransformSequence, Identity]

@dataclass
class Dataset:
    path: str
    coordinateTransformations: Transform


@dataclass
class Metadata:
    coordinateSystems: Union[CoordinateSystem, List[CoordinateSystem]]
    coordinateTransformations: Transform
    datasets: List[Dataset]
    omero: Optional[Omero] = None
    name: str = "image"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None

    @property
    def axes(self) -> List:

        if isinstance(self.coordinateSystems, list):
            if len(self.coordinateSystems) == 1:
                return self.coordinateSystems[0].axes
            return {cs.name: cs.axes for cs in self.coordinateSystems}
        return self.coordinateSystems.axes
