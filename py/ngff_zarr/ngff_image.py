# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

from dask.array.core import Array as DaskArray

from .v04.zarr_metadata import Units
from .v06.zarr_metadata import Transform, CoordinateSystem, Scale, TransformSequence, Translation
from .rfc4 import AnatomicalOrientation

ComputedCallback = Callable[[], None]


@dataclass
class NgffImage:
    data: DaskArray
    coordinateTransformations: Optional[Transform] = None
    name: str = "image"
    axes_orientations: Optional[Mapping[str, AnatomicalOrientation]] = None
    computed_callbacks: List[ComputedCallback] = field(default_factory=list)

    @property
    def coordinateSystems(self) -> Union[List[CoordinateSystem], None]:
        if self.coordinateTransformations is not None:
            coordinate_systems = [self.coordinateTransformations.input, self.coordinateTransformations.output]
            return [cs for cs in coordinate_systems if isinstance(cs, CoordinateSystem)]
        return None

    @property
    def dims(self) -> List[str]:
        return [ax.name for ax in self.coordinateSystems[0].axes]
    
    @property
    def axes_units(self) -> Mapping[str, Units]:
        return {ax.name: ax.unit for ax in self.coordinateSystems[0].axes if ax.unit is not None}

    @property
    def scale(self) -> Mapping[str, float]:
        spatial_axes = [ax.name for ax in self.coordinateTransformations.output.axes if ax.type == "space"]

        t = None
        if isinstance(self.coordinateTransformations, Scale):
            t = self.coordinateTransformations

        elif isinstance(self.coordinateTransformations, TransformSequence):
            for transform in self.coordinateTransformations.sequence:
                if isinstance(transform, Scale):
                    t = transform
                    break

        if t is None:
            return {spatial_axes[i]: 1.0 for i in range(len(spatial_axes))}
        else:
            return {spatial_axes[i]: s for i, s in enumerate(t.scale)}
    
    @property
    def translation(self) -> Mapping[str, float]:
        spatial_axes = [ax.name for ax in self.coordinateTransformations.output.axes if ax.type == "space"]

        t = None
        if isinstance(self.coordinateTransformations, Translation):
            t = self.coordinateTransformations

        elif isinstance(self.coordinateTransformations, TransformSequence):
            for transform in self.coordinateTransformations.sequence:
                if isinstance(transform, Translation):
                    t = transform
                    break

        if t is None:
            return {spatial_axes[i]: 0.0 for i in range(len(spatial_axes))}
        else: 
            return {d: 1.0 for d in self.dims if d in spatial_axes}