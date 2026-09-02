# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from dask.array.core import Array as DaskArray

from .axis_type import AxisType
from .rfc4 import AnatomicalOrientation
from .v04.zarr_metadata import AxisUnit

ComputedCallback = Callable[[], None]


@dataclass
class NgffImage:
    data: DaskArray
    dims: Sequence[str]
    scale: dict[str, float]
    translation: dict[str, float]
    name: str = "image"
    axes_units: Mapping[str, AxisUnit] | None = None
    axes_orientations: Mapping[str, AnatomicalOrientation] | None = None
    axes_types: Mapping[str, AxisType | str] | None = None
    channel_names: list[str] | None = None
    channel_colors: list[str] | None = None
    computed_callbacks: list[ComputedCallback] = field(default_factory=list)


#: The axis type to_multiscales gives a dimension when the image carries none.
_DEFAULT_AXIS_TYPES = {
    "x": AxisType.Space,
    "y": AxisType.Space,
    "z": AxisType.Space,
    "c": AxisType.Channel,
    "t": AxisType.Time,
}


def non_default_axes_types(axes) -> dict[str, AxisType | str] | None:
    """Axis types that a multiscales rebuilt from the dims alone would lose.

    Readers pass these on as ``NgffImage.axes_types`` so a displacement or
    coordinate field stays typed through to_multiscales and a later write.
    """
    types = {
        axis.name: axis.type
        for axis in axes
        if axis.type is not None and axis.type != _DEFAULT_AXIS_TYPES.get(axis.name)
    }
    return types or None
