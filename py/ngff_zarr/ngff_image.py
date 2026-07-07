# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from dask.array.core import Array as DaskArray

from .rfc4 import AnatomicalOrientation
from .v04.zarr_metadata import Units

ComputedCallback = Callable[[], None]


@dataclass
class NgffImage:
    data: DaskArray
    dims: Sequence[str]
    scale: dict[str, float]
    translation: dict[str, float]
    name: str = "image"
    axes_units: Mapping[str, Units] | None = None
    axes_orientations: Mapping[str, AnatomicalOrientation] | None = None
    axes_types: Mapping[str, str] | None = None
    channel_names: list[str] | None = None
    channel_colors: list[str] | None = None
    computed_callbacks: list[ComputedCallback] = field(default_factory=list)
