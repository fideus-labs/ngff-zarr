# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from dask.array.core import Array as DaskArray

from .v04.zarr_metadata import Units
from .rfc4 import AnatomicalOrientation

ComputedCallback = Callable[[], None]


@dataclass
class NgffImage:
    data: DaskArray
    dims: Sequence[str]
    scale: Dict[str, float]
    translation: Dict[str, float]
    name: str = "image"
    axes_units: Optional[Mapping[str, Units]] = None
    axes_orientations: Optional[Mapping[str, AnatomicalOrientation]] = None
    computed_callbacks: List[ComputedCallback] = field(default_factory=list)
