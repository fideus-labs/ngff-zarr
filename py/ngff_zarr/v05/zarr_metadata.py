# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import List, Optional
from dataclasses import dataclass

from ..v04.zarr_metadata import Axis, Transform, Dataset, Omero, MethodMetadata


@dataclass
class Metadata:
    axes: List[Axis]
    datasets: List[Dataset]
    coordinateTransformations: Optional[List[Transform]]
    omero: Optional[Omero] = None
    name: str = "image"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None
