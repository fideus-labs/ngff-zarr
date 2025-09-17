# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .methods import Methods
from .ngff_image import NgffImage
from .v04.zarr_metadata import Metadata as Metadata_v04
from .v05.zarr_metadata import Metadata as Metadata_v05


@dataclass
class Multiscales:
    images: List[NgffImage]
    metadata: Union[Metadata_v04, Metadata_v05]
    scale_factors: Optional[Sequence[Union[Dict[str, int], int]]] = None
    method: Optional[Methods] = None
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None
