# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._zarr_types import StoreLike
from .methods import Methods
from .ngff_image import NgffImage
from .v04.zarr_metadata import Metadata as Metadata_v04
from .v05.zarr_metadata import Metadata as Metadata_v05


@dataclass
class Multiscales:
    images: list[NgffImage]
    metadata: Metadata_v04 | Metadata_v05
    scale_factors: Sequence[dict[str, int] | int] | None = None
    method: Methods | None = None
    chunks: (
        int
        | tuple[int, ...]
        | tuple[tuple[int, ...], ...]
        | Mapping[Any, None | int | tuple[int, ...]]
        | None
    ) = None

    def to_ome_zarr(self, store: StoreLike, **kwargs: Any) -> None:
        """Write this multiscale image to an OME-Zarr store.

        Convenience method that delegates to :func:`ngff_zarr.to_ome_zarr`.
        See that function for full parameter documentation.
        """
        from .to_ngff_zarr import to_ome_zarr

        to_ome_zarr(store, self, **kwargs)

    @classmethod
    def from_ome_zarr(cls, store: StoreLike, **kwargs: Any) -> Multiscales:
        """Read an OME-Zarr store into a Multiscales object.

        Convenience classmethod that delegates to :func:`ngff_zarr.from_ome_zarr`.
        See that function for full parameter documentation.
        """
        from .from_ngff_zarr import from_ome_zarr

        return from_ome_zarr(store, **kwargs)
