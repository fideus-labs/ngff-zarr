# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from pathlib import Path

import dask.config
from platformdirs import user_cache_dir
from zarr.storage import StoreLike
from ._zarr_kwargs import zarr_kwargs

if dask.config.get("temporary-directory") is not None:
    _store_dir = dask.config.get("temporary-directory")
else:
    _store_dir = Path(user_cache_dir("ngff-zarr"))


def default_store_factory():
    try:
        from zarr.storage import DirectoryStore

        return DirectoryStore(_store_dir, **zarr_kwargs)
    except ImportError:
        from zarr.storage import LocalStore

        return LocalStore(_store_dir)


try:
    import psutil

    default_memory_target = int(psutil.virtual_memory().available * 0.5)
except ImportError:
    default_memory_target = int(1e9)


@dataclass
class NgffZarrConfig:
    # Rough memory target in bytes
    memory_target: int = default_memory_target
    task_target: int = 50000
    cache_store: StoreLike = field(default_factory=default_store_factory)

    # HCS cache settings
    hcs_image_cache_size: int = 100  # Max number of images to cache per well
    hcs_well_cache_size: int = 500  # Max number of wells to cache per plate


config = NgffZarrConfig()
