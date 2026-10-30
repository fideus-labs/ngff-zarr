# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from pathlib import Path

import dask.config
from platformdirs import user_cache_dir

from ._store_types import StoreLike

if dask.config.get("temporary-directory") is not None:
    _store_dir = dask.config.get("temporary-directory")
else:
    _store_dir = Path(user_cache_dir("ngff-zarr"))

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
    # The cache directory path; large-image cache arrays are written into it
    # through zarrista.
    cache_store: StoreLike = field(default_factory=lambda: _store_dir)

    # Pyodide only: where to import the @fideus-labs/ngff-zarr JS package
    # from when the host did not register it as the ``ngff_zarr_js`` module.
    js_package_url: str = "https://esm.sh/@fideus-labs/ngff-zarr@0.39.0/browser"

    # HCS cache settings
    hcs_image_cache_size: int = 100  # Max number of images to cache per well
    hcs_well_cache_size: int = 500  # Max number of wells to cache per plate


config = NgffZarrConfig()
