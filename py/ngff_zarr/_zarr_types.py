# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from collections.abc import MutableMapping
from pathlib import Path
from typing import Union

import zarr
import zarr.storage

# Zarr type definitions for compatibility with Zarr Python 2 and 3
if hasattr(zarr.storage, "StoreLike"):
    StoreLike = zarr.storage.StoreLike
else:
    StoreLike = Union[MutableMapping, str, Path, zarr.storage.BaseStore]
