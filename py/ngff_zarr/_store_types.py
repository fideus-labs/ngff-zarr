# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Store type aliases for ngff-zarr's public API.

``StoreLike`` captures what the read/write entry points accept: a local
directory path (``str``/``pathlib.Path``/``os.PathLike``), a remote URL
string, a ``.zip``/``.ozx`` archive path, or a key-to-bytes
``MutableMapping`` (read through the pure-Python store reader).
"""

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Union

StoreLike = Union[str, Path, os.PathLike, MutableMapping]
