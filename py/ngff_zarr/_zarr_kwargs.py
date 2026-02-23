# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)
if zarr_version >= version.parse("3.0.0b1"):
    zarr_kwargs = {"chunk_key_encoding": {"name": "default", "separator": "/"}}
else:
    zarr_kwargs = {"dimension_separator": "/"}
