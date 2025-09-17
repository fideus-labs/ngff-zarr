# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)
if zarr_version >= version.parse("3.0.0b1"):
    from zarr.api.synchronous import open_array
else:
    from zarr.creation import open_array
open_array = open_array
