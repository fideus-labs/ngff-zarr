# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Constants for ngff-zarr package."""
from enum import StrEnum

class NgffVersion(StrEnum):
    V04 = "0.4"
    V05 = "0.5"
    LATEST = "0.5"

# Supported NGFF specification versions
SUPPORTED_VERSIONS = list(NgffVersion)