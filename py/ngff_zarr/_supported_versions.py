# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Constants for ngff-zarr package."""

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Backward compatibility for Python < 3.11
    class StrEnum(str, Enum):
        """
        Enum where members are also (and must be) strings
        """

        def __new__(cls, value):
            if not isinstance(value, str):
                raise TypeError(f"{cls.__name__} values must be strings")
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj


class NgffVersion(StrEnum):
    V01 = "0.1"
    V02 = "0.2"
    V03 = "0.3"
    V04 = "0.4"
    V05 = "0.5"
    LATEST = "0.5"


# Supported NGFF specification versions
SUPPORTED_VERSIONS = (
    NgffVersion.V01,
    NgffVersion.V02,
    NgffVersion.V03,
    NgffVersion.V04,
    NgffVersion.V05,
)
