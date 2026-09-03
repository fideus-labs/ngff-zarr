# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The axis types the specification names."""

from enum import StrEnum


class AxisType(StrEnum):
    """An axis type the specification names.

    RFC-3 (OME-Zarr 0.9.dev1) lets an axis carry any string, so this is the
    named set rather than the permitted one: an API that takes an axis type
    is annotated ``AxisType | str`` and accepts both. The members are the
    values themselves, so ``AxisType.Space == "space"``.

    ``Displacement`` and ``Coordinate`` arrive with RFC-5 (OME-Zarr 0.6) and
    type the component axis of a field; the other three are in the format
    from 0.4 on. Mirrors the TypeScript port's ``AxisType``.
    """

    Space = "space"
    Time = "time"
    Channel = "channel"
    Displacement = "displacement"
    Coordinate = "coordinate"
