# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Constants for ngff-zarr package."""

from enum import StrEnum


class NgffVersion(StrEnum):
    V01 = "0.1"
    V02 = "0.2"
    V03 = "0.3"
    V04 = "0.4"
    V05 = "0.5"
    V06 = "0.6"
    # OME-Zarr 0.9 in development: 0.6 plus RFC-3 (any axis count, names,
    # types and ordering). LATEST stays 0.6, so 0.9.dev1 is opt-in.
    V09dev1 = "0.9.dev1"
    # An alias of V06 (same value): it must stay last.
    LATEST = "0.6"


# Supported NGFF specification versions
SUPPORTED_VERSIONS = (
    NgffVersion.V01,
    NgffVersion.V02,
    NgffVersion.V03,
    NgffVersion.V04,
    NgffVersion.V05,
    NgffVersion.V06,
    NgffVersion.V09dev1,
)

#: The ``ome.version`` string written to disk for the API version ``"0.6"``.
#: With the 0.6 release this is ``"0.6"`` itself: the bundled ``spec/0.6``
#: schemas carry the released tag, so the string the public ``version`` option
#: accepts is the one a store is tagged with. The constant stays because it is
#: a public export and the one place both ports read the tag from; mirrors the
#: TypeScript port's ``V06_ONDISK_VERSION``.
V06_ONDISK_VERSION = NgffVersion.V06

#: 0.6 pre-release tags that earlier ngff-zarr releases wrote and that the
#: bundled schemas no longer accept. The final ``_version.schema`` lists
#: ``0.6`` and ``0.6rc0``, so an ``0.6rc0`` store validates as-is; only
#: ``0.6.dev4`` is left. A store carrying it differs from a valid store in that
#: string alone; the validating reader substitutes ``V06_ONDISK_VERSION``,
#: warns, and checks the rest, and ``upgrade_ome_zarr`` rewrites either
#: pre-release tag to ``0.6``. Any other tag is checked as given, so a tag from
#: a later spec release is not passed off as this one. A literal string, since
#: the pre-release tags are no longer :class:`NgffVersion` members.
V06_SUPERSEDED_TAGS = frozenset({"0.6.dev4"})


def is_v06_version(version: object | None) -> bool:
    """Whether ``version`` identifies OME-Zarr v0.6, dev releases included.

    Mirrors the TypeScript port's ``isV06Version`` so a store written by
    either implementation is recognized the same way. Anything that is not a
    string -- ``None`` included -- is not v0.6.

    Tested with ``isinstance`` rather than ``str()``: a :class:`NgffVersion`
    member is a ``str`` subclass under both the stdlib ``StrEnum`` (3.11+) and
    a ``str, Enum`` backport, but only the former renders as its value under
    ``str()``; the backport renders as ``"NgffVersion.V06"``.
    """
    return isinstance(version, str) and version.startswith("0.6")
