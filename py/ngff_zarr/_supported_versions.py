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
    #: Pre-release tags of the 0.6 spec. Both remain readable: stores written
    #: while 0.6 was a draft carry the ``dev4`` tag on disk.
    V06dev4 = "0.6.dev4"
    V06rc0 = "0.6rc0"
    LATEST = "0.6rc0"


# Supported NGFF specification versions
SUPPORTED_VERSIONS = (
    NgffVersion.V01,
    NgffVersion.V02,
    NgffVersion.V03,
    NgffVersion.V04,
    NgffVersion.V05,
    NgffVersion.V06,
    NgffVersion.V06dev4,
    NgffVersion.V06rc0,
)

#: The ``ome.version`` string written to disk for the API version ``"0.6"``.
#: The 0.6 spec is a release candidate, so a store is tagged with the
#: pre-release the bundled ``spec/0.6`` schemas carry rather than with the bare
#: ``"0.6"`` the public ``version`` option accepts. Mirrors the TypeScript
#: port's ``V06_ONDISK_VERSION``; this constant is the one place the tag lives.
V06_ONDISK_VERSION = NgffVersion.V06rc0

#: 0.6 pre-release tags that earlier ngff-zarr releases wrote and that the
#: bundled schemas no longer accept. A store carrying one differs from a valid
#: store in that string alone; the validating reader says so and checks the
#: rest, and ``upgrade_ome_zarr`` rewrites the tag. Any other tag is checked
#: as given, so a tag from a later spec release is not passed off as this one.
V06_SUPERSEDED_TAGS = frozenset({NgffVersion.V06dev4.value})
