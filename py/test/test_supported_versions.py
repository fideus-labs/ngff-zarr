# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The version contract of the OME-Zarr 0.6 release (issue #565).

With the 0.6 release the pre-release tags ``0.6.dev4`` and ``0.6rc0`` leave the
public ``NgffVersion`` enum and ``SUPPORTED_VERSIONS``, and the on-disk tag for
the API version ``"0.6"`` is ``"0.6"`` itself. The tags stay readable: any
``0.6*`` string a store records is detected as v0.6, and the raw string is
still available to callers that compare it against what a writer would put
down. No store is written here; these are pure metadata checks.
"""

import pytest
from ngff_zarr import V06_ONDISK_VERSION, NgffVersion
from ngff_zarr._supported_versions import (
    SUPPORTED_VERSIONS,
    V06_SUPERSEDED_TAGS,
    is_v06_version,
)
from ngff_zarr.parse_metadata import _detect_version, _ondisk_version

PRERELEASE_TAGS = ("0.6rc0", "0.6.dev4")


@pytest.mark.parametrize("tag", PRERELEASE_TAGS)
def test_prerelease_tags_are_not_enum_members(tag):
    assert tag not in {member.value for member in NgffVersion}
    with pytest.raises(ValueError):
        NgffVersion(tag)


def test_supported_versions_list_the_release_and_the_next_draft_only():
    assert NgffVersion.V06 in SUPPORTED_VERSIONS
    assert "0.6" in SUPPORTED_VERSIONS
    assert NgffVersion.V09dev1 in SUPPORTED_VERSIONS
    assert "0.9.dev1" in SUPPORTED_VERSIONS
    for tag in PRERELEASE_TAGS:
        assert tag not in SUPPORTED_VERSIONS


def test_the_on_disk_tag_and_latest_are_the_release():
    assert V06_ONDISK_VERSION is NgffVersion.V06
    assert V06_ONDISK_VERSION.value == "0.6"
    assert NgffVersion.LATEST == "0.6"
    assert NgffVersion.LATEST is NgffVersion.V06


@pytest.mark.parametrize("tag", ["0.6", "0.6rc0", "0.6.dev4", "0.6.dev9"])
def test_is_v06_version_accepts_the_whole_family(tag):
    assert is_v06_version(tag)


@pytest.mark.parametrize("tag", ["0.5", "0.9.dev1", None])
def test_is_v06_version_rejects_other_versions(tag):
    assert not is_v06_version(tag)


@pytest.mark.parametrize("tag", ["0.6", *PRERELEASE_TAGS])
def test_a_0_6_family_tag_is_detected_as_v06_and_kept_verbatim(tag):
    root_attrs = {"ome": {"version": tag, "multiscales": [{}]}}
    assert _detect_version(root_attrs) is NgffVersion.V06
    assert _ondisk_version(root_attrs) == tag


def test_ondisk_version_falls_back_to_the_per_entry_tag():
    assert _ondisk_version({"multiscales": [{"version": "0.4"}]}) == "0.4"
    assert _ondisk_version({"ome": {"multiscales": [{}]}}) is None
    assert _ondisk_version({}) is None


def test_only_dev4_is_superseded():
    # The final ``_version.schema`` accepts ``0.6rc0``, so it is checked as
    # written; only ``0.6.dev4`` is substituted-and-warned by the reader.
    assert V06_SUPERSEDED_TAGS == frozenset({"0.6.dev4"})
    assert "0.6rc0" not in V06_SUPERSEDED_TAGS
