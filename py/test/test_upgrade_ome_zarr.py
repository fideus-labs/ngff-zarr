# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for :func:`ngff_zarr.upgrade_ome_zarr` (Phases 1 & 2).

Every fixture is synthesized in memory -- no pooch downloads and no network
access -- so these tests exercise the core upgrade behavior in isolation.

Phase 1 covered same-Zarr-format in-place upgrades (0.5<->0.6) and
write-to-new-store for everything. Phase 2 adds the cross-format in-place
upgrade over the Zarr v2<->v3 boundary (0.4->0.5, 0.4->0.6) -- a metadata-only
rewrite that leaves every chunk file byte-for-byte in place -- plus a full
(source, target) x (mode) matrix, no-op short-circuiting, metadata
preservation, and input hardening (unsupported versions, aliasing, HCS roots).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.storage
from ngff_zarr import (
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
    upgrade_ome_zarr,
)
from ngff_zarr.v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow
from packaging import version

zarr_version = version.parse(zarr.__version__)

# Skip when zarr predates the v3 support required for OME-Zarr >= 0.5.
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)

# The on-disk ``ome.version`` string a given API version is written as.
DISK_VERSION = {"0.4": "0.4", "0.5": "0.5", "0.6": "0.6.dev4"}

# Every metadata sidecar name across Zarr v2 and v3, so ``_chunk_files`` can
# isolate the true chunk *data* whose immutability we assert across an upgrade.
_METADATA_NAMES = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}


def _synth_multiscales():
    """Build a small in-memory 2-level multiscales plus its source pixels."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(1, 4, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=("c", "z", "y", "x"))
    return to_multiscales(image, scale_factors=[2]), data


def _chunk_files(root: Path) -> dict:
    """Map each chunk *data* file to ``(size, mtime_ns, bytes)``.

    Metadata files (Zarr v2 ``.z*`` sidecars and Zarr v3 ``zarr.json``) are
    excluded so the result holds only chunk data whose immutability we assert.
    """
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name not in _METADATA_NAMES:
            stat = path.stat()
            files[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                path.read_bytes(),
            )
    return files


def _all_files_digest(root: Path) -> dict:
    """Map every file under ``root`` to its sha256 digest (order-independent)."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_source(store, version_str: str):
    """Write a fresh synthesized multiscales to ``store`` at ``version_str``."""
    multiscales, data = _synth_multiscales()
    to_ome_zarr(store, multiscales, version=version_str)
    return data


def _leftover_v2_sidecars(root: Path) -> list:
    return [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.name in {".zarray", ".zattrs", ".zgroup", ".zmetadata"}
    ]


def _disk_ome_version(store_path, api_version: str) -> str:
    """Read the on-disk spec version, honoring where each format records it.

    v0.4 (Zarr v2) keeps it in ``multiscales[0].version`` at the root; v0.5+
    (Zarr v3) keeps it in the ``ome`` namespace.
    """
    if api_version == "0.4":
        root = zarr.open_group(str(store_path), mode="r", zarr_format=2)
        return root.attrs["multiscales"][0]["version"]
    root = zarr.open_group(str(store_path), mode="r", zarr_format=3)
    return root.attrs["ome"]["version"]


# --------------------------------------------------------------------------- #
# Phase 1 behavior (retained)
# --------------------------------------------------------------------------- #


def test_in_place_0_5_to_0_6(tmp_path):
    multiscales, data = _synth_multiscales()
    store_path = str(tmp_path / "image.ome.zarr")
    to_ome_zarr(store_path, multiscales, version="0.5")

    root_before = zarr.open_group(store_path, mode="r")
    assert root_before.attrs["ome"]["version"] == "0.5"
    chunks_before = _chunk_files(tmp_path)
    assert chunks_before, "expected chunk data files in the source store"

    upgrade_ome_zarr(store_path, version="0.6")

    root_after = zarr.open_group(store_path, mode="r")
    assert root_after.attrs["ome"]["version"] == "0.6.dev4"

    # Array chunk files are byte-identical and unmodified (path, size, mtime,
    # and content) -- only the root group metadata was rewritten.
    chunks_after = _chunk_files(tmp_path)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"array chunk file changed: {rel}"

    reloaded = from_ome_zarr(store_path, version="0.6", validate=True)
    ms0 = root_after.attrs["ome"]["multiscales"][0]
    assert "coordinateSystems" in ms0
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)


# --------------------------------------------------------------------------- #
# Phase 2: in-place same-format (0.5 <-> 0.6) -- chunks byte-identical
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source_version, target_version",
    [("0.5", "0.6"), ("0.6", "0.5")],
)
def test_in_place_same_format_preserves_chunks(
    tmp_path, source_version, target_version
):
    store_path = str(tmp_path / "image.ome.zarr")
    data = _write_source(store_path, source_version)
    chunks_before = _chunk_files(tmp_path)
    assert chunks_before

    upgrade_ome_zarr(store_path, version=target_version)

    chunks_after = _chunk_files(tmp_path)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"chunk changed: {rel}"

    root_after = zarr.open_group(store_path, mode="r")
    assert root_after.attrs["ome"]["version"] == DISK_VERSION[target_version]
    reloaded = from_ome_zarr(store_path, version=target_version, validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)


# --------------------------------------------------------------------------- #
# Phase 2: in-place cross-format upgrade (0.4 -> 0.5, 0.4 -> 0.6)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target_version", ["0.5", "0.6"])
def test_in_place_cross_format_upgrade_preserves_chunks(tmp_path, target_version):
    store_path = str(tmp_path / "image_v04.ome.zarr")
    data = _write_source(store_path, "0.4")

    chunks_before = _chunk_files(tmp_path)
    assert chunks_before, "expected chunk data files in the v0.4 source store"

    upgrade_ome_zarr(store_path, version=target_version)

    # Chunk data files are byte-identical (path, size, mtime, content): the
    # rewrite only rewrote metadata and never touched a chunk binary.
    chunks_after = _chunk_files(tmp_path)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"chunk changed: {rel}"

    # The store is cleanly Zarr v3: no v2 sidecars remain, and it opens as v3.
    assert _leftover_v2_sidecars(tmp_path) == []
    root = zarr.open_group(store_path, mode="r", zarr_format=3)
    assert root.attrs["ome"]["version"] == DISK_VERSION[target_version]

    # It reads back validated at the requested version with equal pixels.
    reloaded = from_ome_zarr(store_path, version=target_version, validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)


def test_in_place_cross_format_upgrade_dot_separator(tmp_path):
    """A v0.4 store using the Zarr v2 '.' chunk separator upgrades in place."""
    store_path = str(tmp_path / "dot.ome.zarr")
    group = zarr.open_group(store_path, mode="w", zarr_format=2)
    payload = np.arange(1 * 2 * 8 * 8, dtype="uint16").reshape(1, 2, 8, 8)
    arr = group.create_array(
        name="0",
        shape=payload.shape,
        chunks=(1, 1, 8, 8),
        dtype="uint16",
        chunk_key_encoding={"name": "v2", "separator": "."},
    )
    arr[:] = payload
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "image",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}
                    ],
                }
            ],
        }
    ]

    chunks_before = _chunk_files(tmp_path)
    # Confirm the '.'-separated chunk keys were actually written.
    assert any("." in Path(rel).name for rel in chunks_before), chunks_before

    upgrade_ome_zarr(store_path, version="0.5")

    chunks_after = _chunk_files(tmp_path)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"chunk changed: {rel}"
    assert _leftover_v2_sidecars(tmp_path) == []

    reloaded = from_ome_zarr(store_path, version="0.5", validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), payload)


# --------------------------------------------------------------------------- #
# Phase 2: in-place cross-format downgrade (0.5 -> 0.4, 0.6 -> 0.4) is rejected
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source_version", ["0.5", "0.6"])
def test_in_place_cross_format_downgrade_raises(tmp_path, source_version):
    store_path = str(tmp_path / "image.ome.zarr")
    _write_source(store_path, source_version)
    chunks_before = _chunk_files(tmp_path)

    with pytest.raises(ValueError, match="output"):
        upgrade_ome_zarr(store_path, version="0.4")

    # The failed downgrade left the store untouched.
    assert _chunk_files(tmp_path) == chunks_before


# --------------------------------------------------------------------------- #
# Phase 2: in-place same-version no-op leaves the store bit-for-bit unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("version_str", ["0.4", "0.5", "0.6"])
def test_in_place_noop_is_bit_for_bit(tmp_path, version_str):
    store_path = str(tmp_path / "image.ome.zarr")
    _write_source(store_path, version_str)
    digest_before = _all_files_digest(tmp_path)

    upgrade_ome_zarr(store_path, version=version_str)

    assert _all_files_digest(tmp_path) == digest_before


# --------------------------------------------------------------------------- #
# Phase 2: write-to-new-store for the full ordered version matrix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source_version", ["0.4", "0.5", "0.6"])
@pytest.mark.parametrize("target_version", ["0.4", "0.5", "0.6"])
def test_write_to_new_store_matrix(tmp_path, source_version, target_version):
    source_path = tmp_path / "source.ome.zarr"
    target_path = tmp_path / "target.ome.zarr"
    data = _write_source(str(source_path), source_version)

    source_before = _all_files_digest(source_path)

    upgrade_ome_zarr(str(source_path), str(target_path), version=target_version)

    # The source store is left completely unchanged.
    assert _all_files_digest(source_path) == source_before

    # The new store reads back validated at the requested version, pixels equal.
    assert (
        _disk_ome_version(target_path, target_version) == DISK_VERSION[target_version]
    )
    reloaded = from_ome_zarr(str(target_path), version=target_version, validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)


def test_write_to_new_store_rejects_store_objects(tmp_path):
    """zarr-python store objects are rejected with TypeError (breaking change).

    Store-to-store upgrades through in-memory zarr-python stores are no longer
    supported: only local paths, remote URLs, .ozx archives, and read-only
    bytes mappings are accepted. Path-based write-to-new-store coverage lives
    in test_write_to_new_store_matrix.
    """
    multiscales, _ = _synth_multiscales()

    # Write side: to_ome_zarr no longer accepts store objects.
    with pytest.raises(TypeError, match="no longer accepted"):
        to_ome_zarr(zarr.storage.MemoryStore(), multiscales, version="0.4")

    source_path = str(tmp_path / "source.ome.zarr")
    to_ome_zarr(source_path, multiscales, version="0.4")

    # upgrade_ome_zarr rejects store objects as the source ...
    with pytest.raises(TypeError, match="zarr-python store object"):
        upgrade_ome_zarr(
            zarr.storage.MemoryStore(),
            str(tmp_path / "target.ome.zarr"),
            version="0.6",
        )

    # ... and as the target.
    with pytest.raises(TypeError, match="no longer accepted"):
        upgrade_ome_zarr(source_path, zarr.storage.MemoryStore(), version="0.6")


# --------------------------------------------------------------------------- #
# Phase 2: non-geometry metadata (OMERO, name) survives an upgrade
# --------------------------------------------------------------------------- #


def _with_omero_and_name(multiscales):
    multiscales.metadata.name = "my special image"
    multiscales.metadata.omero = Omero(
        channels=[
            OmeroChannel(
                color="00FF00",
                window=OmeroWindow(min=0, max=255, start=5, end=250),
                label="green",
            )
        ]
    )
    return multiscales


@pytest.mark.parametrize(
    "source_version, target_version, mode",
    [
        ("0.4", "0.6", "in_place"),
        ("0.5", "0.6", "in_place"),
        ("0.4", "0.6", "new_store"),
    ],
)
def test_omero_and_name_round_trip(tmp_path, source_version, target_version, mode):
    multiscales, _ = _synth_multiscales()
    _with_omero_and_name(multiscales)
    source_path = str(tmp_path / "source.ome.zarr")
    to_ome_zarr(source_path, multiscales, version=source_version)

    if mode == "in_place":
        upgrade_ome_zarr(source_path, version=target_version)
        read_path = source_path
    else:
        read_path = str(tmp_path / "target.ome.zarr")
        upgrade_ome_zarr(source_path, read_path, version=target_version)

    root = zarr.open_group(read_path, mode="r")
    ome = root.attrs["ome"]
    ms0 = ome["multiscales"][0]
    assert ms0["name"] == "my special image"
    # For v0.5+ omero is hoisted next to multiscales inside the ome namespace.
    omero = ome.get("omero", ms0.get("omero"))
    assert omero is not None, "omero metadata was dropped by the upgrade"
    channel = omero["channels"][0]
    assert channel["label"] == "green"
    assert channel["color"] == "00FF00"


def test_method_metadata_and_orientation_survive_cross_format(tmp_path):
    """type/method ``metadata`` and RFC-4 orientation survive 0.4 -> 0.5."""
    rng = np.random.default_rng(2)
    data = rng.integers(0, 255, size=(1, 4, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=("c", "z", "y", "x"))
    multiscales = to_multiscales(image, scale_factors=[2], orientation="LPS")
    store_path = str(tmp_path / "image_v04.ome.zarr")
    to_ome_zarr(store_path, multiscales, version="0.4")

    upgrade_ome_zarr(store_path, version="0.5")

    root = zarr.open_group(store_path, mode="r", zarr_format=3)
    ms0 = root.attrs["ome"]["multiscales"][0]
    # Downsampling method type + metadata carried through.
    assert ms0.get("type") == "itkwasm_gaussian"
    assert ms0.get("metadata") is not None
    # RFC-4 anatomical orientation carried onto the axes.
    assert any("orientation" in axis for axis in ms0["axes"])


# --------------------------------------------------------------------------- #
# Phase 2: focused edge cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_version", ["0.3", "0.2", "0.1"])
def test_unsupported_target_version_raises(tmp_path, bad_version):
    store_path = str(tmp_path / "image.ome.zarr")
    _write_source(store_path, "0.5")
    with pytest.raises(ValueError, match="[Uu]nsupported target version"):
        upgrade_ome_zarr(store_path, version=bad_version)


def test_output_equal_to_input_routes_in_place(tmp_path):
    """Passing output == input must use the safe in-place path, not erase data."""
    store_path = str(tmp_path / "image.ome.zarr")
    data = _write_source(store_path, "0.5")
    chunks_before = _chunk_files(tmp_path)

    upgrade_ome_zarr(store_path, store_path, version="0.6")

    root_after = zarr.open_group(store_path, mode="r")
    assert root_after.attrs["ome"]["version"] == "0.6.dev4"
    chunks_after = _chunk_files(tmp_path)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"chunk changed: {rel}"
    reloaded = from_ome_zarr(store_path, version="0.6", validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)


def test_output_equal_to_input_cross_format(tmp_path):
    """Aliasing across the v2<->v3 boundary also routes to the in-place rewrite."""
    store_path = str(tmp_path / "image_v04.ome.zarr")
    data = _write_source(store_path, "0.4")
    chunks_before = _chunk_files(tmp_path)

    upgrade_ome_zarr(store_path, store_path, version="0.5")

    chunks_after = _chunk_files(tmp_path)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"chunk changed: {rel}"
    assert _leftover_v2_sidecars(tmp_path) == []
    reloaded = from_ome_zarr(store_path, version="0.5", validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)


def test_missing_store_surfaces_io_error(tmp_path):
    """A nonexistent store raises the real I/O error, not a version-detect error.

    Regression guard: the root-attribute reader only swallows the Zarr
    format-mismatch case (so a v0.4/v2 store can be retried as Zarr v2); a
    genuine missing/unreadable store must propagate instead of degrading into a
    generic empty-attributes version-detection failure.
    """
    missing = str(tmp_path / "does_not_exist.ome.zarr")
    with pytest.raises(FileNotFoundError):
        upgrade_ome_zarr(missing, version="0.6")


def test_hcs_plate_root_raises_guidance(tmp_path):
    """An HCS plate root is rejected with guidance toward the per-image path."""
    plate_path = str(tmp_path / "plate.ome.zarr")
    group = zarr.open_group(plate_path, mode="w", zarr_format=2)
    group.attrs["plate"] = {
        "columns": [{"name": "1"}],
        "rows": [{"name": "A"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
        "version": "0.4",
    }
    with pytest.raises(ValueError, match="[Pp]late"):
        upgrade_ome_zarr(plate_path, version="0.5")


def test_bioformats2raw_container_root_raises_guidance(tmp_path):
    """A bioformats2raw container root is rejected with per-image guidance."""
    container_path = str(tmp_path / "container.ome.zarr")
    group = zarr.open_group(container_path, mode="w", zarr_format=2)
    group.attrs["bioformats2raw.layout"] = 3
    with pytest.raises(ValueError, match="bioformats2raw"):
        upgrade_ome_zarr(container_path, version="0.5")
