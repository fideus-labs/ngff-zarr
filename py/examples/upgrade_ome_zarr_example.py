#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Runnable, network-free end-to-end example of :func:`upgrade_ome_zarr`.

Demonstrates both upgrade modes on a single synthesized multiscale image:

1. **Write-to-new-store** across the Zarr v2 <-> v3 boundary: a store written at
   OME-Zarr 0.4 (Zarr v2) is upgraded to 0.6 (Zarr v3) at a *new* location,
   leaving the 0.4 source untouched.
2. **In-place metadata-only** within Zarr v3: a fresh copy written at 0.5 is
   upgraded to 0.6 in place -- only the root group's ``zarr.json`` is rewritten,
   and every array chunk file on disk is left byte-for-byte identical. This is
   the fix for issue #219 (https://github.com/fideus-labs/ngff-zarr/issues/219),
   where an earlier erase-then-reload approach destroyed the array data it was
   meant to preserve.

For each mode the script prints the before/after on-disk spec versions and
confirms the upgraded store reads back with its array data intact (pixel values
equal to the source). It requires no network access and no user input.

Run from the ``py/`` directory:

    pixi run -e test python examples/upgrade_ome_zarr_example.py

The process exits non-zero if any version does not update as expected, if the
in-place upgrade touches a chunk file, or if the reloaded pixels differ from the
source, so the script doubles as a self-checking smoke test.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr
from ngff_zarr import (
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
    upgrade_ome_zarr,
)

# The root group metadata file for a Zarr v3 store. An in-place same-format
# upgrade rewrites only this file; per-array metadata and chunks are untouched.
ROOT_METADATA = "zarr.json"


def read_ome_version(store_path: str) -> str:
    """Return the on-disk OME-Zarr spec version recorded at ``store_path``.

    Handles both a Zarr v3 store (0.5/0.6, version under the ``ome`` key) and a
    Zarr v2 store (0.4, version under the top-level ``multiscales``), mirroring
    the reader's own v2 fallback so a 0.4 store opened without an explicit
    format still resolves.
    """
    attrs = zarr.open_group(store_path, mode="r").attrs.asdict()
    if not attrs:
        attrs = zarr.open_group(store_path, mode="r", zarr_format=2).attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"]["version"]
    return attrs["multiscales"][0]["version"]


def chunk_digests(root: Path) -> dict[str, str]:
    """Map every array/chunk data file under ``root`` to its SHA-256 digest.

    Excludes the ``zarr.json`` / ``.zarray`` / ``.zattrs`` / ``.zgroup``
    metadata sidecars so the result is exactly the array chunk payload -- the
    bytes an in-place upgrade must never touch.
    """
    metadata_names = {ROOT_METADATA, ".zarray", ".zattrs", ".zgroup", ".zmetadata"}
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in metadata_names:
            rel = path.relative_to(root).as_posix()
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def synthesize_multiscales():
    """Build a small deterministic 2-level ``c, z, y, x`` multiscale image."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(1, 4, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=("c", "z", "y", "x"))
    return to_multiscales(image, scale_factors=[2]), data


def demo_write_to_new_store(tmp: Path) -> None:
    """0.4 -> 0.6 write-to-new-store: the 0.4 source is left intact."""
    print("== Mode 1: write-to-new-store (0.4 -> 0.6, Zarr v2 -> v3) ==")
    multiscales, source_data = synthesize_multiscales()

    src = str(tmp / "image_v04.ome.zarr")
    dst = str(tmp / "image_v06.ome.zarr")

    to_ome_zarr(src, multiscales, version="0.4")
    src_before = read_ome_version(src)
    print(f"  source version (before) : {src_before!r}")
    assert src_before == "0.4", src_before

    # Upgrade into a brand-new store; the source is read lazily, never erased.
    upgrade_ome_zarr(src, dst, version="0.6")

    src_after = read_ome_version(src)
    dst_after = read_ome_version(dst)
    print(f"  source version (after)  : {src_after!r}  (unchanged)")
    print(f"  new store version       : {dst_after!r}")
    assert src_after == "0.4", src_after
    assert dst_after == "0.6.dev4", dst_after

    # Confirm the upgraded store reads back with pixel data intact.
    reloaded = from_ome_zarr(dst, version="0.6")
    full_res = np.asarray(reloaded.images[0].data)
    print(f"  reloaded full-res shape : {full_res.shape}")
    assert np.array_equal(full_res, source_data), "pixel data changed on upgrade"
    print("  OK: 0.4 source preserved, 0.6 copy reads back with data intact\n")


def demo_in_place(tmp: Path) -> None:
    """0.5 -> 0.6 in place: metadata-only, array chunks left byte-identical."""
    print("== Mode 2: in-place metadata-only (0.5 -> 0.6, Zarr v3) ==")
    multiscales, source_data = synthesize_multiscales()

    store = str(tmp / "image_v05.ome.zarr")
    store_root = Path(store)

    to_ome_zarr(store, multiscales, version="0.5")
    before_version = read_ome_version(store)
    before_chunks = chunk_digests(store_root)
    print(f"  version (before)        : {before_version!r}")
    print(f"  array/chunk data files  : {len(before_chunks)}")
    assert before_version == "0.5", before_version

    # In-place upgrade: rewrites only the root zarr.json.
    upgrade_ome_zarr(store, version="0.6")

    after_version = read_ome_version(store)
    after_chunks = chunk_digests(store_root)
    print(f"  version (after)         : {after_version!r}")
    assert after_version == "0.6.dev4", after_version

    # Prove every array chunk file is byte-for-byte identical.
    assert after_chunks == before_chunks, "array chunk data changed on in-place upgrade"
    print(f"  {len(after_chunks)} array/chunk data files byte-for-byte identical")

    # Confirm the upgraded store reads back with pixel data intact.
    reloaded = from_ome_zarr(store, version="0.6")
    full_res = np.asarray(reloaded.images[0].data)
    print(f"  reloaded full-res shape : {full_res.shape}")
    assert np.array_equal(full_res, source_data), "pixel data changed on upgrade"
    print("  OK: metadata upgraded 0.5 -> 0.6, chunks untouched, data intact\n")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ngff_upgrade_example_"))
    try:
        demo_write_to_new_store(tmp)
        demo_in_place(tmp)
        print("SUCCESS: both upgrade modes completed with array data intact")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
