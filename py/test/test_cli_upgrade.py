# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the ``ngff-zarr upgrade`` CLI subcommand (Phase 3).

The subcommand is exercised end-to-end through the same subprocess mechanism
the other ``test_cli_*.py`` modules use, so the non-breaking ``main()``
dispatch, argument parsing, and Rich feedback are all covered as a user would
hit them. Every store is synthesized in memory/on tmp_path -- no pooch
downloads and no network access.
"""

import os
import subprocess
import sys

import numpy as np
import packaging.version
import pytest
import zarr
from ngff_zarr import to_multiscales, to_ngff_image, to_ome_zarr

zarr_version = packaging.version.parse(zarr.__version__)

# OME-Zarr >= 0.5 requires the Zarr v3 support added in zarr-python 3.0.0b1.
pytestmark = pytest.mark.skipif(
    zarr_version < packaging.version.parse("3.0.0b1"),
    reason="zarr version < 3.0.0b1",
)

# The on-disk ``ome.version`` string each API version is written as.
DISK_VERSION = {"0.4": "0.4", "0.5": "0.5", "0.6": "0.6.dev4"}

# Metadata sidecars across Zarr v2 and v3, excluded when isolating chunk data.
_METADATA_NAMES = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}


def _run_ngff_zarr(*args):
    """Run the ngff-zarr CLI as a subprocess and return the result."""
    cmd = [
        sys.executable,
        "-c",
        "from ngff_zarr.cli import main; main()",
    ] + list(args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _synth_multiscales():
    """Build a small in-memory 2-level multiscales plus its source pixels."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(1, 4, 32, 32), dtype=np.uint8)
    image = to_ngff_image(data, dims=("c", "z", "y", "x"))
    return to_multiscales(image, scale_factors=[2]), data


def _write_source(store, version_str):
    """Write a fresh synthesized multiscales to ``store`` at ``version_str``."""
    multiscales, data = _synth_multiscales()
    to_ome_zarr(store, multiscales, version=version_str)
    return data


def _chunk_files(root):
    """Map each chunk *data* file to ``(size, mtime_ns, bytes)``.

    Metadata sidecars are excluded so the result holds only chunk data whose
    immutability we assert across an in-place upgrade.
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


def _all_files_digest(root):
    """Map every file under ``root`` to its size + bytes (order-independent)."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_in_place_upgrade_0_5_to_0_6(tmp_path):
    """In-place 0.5 -> 0.6 rewrites metadata only; chunk files are untouched."""
    store = tmp_path / "image.ome.zarr"
    _write_source(str(store), "0.5")
    chunks_before = _chunk_files(store)
    assert chunks_before, "expected chunk data files in the source store"

    result = _run_ngff_zarr("upgrade", str(store), "--to", "0.6")
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    # The on-disk spec version was bumped to the 0.6 development string.
    root = zarr.open_group(str(store), mode="r", zarr_format=3)
    assert root.attrs["ome"]["version"] == DISK_VERSION["0.6"]

    # Every array chunk file is byte-identical and unmodified.
    chunks_after = _chunk_files(store)
    assert set(chunks_after) == set(chunks_before)
    for rel, sig in chunks_before.items():
        assert chunks_after[rel] == sig, f"array chunk file changed: {rel}"

    # The Rich summary names the mode and reassures about the chunks. Collapse
    # whitespace first: Rich soft-wraps the summary to the (non-TTY) console
    # width, which can split the message mid-phrase when the input path is long
    # (e.g. on the Windows CI runner), so assert against width-independent text.
    summary = " ".join(result.stdout.split())
    assert "in-place" in summary
    assert "chunks left untouched" in summary


def test_write_to_new_store_0_4_to_0_5(tmp_path):
    """0.4 -> 0.5 written to a new store; the source is left unchanged."""
    from ngff_zarr import from_ome_zarr

    src = tmp_path / "source.ome.zarr"
    dst = tmp_path / "target.ome.zarr"
    data = _write_source(str(src), "0.4")
    source_before = _all_files_digest(src)

    result = _run_ngff_zarr("upgrade", str(src), "-o", str(dst), "--to", "0.5")
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    # The destination reads back validated at 0.5 with equal pixels.
    dst_root = zarr.open_group(str(dst), mode="r", zarr_format=3)
    assert dst_root.attrs["ome"]["version"] == DISK_VERSION["0.5"]
    reloaded = from_ome_zarr(str(dst), version="0.5", validate=True)
    np.testing.assert_array_equal(reloaded.images[0].data.compute(), data)

    # The source store is completely unchanged.
    assert _all_files_digest(src) == source_before


def test_in_place_cross_format_downgrade_exits_nonzero(tmp_path):
    """0.6 -> 0.4 in place is rejected with guidance and a non-zero exit code."""
    store = tmp_path / "image.ome.zarr"
    _write_source(str(store), "0.6")
    chunks_before = _chunk_files(store)

    result = _run_ngff_zarr("upgrade", str(store), "--to", "0.4")
    assert result.returncode != 0

    # A concise, actionable message (not a traceback) directs the user to
    # write to a new output store.
    combined = result.stdout + result.stderr
    assert "output" in combined
    assert "Traceback" not in combined

    # The rejected downgrade left the store untouched.
    assert _chunk_files(store) == chunks_before


def test_invalid_target_version_rejected(tmp_path):
    """An unsupported --to value is rejected by argparse choices (exit != 0)."""
    store = tmp_path / "image.ome.zarr"
    _write_source(str(store), "0.5")

    result = _run_ngff_zarr("upgrade", str(store), "--to", "9.9")
    assert result.returncode != 0
    assert "9.9" in (result.stdout + result.stderr)


def test_upgrade_help_mentions_both_modes():
    """``ngff-zarr upgrade --help`` runs and describes both upgrade modes."""
    result = _run_ngff_zarr("upgrade", "--help")
    assert result.returncode == 0

    # Collapse wrapping so phrases split across lines still match.
    help_text = " ".join(result.stdout.split())
    assert "in-place" in help_text
    assert "new store" in help_text
    assert "--output" in help_text


def test_non_upgrade_invocation_routes_to_flat_parser():
    """A non-``upgrade`` first token still hits the flat conversion parser.

    Network-free: we only need to prove the dispatch fell through to the flat
    parser, which is demonstrated by its own required-argument enforcement.
    """
    # ``-o`` without ``-i`` reaches the flat parser, which rejects it because
    # ``-i/--input`` is required -- exactly as before the subcommand dispatch.
    result = _run_ngff_zarr("-o", "unused.ome.zarr")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "-i" in combined or "--input" in combined
    assert "Traceback" not in combined


def test_no_args_produces_sensible_output():
    """``ngff-zarr`` with no args yields the flat parser's usage, not a crash."""
    result = _run_ngff_zarr()
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "usage" in combined.lower() or "--input" in combined
    assert "Traceback" not in combined
