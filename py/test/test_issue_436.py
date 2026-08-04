# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Regression tests for issue #436.

Two distinct failure modes are covered:

1. Multi-series TIFF (every Aperio ``.svs`` whole-slide file has Baseline +
   Thumbnail + Label + Macro series) written to a single ``.ozx`` path. The
   requested ``.ozx`` must actually be created (holding the primary series),
   instead of being silently replaced by ``_<series>.ome.zarr`` directories.

2. A truncated / corrupt TIFF whose IFD chain points beyond EOF. tifffile
   logs an error and silently stops reading pages; ngff-zarr must surface a
   warning rather than producing an incomplete output without notice.
"""

import os
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import packaging.version
import pytest
import zarr

tifffile = pytest.importorskip("tifffile")

zarr_version_major = packaging.version.parse(zarr.__version__).major


def _run_ngff_zarr(*args):
    """Run the ngff-zarr CLI as a subprocess."""
    cmd = [sys.executable, "-c", "from ngff_zarr.cli import main; main()"] + list(args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _write_multiseries_tiff(path):
    """Write a 2-series pyramidal RGB OME-TIFF (Baseline pyramid + Thumbnail)."""
    rng = np.random.default_rng(0)
    base = np.full((512, 512, 3), 240, np.uint8)
    base[:256, :256] = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    # Uncompressed so the tests do not require the optional imagecodecs codec.
    opts = {"photometric": "rgb", "tile": (128, 128), "compression": None}
    with tifffile.TiffWriter(path, ome=True, bigtiff=True) as tif:
        tif.write(base, subifds=1, metadata={"Name": "Baseline"}, **opts)
        tif.write(base[::2, ::2].copy(), subfiletype=1, **opts)
        tif.write(
            np.full((64, 64, 3), 99, np.uint8),
            metadata={"Name": "Thumbnail"},
            **opts,
        )
    return base


# --------------------------------------------------------------------------
# _series_output_target unit tests (fast, no I/O)
# --------------------------------------------------------------------------


class TestSeriesOutputTarget:
    """Unit tests for the ``_series_output_target`` path-routing helper."""

    def test_ozx_primary_series_uses_exact_path(self):
        """Test the primary ``.ozx`` series keeps the exact requested path."""
        from ngff_zarr.cli import _series_output_target

        args = SimpleNamespace(output="/tmp/slide.ozx")
        store, path = _series_output_target(args, "/tmp/slide.ozx", "Baseline", 0, 4)
        assert store == "/tmp/slide.ozx"
        assert path == "/tmp/slide.ozx"

    def test_ozx_secondary_series_uses_suffixed_ozx(self):
        """Test secondary series get a ``_<name>.ozx`` suffix (zip preserved)."""
        from ngff_zarr.cli import _series_output_target

        args = SimpleNamespace(output="/tmp/slide.ozx")
        store, path = _series_output_target(args, "/tmp/slide.ozx", "Thumbnail", 1, 4)
        # .ozx format preserved; string path is passed straight to to_ngff_zarr
        assert store == "/tmp/slide_Thumbnail.ozx"
        assert path == "/tmp/slide_Thumbnail.ozx"

    def test_single_series_uses_requested_store_unchanged(self):
        """Test a single-series input reuses the requested store unchanged."""
        from ngff_zarr.cli import _series_output_target

        args = SimpleNamespace(output="/tmp/slide.ozx")
        sentinel = object()
        store, path = _series_output_target(args, sentinel, "Baseline", 0, 1)
        assert store is sentinel
        assert path == "/tmp/slide.ozx"

    def test_directory_output_keeps_per_series_ome_zarr(self):
        """Test directory output routes each series to its own ``.ome.zarr``."""
        from ngff_zarr.cli import _series_output_target

        args = SimpleNamespace(output="/tmp/out/")
        store, path = _series_output_target(args, None, "GFP", 0, 2)
        assert path.endswith("_GFP.ome.zarr")
        assert not isinstance(store, str)

    def test_no_output_returns_none(self):
        """Test a missing output target yields no store or path."""
        from ngff_zarr.cli import _series_output_target

        args = SimpleNamespace(output=None)
        store, path = _series_output_target(args, None, "Baseline", 0, 3)
        assert store is None
        assert path is None


# --------------------------------------------------------------------------
# End-to-end CLI: multi-series TIFF -> single .ozx
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    zarr_version_major < 3,
    reason="RFC-9 .ozx and OME-Zarr 0.5 require zarr-python >= 3.0.0",
)
def test_multiseries_tiff_to_ozx_creates_requested_file():
    """Test a multi-series TIFF creates the requested ``.ozx`` with real data."""
    tmp = Path(tempfile.mkdtemp(prefix="issue436_"))
    tiff_path = tmp / "slide.svs.ome.tif"
    base = _write_multiseries_tiff(tiff_path)

    out = tmp / "out.ozx"
    result = _run_ngff_zarr(
        "--ome-zarr-version", "0.5", "-i", str(tiff_path), "-o", str(out)
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    # The exact requested .ozx must exist (was previously never created).
    assert out.exists(), (
        "Requested .ozx was not created for a multi-series TIFF; "
        f"workdir contents: {[p.name for p in tmp.iterdir()]}"
    )
    # The auxiliary series is written alongside, preserving the .ozx format.
    assert (tmp / "out_Thumbnail.ozx").exists()

    # The primary .ozx must hold the full-resolution Baseline pyramid,
    # normalized from the TIFF's YXS order to the spec axis order (c, y, x).
    store = zarr.storage.ZipStore(str(out), mode="r")
    root = zarr.open_group(store, mode="r")
    scale0 = root["scale0"]
    arr = scale0[next(iter(scale0.keys()))]
    assert arr.shape[1] == base.shape[0]
    assert arr.shape[2] == base.shape[1]
    # Real pixel data, not an all-zero / blank array.
    assert int(np.asarray(arr[:]).max()) > 0


@pytest.mark.skipif(
    zarr_version_major < 3,
    reason="CLI default OME-Zarr 0.5 requires zarr-python >= 3.0.0",
)
def test_multiseries_tiff_to_directory_keeps_per_series_stores():
    """Directory-style .ome.zarr output keeps the documented per-series layout."""
    tmp = Path(tempfile.mkdtemp(prefix="issue436dir_"))
    tiff_path = tmp / "slide.svs.ome.tif"
    _write_multiseries_tiff(tiff_path)

    out = tmp / "out.ome.zarr"
    result = _run_ngff_zarr("-i", str(tiff_path), "-o", str(out))
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    created = sorted(p.name for p in tmp.iterdir() if p.name != tiff_path.name)
    # One .ome.zarr store per series (documented behavior), not a single bundle.
    assert any(name.endswith("_Baseline.ome.zarr") for name in created), created
    assert any(name.endswith("_Thumbnail.ome.zarr") for name in created), created


# --------------------------------------------------------------------------
# Truncated / corrupt TIFF -> warning instead of silent data loss
# --------------------------------------------------------------------------


def _write_truncated_tiff(path):
    """Write a 3-page classic TIFF, then point page 0's next-IFD offset past EOF."""
    pages = [
        np.full((64, 64), 10, np.uint8),
        np.full((32, 32), 20, np.uint8),
        np.full((16, 16), 30, np.uint8),
    ]
    with tifffile.TiffWriter(path, bigtiff=False, byteorder="<") as tif:
        for p in pages:
            tif.write(p, photometric="minisblack", compression=None)

    data = bytearray(path.read_bytes())
    ifd0 = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, ifd0)[0]
    next_off_pos = ifd0 + 2 + count * 12
    struct.pack_into("<I", data, next_off_pos, len(data) + 1_000_000)
    path.write_bytes(data)
    return len(pages)


def test_truncated_tiff_emits_warning():
    """Test a truncated TIFF surfaces a RuntimeWarning, not silent data loss."""
    from ngff_zarr import tiff_file_to_ngff_images

    tmp = Path(tempfile.mkdtemp(prefix="issue436trunc_"))
    tiff_path = tmp / "truncated.tif"
    _write_truncated_tiff(tiff_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiff_file_to_ngff_images(tiff_path)

    structural = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "tifffile reported errors" in str(w.message)
    ]
    assert structural, (
        "Expected a RuntimeWarning about truncated/corrupt TIFF; "
        f"got: {[str(w.message) for w in caught]}"
    )
    assert "truncated or corrupt" in str(structural[0].message)


def test_clean_tiff_emits_no_structural_warning():
    """Test a clean TIFF emits no spurious structural-error warning."""
    from ngff_zarr import tiff_file_to_ngff_images

    tmp = Path(tempfile.mkdtemp(prefix="issue436clean_"))
    tiff_path = tmp / "clean.ome.tif"
    _write_multiseries_tiff(tiff_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiff_file_to_ngff_images(tiff_path, reuse_existing_pyramids=True)

    structural = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "tifffile reported errors" in str(w.message)
    ]
    assert not structural, (
        f"Unexpected structural-error warning on a clean file: "
        f"{[str(w.message) for w in structural]}"
    )


# --------------------------------------------------------------------------
# Error-message summarization helper
# --------------------------------------------------------------------------


class TestSummarizeTiffErrorMessages:
    """Unit tests for the ``_summarize_tiff_error_messages`` helper."""

    def test_dedupes_and_joins(self):
        """Test duplicates are removed and remaining messages joined by ``; ``."""
        from ngff_zarr.tiff_to_ngff_image import _summarize_tiff_error_messages

        out = _summarize_tiff_error_messages(["a", "a", "b"])
        assert out == "a; b"

    def test_truncates_beyond_limit_with_more_note(self):
        """Test messages beyond the limit are capped with a ``... (N more)`` note."""
        from ngff_zarr.tiff_to_ngff_image import _summarize_tiff_error_messages

        messages = [f"err{i}" for i in range(8)]
        out = _summarize_tiff_error_messages(messages, limit=5)
        assert out.startswith("err0; err1; err2; err3; err4")
        assert out.endswith("... (3 more)")

    def test_empty(self):
        """Test an empty message list summarizes to an empty string."""
        from ngff_zarr.tiff_to_ngff_image import _summarize_tiff_error_messages

        assert _summarize_tiff_error_messages([]) == ""
