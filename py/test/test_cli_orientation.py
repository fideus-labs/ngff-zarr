# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the CLI ``--orientation`` flag (RFC 4 anatomical orientation)."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import packaging.version
import pytest
import zarr

from ._data import test_data_dir

zarr_version = packaging.version.parse(zarr.__version__)


class _FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(str(a) for a in args))


class _FakeLive:
    def __init__(self):
        self.console = _FakeConsole()


def test_parse_orientation_arg_preset():
    """A single preset name resolves to the matching coordinate-system mapping."""
    from ngff_zarr.cli import _parse_orientation_arg

    live = _FakeLive()
    result = _parse_orientation_arg(["LPS"], live)
    assert {dim: o.value.value for dim, o in result.items()} == {
        "x": "right-to-left",
        "y": "anterior-to-posterior",
        "z": "inferior-to-superior",
    }


def test_parse_orientation_arg_preset_case_insensitive():
    """Preset names are case-insensitive."""
    from ngff_zarr.cli import _parse_orientation_arg

    live = _FakeLive()
    result = _parse_orientation_arg(["ras"], live)
    assert {dim: o.value.value for dim, o in result.items()} == {
        "x": "left-to-right",
        "y": "posterior-to-anterior",
        "z": "inferior-to-superior",
    }


def test_parse_orientation_arg_pairs():
    """Ordered dim/value pairs map each axis to its orientation."""
    from ngff_zarr.cli import _parse_orientation_arg

    live = _FakeLive()
    result = _parse_orientation_arg(
        ["x", "right-to-left", "y", "anterior-to-posterior"], live
    )
    assert {dim: o.value.value for dim, o in result.items()} == {
        "x": "right-to-left",
        "y": "anterior-to-posterior",
    }


def test_parse_orientation_arg_unknown_preset_exits():
    """An unknown single value (not a preset) exits with an error."""
    from ngff_zarr.cli import _parse_orientation_arg

    live = _FakeLive()
    with pytest.raises(SystemExit):
        _parse_orientation_arg(["FOO"], live)
    assert any("orientation preset" in m for m in live.console.messages)


def test_parse_orientation_arg_odd_pairs_exits():
    """An odd number of pair arguments exits with an error."""
    from ngff_zarr.cli import _parse_orientation_arg

    live = _FakeLive()
    with pytest.raises(SystemExit):
        _parse_orientation_arg(["x", "right-to-left", "y"], live)
    assert any("dim/value pairs" in m for m in live.console.messages)


def test_parse_orientation_arg_invalid_value_exits():
    """An unsupported orientation value in a pair exits with an error."""
    from ngff_zarr.cli import _parse_orientation_arg

    live = _FakeLive()
    with pytest.raises(SystemExit):
        _parse_orientation_arg(["x", "not-a-real-orientation"], live)
    assert any("Unsupported anatomical orientation" in m for m in live.console.messages)


def _run_ngff_zarr(*args):
    """Run ngff-zarr CLI as a subprocess and return the result."""
    cmd = [
        sys.executable,
        "-c",
        "from ngff_zarr.cli import main; main()",
    ] + list(args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # Bound the run so a CLI regression cannot hang the test job indefinitely.
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


def _read_axes_metadata(store_path):
    zarr_group = zarr.open(str(store_path), mode="r")
    raw_attrs = dict(zarr_group.attrs)
    if "ome" in raw_attrs:
        multiscales_metadata = raw_attrs["ome"]["multiscales"][0]
    else:
        multiscales_metadata = raw_attrs["multiscales"][0]
    return multiscales_metadata["axes"]


@pytest.mark.skipif(
    zarr_version < packaging.version.parse("3.0.0b1"),
    reason="zarr version >= 3.0.0b1 required for OME-Zarr version >= 0.5",
)
def test_cli_orientation_preset_end_to_end(input_images):
    """``--orientation LPS`` writes anatomical orientation to the output store."""
    input_path = str(test_data_dir / "input" / "cthead1.png")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "cthead1.ome.zarr")

        result = _run_ngff_zarr(
            "-i", input_path, "-o", output_path, "--orientation", "LPS"
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        axes_metadata = _read_axes_metadata(output_path)
        # cthead1 is a 2D image (y, x); both spatial axes carry LPS orientation.
        spatial = {ax["name"]: ax for ax in axes_metadata if ax["name"] in ("x", "y")}
        assert spatial["x"]["orientation"]["value"] == "right-to-left"
        assert spatial["y"]["orientation"]["value"] == "anterior-to-posterior"


@pytest.mark.skipif(
    zarr_version < packaging.version.parse("3.0.0b1"),
    reason="zarr version >= 3.0.0b1 required for OME-Zarr version >= 0.5",
)
def test_cli_itk_input_writes_orientation_automatically(input_images):
    """ITK-backed inputs supply orientation, which is written without any flag.

    Reading ``cthead1.png`` through the ITK backend attaches anatomical
    orientation (identity direction -> LPS). With the data-driven model, that
    orientation is written to the output automatically -- no ``--orientation``
    or opt-in flag required.
    """
    input_path = str(test_data_dir / "input" / "cthead1.png")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "cthead1.ome.zarr")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        axes_metadata = _read_axes_metadata(output_path)
        spatial = {ax["name"]: ax for ax in axes_metadata if ax["name"] in ("x", "y")}
        assert spatial["x"]["orientation"]["type"] == "anatomical"
        assert spatial["y"]["orientation"]["type"] == "anatomical"
