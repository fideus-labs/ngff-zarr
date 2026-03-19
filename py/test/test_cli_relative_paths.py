# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test that CLI correctly handles relative paths on all platforms."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr

from ._data import test_data_dir

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

ome_zarr_versions = ["0.4"] + (["0.5"] if zarr_version_major >= 3 else [])


def _run_ngff_zarr(*args, cwd=None):
    """Run ngff-zarr CLI as a subprocess and return the result."""
    cmd = [
        sys.executable,
        "-c",
        "from ngff_zarr.cli import main; main()",
    ] + list(args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


class TestRelativePathHandling:
    @pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
    def test_relative_input_path(self, input_images, ome_zarr_version):  # noqa: ARG002
        """Test that relative input paths are resolved correctly."""
        # Create a temp directory to use as working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Copy test image to temp dir
            input_file = test_data_dir / "input" / "cthead1.png"
            test_input = tmpdir_path / "test_input.png"
            test_input.write_bytes(input_file.read_bytes())

            output_path = tmpdir_path / "output.zarr"

            # Run with relative input path from tmpdir
            result = _run_ngff_zarr(
                "-i",
                "test_input.png",
                "-o",
                str(output_path),
                "--ome-zarr-version",
                ome_zarr_version,
                cwd=tmpdir,
            )

            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert output_path.exists(), "Output should exist at expected location"

    @pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
    def test_relative_output_path(self, input_images, ome_zarr_version):  # noqa: ARG002
        """Test that relative output paths are resolved correctly.

        This reproduces the issue where using a relative output path
        on Windows would create the file in an unexpected location.
        """
        # Create a temp directory to use as working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Use an absolute input path
            input_file = test_data_dir / "input" / "cthead1.png"

            # But use a relative output path
            result = _run_ngff_zarr(
                "-i",
                str(input_file),
                "-o",
                "output.zarr",
                "--ome-zarr-version",
                ome_zarr_version,
                cwd=tmpdir,
            )

            assert result.returncode == 0, f"stderr: {result.stderr}"

            # The output should be created in tmpdir, not elsewhere
            expected_output = tmpdir_path / "output.zarr"
            assert expected_output.exists(), (
                f"Output should exist at {expected_output} (cwd was {tmpdir})"
            )

    @pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
    def test_relative_output_ome_zarr(self, input_images, ome_zarr_version):  # noqa: ARG002
        """Test that relative .ome.zarr output paths are resolved correctly.

        This specifically tests the issue described in the bug report
        where relative output paths were not created in the expected location.
        """
        # Create a temp directory to use as working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Use an absolute input path
            input_file = test_data_dir / "input" / "cthead1.png"

            # But use a relative output path
            result = _run_ngff_zarr(
                "-i",
                str(input_file),
                "-o",
                "output.ome.zarr",
                "--ome-zarr-version",
                ome_zarr_version,
                cwd=tmpdir,
            )

            assert result.returncode == 0, f"stderr: {result.stderr}"

            # The output should be created in tmpdir, not elsewhere
            expected_output = tmpdir_path / "output.ome.zarr"
            assert expected_output.exists(), (
                f"Output should exist at {expected_output} (cwd was {tmpdir})"
            )
            assert expected_output.is_dir(), "Output should be a directory"

    @pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
    def test_both_relative_paths(self, input_images, ome_zarr_version):  # noqa: ARG002
        """Test that both relative input and output paths work together."""
        # Create a temp directory to use as working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Copy test image to temp dir
            input_file = test_data_dir / "input" / "cthead1.png"
            test_input = tmpdir_path / "test_input.png"
            test_input.write_bytes(input_file.read_bytes())

            # Use both relative paths
            result = _run_ngff_zarr(
                "-i",
                "test_input.png",
                "-o",
                "output.zarr",
                "--ome-zarr-version",
                ome_zarr_version,
                cwd=tmpdir,
            )

            assert result.returncode == 0, f"stderr: {result.stderr}"

            # Both should be resolved relative to tmpdir
            expected_output = tmpdir_path / "output.zarr"
            assert expected_output.exists(), (
                f"Output should exist at {expected_output} (cwd was {tmpdir})"
            )

    @pytest.mark.skipif(
        zarr_version_major < 3, reason="RFC-9 (.ozx) requires zarr-python >= 3.0.0"
    )
    def test_relative_output_ozx(self):
        """Test that relative .ozx output paths are resolved to the working directory.

        This specifically tests the bug reported where relative output paths with
        .ozx extension were not created in the expected working directory location.
        """
        from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a minimal OME-Zarr store directly as input
            input_store = tmpdir_path / "input.ome.zarr"
            data = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
            image = to_ngff_image(data=data, dims=["y", "x"])
            multiscales = to_multiscales(image)
            to_ngff_zarr(str(input_store), multiscales, version="0.5")

            # Run CLI with absolute input but relative output from tmpdir
            result = _run_ngff_zarr(
                "-i", str(input_store), "-o", "output.ozx", cwd=tmpdir
            )

            assert result.returncode == 0, f"stderr: {result.stderr}"

            # The .ozx file should be created in tmpdir, not elsewhere
            expected_output = tmpdir_path / "output.ozx"
            assert expected_output.exists(), (
                f"Output should exist at {expected_output} (cwd was {tmpdir})"
            )
