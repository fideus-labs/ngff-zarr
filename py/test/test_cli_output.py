# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import os
import subprocess
import sys
import tempfile
from pathlib import Path


from ._data import test_data_dir


def _run_ngff_zarr(*args):
    """Run ngff-zarr CLI as a subprocess and return the result."""
    cmd = [
        sys.executable,
        "-c",
        "from ngff_zarr.cli import main; main()",
    ] + list(args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _temp_output_path(suffix):
    """Create a temp file path that does not exist yet (for output)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        return f.name


class TestTifffileOutput:
    def test_png_to_tif(self, input_images):  # noqa: ARG002
        """Test converting a PNG to TIFF via tifffile."""
        input_path = str(test_data_dir / "input" / "cthead1.png")
        output_path = _temp_output_path(".tif")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        import tifffile

        data = tifffile.imread(output_path)
        assert data.ndim >= 2
        assert data.shape[-2] > 0  # height
        assert data.shape[-1] > 0  # width

        Path(output_path).unlink(missing_ok=True)

    def test_png_to_tiff(self, input_images):  # noqa: ARG002
        """Test converting a PNG to TIFF with .tiff extension."""
        input_path = str(test_data_dir / "input" / "cthead1.png")
        output_path = _temp_output_path(".tiff")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        import tifffile

        data = tifffile.imread(output_path)
        assert data.ndim >= 2

        Path(output_path).unlink(missing_ok=True)

    def test_tif_output_readable(self, input_images):  # noqa: ARG002
        """Test that TIFF output produces a valid, non-empty file."""
        input_path = str(test_data_dir / "input" / "cthead1.png")
        output_path = _temp_output_path(".tif")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert Path(output_path).stat().st_size > 0

        import tifffile

        data = tifffile.imread(output_path)
        # cthead1.png is a 2D grayscale image
        assert data.size > 0

        Path(output_path).unlink(missing_ok=True)


class TestITKOutput:
    def test_png_to_nrrd(self, input_images):  # noqa: ARG002
        """Test converting a PNG to NRRD via ITK."""
        input_path = str(test_data_dir / "input" / "cthead1.png")
        output_path = _temp_output_path(".nrrd")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert Path(output_path).stat().st_size > 0

        Path(output_path).unlink(missing_ok=True)

    def test_png_to_mha(self, input_images):  # noqa: ARG002
        """Test converting a PNG to MHA via ITK."""
        input_path = str(test_data_dir / "input" / "cthead1.png")
        output_path = _temp_output_path(".mha")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert Path(output_path).stat().st_size > 0

        Path(output_path).unlink(missing_ok=True)


class TestNibabelOutput:
    def test_nifti_to_nii(self, input_images):  # noqa: ARG002
        """Test converting a NIfTI .nii.gz to .nii via nibabel."""
        input_path = str(test_data_dir / "input" / "mri_denoised.nii.gz")
        output_path = _temp_output_path(".nii")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert Path(output_path).stat().st_size > 0

        import nibabel as nib

        img = nib.load(output_path)
        assert img.shape == (256, 256, 256)

        Path(output_path).unlink(missing_ok=True)

    def test_nifti_to_nii_gz(self, input_images):  # noqa: ARG002
        """Test converting a NIfTI to compressed .nii.gz."""
        input_path = str(test_data_dir / "input" / "mri_denoised.nii.gz")
        output_path = _temp_output_path(".nii.gz")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert Path(output_path).stat().st_size > 0

        import nibabel as nib

        img = nib.load(output_path)
        assert img.shape == (256, 256, 256)

        Path(output_path).unlink(missing_ok=True)

    def test_nifti_preserves_spatial_metadata(self, input_images):  # noqa: ARG002
        """Test that NIfTI output preserves voxel spacings."""
        import nibabel as nib
        import numpy as np

        input_path = str(test_data_dir / "input" / "mri_denoised.nii.gz")
        output_path = _temp_output_path(".nii")

        result = _run_ngff_zarr("-i", input_path, "-o", output_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        original = nib.load(input_path)
        written = nib.load(output_path)

        # Voxel spacings (column norms of 3x3 affine) should be preserved
        orig_spacing = np.linalg.norm(original.affine[:3, :3], axis=0)
        written_spacing = np.linalg.norm(written.affine[:3, :3], axis=0)
        np.testing.assert_allclose(orig_spacing, written_spacing, atol=1e-5)

        Path(output_path).unlink(missing_ok=True)


class TestUnsupportedOutput:
    def test_unsupported_lif_output(self, input_images):  # noqa: ARG002
        """Test that .lif output is rejected with informative error."""
        input_path = str(test_data_dir / "input" / "cthead1.png")
        result = _run_ngff_zarr("-i", input_path, "-o", "output.lif")
        assert result.returncode == 1
