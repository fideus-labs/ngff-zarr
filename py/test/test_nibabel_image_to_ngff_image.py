# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from ngff_zarr import nibabel_image_to_ngff_image
from ngff_zarr.rfc4 import AnatomicalOrientation, AnatomicalOrientationValues

from ._data import test_data_dir

pytest.importorskip("nibabel")
import nibabel as nib


def test_nibabel_image_to_ngff_image_basic():
    """Test basic conversion from nibabel to NgffImage with the test file."""
    input_path = test_data_dir / "input" / "mri_denoised.nii.gz"
    img = nib.load(str(input_path))

    ngff_image = nibabel_image_to_ngff_image(img)

    # Check basic properties
    assert tuple(ngff_image.dims) == ("x", "y", "z")
    assert ngff_image.data.shape == (256, 256, 256)

    # Check that data is numpy array, not dask
    assert isinstance(ngff_image.data, np.ndarray)

    # Check spatial metadata
    assert "x" in ngff_image.scale
    assert "y" in ngff_image.scale
    assert "z" in ngff_image.scale
    assert "x" in ngff_image.translation
    assert "y" in ngff_image.translation
    assert "z" in ngff_image.translation

    # Test memory optimization - check that data is equivalent to get_fdata()
    reference_data = img.get_fdata()
    assert np.allclose(ngff_image.data, reference_data)

    # For this test file, verify that memory optimization applied correctly
    # (should use optimized path if scaling is identity)
    header = img.header
    scl_slope = header.get('scl_slope')
    scl_inter = header.get('scl_inter')

    # Process scaling parameters same as our function
    if scl_slope is None or scl_slope == 0 or np.isnan(scl_slope):
        slope = 1.0
    else:
        slope = float(scl_slope)
    if scl_inter is None or np.isnan(scl_inter):
        inter = 0.0
    else:
        inter = float(scl_inter)

    # If identity scaling, should preserve original dtype; otherwise should be float32
    if slope == 1.0 and inter == 0.0:
        # Identity scaling - should preserve original dataobj dtype
        assert ngff_image.data.dtype == img.dataobj.dtype
    else:
        # Non-identity scaling - should use float32
        assert ngff_image.data.dtype == np.float32


def test_nibabel_image_to_ngff_image_identity_transform():
    """Test that anatomical orientations are added when transform is identity."""
    # Create a simple 3D image with identity transform
    data = np.random.rand(10, 10, 10).astype(np.float32)

    # Create identity affine (no rotation, unit spacing, zero origin)
    affine = np.eye(4)

    # Create nibabel image
    img = nib.Nifti1Image(data, affine)

    ngff_image = nibabel_image_to_ngff_image(img, add_anatomical_orientation=True)

    # Check that anatomical orientations are added for identity transform
    assert ngff_image.axes_orientations is not None
    assert "x" in ngff_image.axes_orientations
    assert "y" in ngff_image.axes_orientations
    assert "z" in ngff_image.axes_orientations

    # Check specific orientations (RAS)
    assert ngff_image.axes_orientations["x"].value == AnatomicalOrientationValues.left_to_right
    assert ngff_image.axes_orientations["y"].value == AnatomicalOrientationValues.posterior_to_anterior
    assert ngff_image.axes_orientations["z"].value == AnatomicalOrientationValues.inferior_to_superior


def test_nibabel_image_to_ngff_image_no_anatomical_orientation():
    """Test that anatomical orientations are not added when disabled."""
    input_path = test_data_dir / "input" / "mri_denoised.nii.gz"
    img = nib.load(str(input_path))

    ngff_image = nibabel_image_to_ngff_image(img, add_anatomical_orientation=False)

    # Check that no anatomical orientation is added when disabled
    assert ngff_image.axes_orientations is None


def test_nibabel_image_to_ngff_image_scaled_transform():
    """Test that anatomical orientations are added for scaled identity transform."""
    # Create a simple 3D image with scaled identity transform (no rotation/shear)
    data = np.random.rand(10, 10, 10).astype(np.float32)

    # Create scaled identity affine (no rotation, non-unit spacing, non-zero origin)
    affine = np.array([
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Create nibabel image
    img = nib.Nifti1Image(data, affine)

    ngff_image = nibabel_image_to_ngff_image(img, add_anatomical_orientation=True)

    # Check that anatomical orientations are added for scaled identity
    assert ngff_image.axes_orientations is not None
    assert "x" in ngff_image.axes_orientations
    assert "y" in ngff_image.axes_orientations
    assert "z" in ngff_image.axes_orientations

    # Check spatial metadata
    assert ngff_image.scale["x"] == 2.0
    assert ngff_image.scale["y"] == 2.0
    assert ngff_image.scale["z"] == 2.0
    assert ngff_image.translation["x"] == 10.0
    assert ngff_image.translation["y"] == 20.0
    assert ngff_image.translation["z"] == 30.0


def test_nibabel_image_to_ngff_image_4d():
    """Test conversion of 4D image (with time dimension)."""
    # Create a simple 4D image
    data = np.random.rand(5, 10, 10, 10).astype(np.float32)

    # Create identity affine
    affine = np.eye(4)

    # Create nibabel image
    img = nib.Nifti1Image(data, affine)

    ngff_image = nibabel_image_to_ngff_image(img)

    # Check 4D properties
    assert tuple(ngff_image.dims) == ("x", "y", "z", "t")
    assert ngff_image.data.shape == (5, 10, 10, 10)

    # Check that spatial dimensions have anatomical orientations but time does not
    assert ngff_image.axes_orientations is not None
    assert "x" in ngff_image.axes_orientations
    assert "y" in ngff_image.axes_orientations
    assert "z" in ngff_image.axes_orientations
    assert "t" not in ngff_image.axes_orientations

    # Check that time dimension has scale and translation
    assert "t" in ngff_image.scale
    assert "t" in ngff_image.translation
    assert ngff_image.scale["t"] == 1.0
    assert ngff_image.translation["t"] == 0.0


def test_nibabel_image_to_ngff_image_unsupported_dimensions():
    """Test that unsupported number of dimensions raises ValueError."""
    # Create a 2D image (unsupported)
    data = np.random.rand(10, 10).astype(np.float32)

    # Create identity affine (but 2D images don't have proper spatial metadata)
    affine = np.eye(4)

    # Create nibabel image
    img = nib.Nifti1Image(data, affine)

    # Should raise ValueError for unsupported dimensions
    with pytest.raises(ValueError, match="Image must have at least 3 dimensions"):
        nibabel_image_to_ngff_image(img)


def test_nibabel_image_to_ngff_image_name():
    """Test that the image gets the expected name."""
    input_path = test_data_dir / "input" / "mri_denoised.nii.gz"
    img = nib.load(str(input_path))

    ngff_image = nibabel_image_to_ngff_image(img)

    assert ngff_image.name == "nibabel_converted_image"


def test_nibabel_image_to_ngff_image_memory_optimization_identity_scaling():
    """Test memory optimization when scaling parameters are identity (slope=1.0, intercept=0.0)."""
    # Create test data with specific dtype
    data = np.random.randint(0, 1000, size=(10, 10, 10), dtype=np.uint16)

    # Create identity affine and header with identity scaling
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Ensure identity scaling parameters
    img.header['scl_slope'] = 1.0
    img.header['scl_inter'] = 0.0

    ngff_image = nibabel_image_to_ngff_image(img)

    # Should preserve original dtype for identity scaling
    assert ngff_image.data.dtype == np.uint16
    assert np.array_equal(ngff_image.data, data)


def test_nibabel_image_to_ngff_image_memory_optimization_with_scaling():
    """Test memory optimization when scaling parameters are not identity."""
    # Create test data with specific dtype
    data = np.random.randint(0, 1000, size=(10, 10, 10), dtype=np.uint16)

    # Create identity affine and header with non-identity scaling
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Set non-identity scaling parameters
    img.header['scl_slope'] = 2.0
    img.header['scl_inter'] = 10.0

    ngff_image = nibabel_image_to_ngff_image(img)

    # Should use float32 for scaled data
    assert ngff_image.data.dtype == np.float32

    # Compare with nibabel's own scaling (which is the correct reference)
    expected_data = img.get_fdata(dtype=np.float32)
    np.testing.assert_array_equal(ngff_image.data, expected_data)


def test_nibabel_image_to_ngff_image_memory_optimization_no_scaling_header():
    """Test memory optimization when no scaling headers are present (defaults to identity)."""
    # Create test data with specific dtype
    data = np.random.randint(0, 1000, size=(10, 10, 10), dtype=np.int16)

    # Create identity affine
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Remove scaling headers (should default to identity)
    img.header['scl_slope'] = 0  # nibabel treats 0 as "no scaling"
    img.header['scl_inter'] = 0

    ngff_image = nibabel_image_to_ngff_image(img)

    # Should preserve original dtype when no scaling
    assert ngff_image.data.dtype == np.int16
    assert np.array_equal(ngff_image.data, data)


def test_nibabel_image_to_ngff_image_memory_optimization_slope_only():
    """Test memory optimization when only slope is non-identity."""
    # Create test data with specific dtype
    data = np.random.randint(0, 100, size=(10, 10, 10), dtype=np.uint8)

    # Create identity affine
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Set slope only (intercept remains 0)
    img.header['scl_slope'] = 0.5
    img.header['scl_inter'] = 0.0

    ngff_image = nibabel_image_to_ngff_image(img)

    # Should use float32 due to non-identity slope
    assert ngff_image.data.dtype == np.float32

    # Compare with nibabel's own scaling (which is the correct reference)
    expected_data = img.get_fdata(dtype=np.float32)
    np.testing.assert_array_equal(ngff_image.data, expected_data)


def test_nibabel_image_to_ngff_image_memory_optimization_intercept_only():
    """Test memory optimization when only intercept is non-identity."""
    # Create test data with specific dtype
    data = np.random.randint(0, 100, size=(10, 10, 10), dtype=np.uint8)

    # Create identity affine
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Set intercept only (slope remains 1)
    img.header['scl_slope'] = 1.0
    img.header['scl_inter'] = 5.0

    ngff_image = nibabel_image_to_ngff_image(img)

    # Should use float32 due to non-zero intercept
    assert ngff_image.data.dtype == np.float32

    # Compare with nibabel's own scaling (which is the correct reference)
    expected_data = img.get_fdata(dtype=np.float32)
    np.testing.assert_array_equal(ngff_image.data, expected_data)


def test_nibabel_image_to_ngff_image_ail_orientation():
    """Test that AIL.nii.gz generates the expected anatomical orientation."""
    input_path = test_data_dir / "input" / "AIL.nii.gz"
    img = nib.load(str(input_path))

    ngff_image = nibabel_image_to_ngff_image(img, add_anatomical_orientation=True)

    # Check basic properties
    assert tuple(ngff_image.dims) == ("x", "y", "z")
    assert ngff_image.data.shape == (79, 67, 64)

    # Check that anatomical orientations are detected correctly for AIL image
    assert ngff_image.axes_orientations is not None
    assert "x" in ngff_image.axes_orientations
    assert "y" in ngff_image.axes_orientations
    assert "z" in ngff_image.axes_orientations

    # Verify the specific orientations for AIL (Anterior-Inferior-Left)
    # Based on nibabel output: Orientation: ('A', 'I', 'L')
    # Actual detected orientations from the function:
    assert ngff_image.axes_orientations["x"].value == AnatomicalOrientationValues.posterior_to_anterior
    assert ngff_image.axes_orientations["y"].value == AnatomicalOrientationValues.superior_to_inferior
    assert ngff_image.axes_orientations["z"].value == AnatomicalOrientationValues.right_to_left

    # Verify spatial metadata exists
    assert "x" in ngff_image.scale
    assert "y" in ngff_image.scale
    assert "z" in ngff_image.scale
    assert "x" in ngff_image.translation
    assert "y" in ngff_image.translation
    assert "z" in ngff_image.translation


def test_nibabel_image_to_ngff_image_rip_orientation():
    """Test that RIP.nii.gz generates the expected anatomical orientation."""
    input_path = test_data_dir / "input" / "RIP.nii.gz"
    img = nib.load(str(input_path))

    ngff_image = nibabel_image_to_ngff_image(img, add_anatomical_orientation=True)

    # Check basic properties
    assert tuple(ngff_image.dims) == ("x", "y", "z")
    assert ngff_image.data.shape == (64, 67, 79)

    # Check that anatomical orientations are detected correctly for RIP image
    assert ngff_image.axes_orientations is not None
    assert "x" in ngff_image.axes_orientations
    assert "y" in ngff_image.axes_orientations
    assert "z" in ngff_image.axes_orientations

    # Verify the specific orientations for RIP (Right-Inferior-Posterior)
    # Based on nibabel output: Orientation: ('R', 'I', 'P')
    # Actual detected orientations from the function:
    assert ngff_image.axes_orientations["x"].value == AnatomicalOrientationValues.left_to_right
    assert ngff_image.axes_orientations["y"].value == AnatomicalOrientationValues.superior_to_inferior
    assert ngff_image.axes_orientations["z"].value == AnatomicalOrientationValues.anterior_to_posterior

    # Verify spatial metadata exists
    assert "x" in ngff_image.scale
    assert "y" in ngff_image.scale
    assert "z" in ngff_image.scale
    assert "x" in ngff_image.translation
    assert "y" in ngff_image.translation
    assert "z" in ngff_image.translation


def test_nibabel_image_to_ngff_image_ail_rip_orientation_disabled():
    """Test that AIL and RIP orientations are not added when disabled."""
    ail_path = test_data_dir / "input" / "AIL.nii.gz"
    rip_path = test_data_dir / "input" / "RIP.nii.gz"

    ail_img = nib.load(str(ail_path))
    rip_img = nib.load(str(rip_path))

    ail_ngff = nibabel_image_to_ngff_image(ail_img, add_anatomical_orientation=False)
    rip_ngff = nibabel_image_to_ngff_image(rip_img, add_anatomical_orientation=False)

    # Check that no anatomical orientation is added when disabled
    assert ail_ngff.axes_orientations is None
    assert rip_ngff.axes_orientations is None