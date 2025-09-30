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
    assert ngff_image.dims == ("x", "y", "z")
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
    
    # Check that no anatomical orientation is added due to non-identity transform
    assert ngff_image.axes_orientations is None


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


def test_nibabel_image_to_ngff_image_rotated_transform():
    """Test that anatomical orientations are NOT added for rotated transform."""
    # Create a simple 3D image with rotated transform
    data = np.random.rand(10, 10, 10).astype(np.float32)
    
    # Create rotated affine (90 degree rotation around z-axis)
    cos_90 = 0.0
    sin_90 = 1.0
    affine = np.array([
        [cos_90, -sin_90, 0.0, 0.0],
        [sin_90,  cos_90, 0.0, 0.0],
        [0.0,     0.0,    1.0, 0.0],
        [0.0,     0.0,    0.0, 1.0]
    ])
    
    # Create nibabel image
    img = nib.Nifti1Image(data, affine)
    
    ngff_image = nibabel_image_to_ngff_image(img, add_anatomical_orientation=True)
    
    # Check that anatomical orientations are NOT added for rotated transform
    assert ngff_image.axes_orientations is None


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
    assert ngff_image.dims == ("t", "x", "y", "z")
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