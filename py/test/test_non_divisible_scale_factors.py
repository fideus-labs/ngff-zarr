# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test that scale factors that are not powers of 2 work correctly."""
import pytest
import numpy as np
from pathlib import Path
from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr, from_ngff_zarr


def test_non_divisible_scale_factors_2_3_4(tmp_path):
    """Test scale factors [2, 3, 4] produce correct shapes."""
    # Generate pixel data
    data = np.random.randint(0, 256, size=(120, 120), dtype=np.uint8)
    image = to_ngff_image(data, scale={'y': 1.0, 'x': 1.0})
    
    # Generate multiscales with non-divisible factors
    multiscales = to_multiscales(image, scale_factors=[2, 3, 4])
    
    # Check that multiscales are correct before writing
    assert multiscales.images[0].data.shape == (120, 120), "Original should be (120, 120)"
    assert multiscales.images[1].data.shape == (60, 60), "Scale 2 should be (60, 60)"
    assert multiscales.images[2].data.shape == (40, 40), f"Scale 3 should be (40, 40), got {multiscales.images[2].data.shape}"
    assert multiscales.images[3].data.shape == (30, 30), f"Scale 4 should be (30, 30), got {multiscales.images[3].data.shape}"
    
    # Write to zarr
    zarr_path = tmp_path / 'test.ome.zarr'
    to_ngff_zarr(str(zarr_path), multiscales)
    
    # Read back from zarr
    loaded = from_ngff_zarr(str(zarr_path))
    
    # Check that loaded shapes are correct
    assert loaded.images[0].data.shape == (120, 120), "Loaded original should be (120, 120)"
    assert loaded.images[1].data.shape == (60, 60), "Loaded scale 2 should be (60, 60)"
    assert loaded.images[2].data.shape == (40, 40), f"Loaded scale 3 should be (40, 40), got {loaded.images[2].data.shape}"
    assert loaded.images[3].data.shape == (30, 30), f"Loaded scale 4 should be (30, 30), got {loaded.images[3].data.shape}"


def test_non_divisible_scale_factors_2_5(tmp_path):
    """Test scale factors [2, 5] produce correct shapes."""
    # Generate pixel data with size divisible by both 2 and 5
    data = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    image = to_ngff_image(data, scale={'y': 1.0, 'x': 1.0})
    
    # Generate multiscales with non-divisible factors
    multiscales = to_multiscales(image, scale_factors=[2, 5])
    
    # Check that multiscales are correct before writing
    assert multiscales.images[0].data.shape == (100, 100), "Original should be (100, 100)"
    assert multiscales.images[1].data.shape == (50, 50), "Scale 2 should be (50, 50)"
    assert multiscales.images[2].data.shape == (20, 20), f"Scale 5 should be (20, 20), got {multiscales.images[2].data.shape}"
    
    # Write to zarr
    zarr_path = tmp_path / 'test2.ome.zarr'
    to_ngff_zarr(str(zarr_path), multiscales)
    
    # Read back from zarr
    loaded = from_ngff_zarr(str(zarr_path))
    
    # Check that loaded shapes are correct
    assert loaded.images[0].data.shape == (100, 100), "Loaded original should be (100, 100)"
    assert loaded.images[1].data.shape == (50, 50), "Loaded scale 2 should be (50, 50)"
    assert loaded.images[2].data.shape == (20, 20), f"Loaded scale 5 should be (20, 20), got {loaded.images[2].data.shape}"


def test_non_divisible_scale_factors_with_dict(tmp_path):
    """Test dict-based scale factors with non-divisible values."""
    # Generate pixel data
    data = np.random.randint(0, 256, size=(120, 120), dtype=np.uint8)
    image = to_ngff_image(data, scale={'y': 1.0, 'x': 1.0})
    
    # Generate multiscales with dict-based non-divisible factors
    multiscales = to_multiscales(image, scale_factors=[{'y': 2, 'x': 2}, {'y': 3, 'x': 3}])
    
    # Check that multiscales are correct before writing
    assert multiscales.images[0].data.shape == (120, 120), "Original should be (120, 120)"
    assert multiscales.images[1].data.shape == (60, 60), "Scale 2 should be (60, 60)"
    assert multiscales.images[2].data.shape == (40, 40), f"Scale 3 should be (40, 40), got {multiscales.images[2].data.shape}"
    
    # Write to zarr
    zarr_path = tmp_path / 'test3.ome.zarr'
    to_ngff_zarr(str(zarr_path), multiscales)
    
    # Read back from zarr
    loaded = from_ngff_zarr(str(zarr_path))
    
    # Check that loaded shapes are correct
    assert loaded.images[0].data.shape == (120, 120), "Loaded original should be (120, 120)"
    assert loaded.images[1].data.shape == (60, 60), "Loaded scale 2 should be (60, 60)"
    assert loaded.images[2].data.shape == (40, 40), f"Loaded scale 3 should be (40, 40), got {loaded.images[2].data.shape}"
