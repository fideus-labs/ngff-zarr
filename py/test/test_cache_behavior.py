#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Test script to verify HCS cache behavior.
"""

from ngff_zarr.hcs import LRUCache, from_hcs_zarr
from ngff_zarr.config import config


def test_lru_cache():
    """Test the LRU cache implementation."""
    # Test with small cache size
    cache = LRUCache(max_size=3)

    # Add items
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3

    # All should be present
    assert "a" in cache
    assert "b" in cache
    assert "c" in cache
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert cache["c"] == 3

    # Add another item - should evict "a" (least recently used)
    cache["d"] = 4

    # "a" should be evicted, others should remain
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache
    assert "d" in cache

    # Access "b" to make it most recently used
    _ = cache["b"]

    # Add another item - should evict "c"
    cache["e"] = 5

    assert "b" in cache  # recently accessed
    assert "c" not in cache  # evicted
    assert "d" in cache
    assert "e" in cache

    print("✓ LRU cache working correctly")


def test_cache_size_config():
    """Test that cache sizes can be configured."""
    # Test default values
    assert hasattr(config, "hcs_image_cache_size")
    assert hasattr(config, "hcs_well_cache_size")
    assert config.hcs_image_cache_size == 100
    assert config.hcs_well_cache_size == 500

    print("✓ Cache size configuration available")


def test_hcs_with_cache_params():
    """Test that HCS functions accept cache parameters."""
    # Try to test with existing test data
    try:
        from test._data import test_data_dir

        hcs_path = test_data_dir / "plate.zarr"

        if hcs_path.exists():
            # Test with custom cache sizes
            plate = from_hcs_zarr(
                str(hcs_path), well_cache_size=50, image_cache_size=25
            )

            # Check that cache sizes are passed through
            assert plate._wells.max_size == 50

            # Get a well and check its cache size
            first_well = None
            for row in plate.rows:
                for col in plate.columns:
                    well = plate.get_well(row.name, col.name)
                    if well is not None:
                        first_well = well
                        break
                if first_well:
                    break

            if first_well:
                assert first_well._images.max_size == 25
                print("✓ Custom cache sizes applied correctly")
            else:
                print("⚠ No wells found in test data, cache size test skipped")
        else:
            print("⚠ HCS test data not found, cache behavior test skipped")
    except ImportError:
        print("⚠ Test data module not available, cache behavior test skipped")
