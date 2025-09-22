#!/usr/bin/env python3
"""Simple test to verify HCS functionality works."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ngff_zarr.hcs import from_hcs_zarr


def test_hcs():
    """Test basic HCS functionality."""
    print("Testing HCS loading...")

    # Load HCS data
    hcs_path = "test/data/input/hcs.ome.zarr"
    plate = from_hcs_zarr(hcs_path)

    print(f"Loaded HCS plate: {plate.metadata.name}")
    print(f"Rows: {len(plate.metadata.rows)}, Columns: {len(plate.metadata.columns)}")
    print(f"Wells: {len(plate.metadata.wells)}")

    # Test well access
    well = plate.get_well("A", "1")
    if well:
        print(f"Well A/1 has {len(well.images)} images")

        # Test image loading
        image = well.get_image(0)
        if image:
            print(f"Image loaded with {len(image.images)} scale levels")
            print(f"Shape: {image.images[0].data.shape}")
            print("HCS test passed!")
        else:
            print("Failed to load image")
            assert False
    else:
        print("Failed to get well")
        assert False


if __name__ == "__main__":
    success = test_hcs()
    sys.exit(0 if success else 1)
