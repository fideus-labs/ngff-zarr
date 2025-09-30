#!/usr/bin/env python3
"""
Simple integration test for NIBABEL backend
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test imports
    import nibabel as nib
    print("✓ nibabel imported successfully")
    
    from ngff_zarr.detect_cli_io_backend import ConversionBackend, detect_cli_io_backend
    print("✓ ConversionBackend imported successfully")
    
    from ngff_zarr.nibabel_image_to_ngff_image import nibabel_image_to_ngff_image
    print("✓ nibabel_image_to_ngff_image imported successfully")
    
    from ngff_zarr.cli_input_to_ngff_image import cli_input_to_ngff_image
    print("✓ cli_input_to_ngff_image imported successfully")
    
    # Test backend detection
    backend_nii_gz = detect_cli_io_backend(['test.nii.gz'])
    backend_nii = detect_cli_io_backend(['test.nii'])
    print(f"✓ Backend detection: .nii.gz -> {backend_nii_gz}, .nii -> {backend_nii}")
    
    assert backend_nii_gz == ConversionBackend.NIBABEL, f"Expected NIBABEL, got {backend_nii_gz}"
    assert backend_nii == ConversionBackend.NIBABEL, f"Expected NIBABEL, got {backend_nii}"
    print("✓ Backend detection tests passed")
    
    # Test with real file
    test_file = Path('test/data/input/mri_denoised.nii.gz')
    if test_file.exists():
        # Load with nibabel
        img = nib.load(str(test_file))
        print(f"✓ Loaded test file: shape {img.shape}")
        
        # Convert with our function
        ngff_img = nibabel_image_to_ngff_image(img)
        print(f"✓ Converted to NgffImage: dims {ngff_img.dims}, shape {ngff_img.data.shape}")
        
        # Test CLI workflow
        input_files = [str(test_file)]
        detected_backend = detect_cli_io_backend(input_files)
        cli_ngff_img = cli_input_to_ngff_image(detected_backend, input_files)
        print(f"✓ CLI workflow: backend {detected_backend}, dims {cli_ngff_img.dims}")
        
        # Verify expectations
        import numpy as np
        assert isinstance(ngff_img.data, np.ndarray), "Data should be numpy array"
        assert ngff_img.axes_orientations is None, "Should not have anatomical orientations for non-identity transform"
        assert ngff_img.dims == ('x', 'y', 'z'), f"Expected ('x', 'y', 'z'), got {ngff_img.dims}"
        print("✓ All assertions passed")
    else:
        print(f"⚠ Test file {test_file} not found, skipping file-based tests")
    
    print("\n🎉 All tests passed! NIBABEL backend integration successful.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)