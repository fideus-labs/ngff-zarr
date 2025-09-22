# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Utility functions for extracting method metadata information.
"""

from typing import Optional, Dict, Any
from ..v04.zarr_metadata import MethodMetadata


def get_method_metadata(method: Any) -> Optional[MethodMetadata]:
    """
    Get metadata information for a given downsampling method.

    Args:
        method: The downsampling method enum

    Returns:
        MethodMetadata with description, method (package.function), and version
    """
    method_info = _METHOD_INFO.get(method.name)
    if not method_info:
        return None

    # Try to get the package version dynamically
    try:
        import importlib.metadata
        package_name = method_info["package"]
        # Handle package name variations
        if package_name == "dask-image":
            # Try dask-image first, then dask_image
            try:
                version = importlib.metadata.version("dask-image")
            except importlib.metadata.PackageNotFoundError:
                version = importlib.metadata.version("dask_image")
        else:
            version = importlib.metadata.version(package_name)
    except Exception:
        version = "unknown"

    return MethodMetadata(
        description=method_info["description"],
        method=method_info["method"],
        version=version
    )


# Method information mapping using method names as strings
_METHOD_INFO: Dict[str, Dict[str, str]] = {
    "ITKWASM_GAUSSIAN": {
        "description": "Smoothed with a discrete gaussian filter to generate a scale space, ideal for intensity images. ITK-Wasm implementation is extremely portable and SIMD accelerated.",
        "package": "itkwasm-downsample",
        "method": "itkwasm_downsample.downsample"
    },
    "ITKWASM_BIN_SHRINK": {
        "description": "Uses the local mean for the output value. WebAssembly build. Fast but generates more artifacts than gaussian-based methods. Appropriate for intensity images.",
        "package": "itkwasm-downsample",
        "method": "itkwasm_downsample.downsample_bin_shrink"
    },
    "ITKWASM_LABEL_IMAGE": {
        "description": "A sample is the mode of the linearly weighted local labels in the image. Fast and minimal artifacts. For label images.",
        "package": "itkwasm-downsample",
        "method": "itkwasm_downsample.downsample_label_image"
    },
    "ITK_GAUSSIAN": {
        "description": "Similar to ITKWASM_GAUSSIAN, but built to native binaries. Smoothed with a discrete gaussian filter to generate a scale space, ideal for intensity images.",
        "package": "itk",
        "method": "itk.DiscreteGaussianImageFilter"
    },
    "ITK_BIN_SHRINK": {
        "description": "Uses the local mean for the output value. Native binary build. Fast but generates more artifacts than gaussian-based methods. Appropriate for intensity images.",
        "package": "itk",
        "method": "itk.BinShrinkImageFilter"
    },
    "DASK_IMAGE_GAUSSIAN": {
        "description": "Smoothed with a discrete gaussian filter to generate a scale space, ideal for intensity images. dask-image implementation based on scipy.",
        "package": "dask-image",
        "method": "dask_image.ndfilters.gaussian_filter"
    },
    "DASK_IMAGE_MODE": {
        "description": "Local mode for label images. Fewer artifacts than simple nearest neighbor interpolation. Slower.",
        "package": "dask-image",
        "method": "dask_image.ndfilters.generic_filter"
    },
    "DASK_IMAGE_NEAREST": {
        "description": "Nearest neighbor for label images. Will have many artifacts for high-frequency content and/or multiple scales.",
        "package": "dask-image",
        "method": "dask_image.ndinterp.affine_transform"
    }
}
