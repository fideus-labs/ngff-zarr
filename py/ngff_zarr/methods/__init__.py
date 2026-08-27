# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Anti-aliasing methods used to downsample images into a multiscale pyramid."""

from enum import Enum


class Methods(Enum):
    """Anti-aliasing methods for downsampling an image into a multiscale pyramid.

    To avoid aliasing artifacts when generating a multiscale representation, the
    input image is smoothed before it is downsampled. The available methods
    differ primarily in the smoothing that is applied prior to resampling, and
    therefore in their amount of artifacts, speed, hardware requirements and
    portability, and suitability for intensity or label images.

    Pass a member to {py:func}`~ngff_zarr.to_multiscales.to_multiscales`, for
    example ``to_multiscales(image, method=Methods.ITKWASM_GAUSSIAN)``. See the
    {doc}`Methods guide </methods>` for installation requirements and
    GPU-accelerated variants of these methods.
    """

    ITKWASM_GAUSSIAN = "itkwasm_gaussian"
    """Smoothed with a discrete Gaussian filter to generate a scale space, ideal
    for intensity images. The ITK-Wasm implementation is extremely portable and
    SIMD accelerated. This is the default method."""

    ITKWASM_BIN_SHRINK = "itkwasm_bin_shrink"
    """Uses the local mean for the output value. WebAssembly build. Fast but
    generates more artifacts than Gaussian-based methods. Appropriate for
    intensity images."""

    ITKWASM_LABEL_IMAGE = "itkwasm_label_image"
    """A sample is the mode of the linearly weighted local labels in the image.
    Fast with minimal artifacts. For label images."""

    ITK_GAUSSIAN = "itk_gaussian"
    """Similar to ITKWASM_GAUSSIAN, but built to native binaries. Smoothed with a
    discrete Gaussian filter to generate a scale space, ideal for intensity
    images."""

    ITK_BIN_SHRINK = "itk_bin_shrink"
    """Uses the local mean for the output value. Native binary build. Fast but
    generates more artifacts than Gaussian-based methods. Appropriate for
    intensity images."""

    # ITK_LABEL_GAUSSIAN = "itk_label_gaussian"

    DASK_IMAGE_GAUSSIAN = "dask_image_gaussian"
    """Smoothed with a discrete Gaussian filter to generate a scale space, ideal
    for intensity images. dask-image implementation based on SciPy."""

    DASK_IMAGE_MODE = "dask_image_mode"
    """Local mode for label images. Fewer artifacts than simple nearest neighbor
    interpolation, but slower."""

    DASK_IMAGE_NEAREST = "dask_image_nearest"
    """Nearest neighbor for label images. Will have many artifacts for
    high-frequency content and/or multiple scales."""

    DASK_BIN_SHRINK = "dask_bin_shrink"
    """Uses the local mean for the output value, computed with dask.array.coarsen.
    Same samples as the ITK bin-shrink methods, no native or WebAssembly
    dependency and no limit on block size. Appropriate for intensity images."""


# Backwards-compatible (name, value) pairs and value strings, derived from the
# Methods enum. ``methods_values`` is used for CLI ``--method`` argument choices.
methods = [(member.name, member.value) for member in Methods]
"""Backwards-compatible ``(name, value)`` pairs for each ``Methods`` member."""

methods_values = [member.value for member in Methods]
"""String values of the ``Methods`` members, e.g. the CLI ``--method`` choices."""
