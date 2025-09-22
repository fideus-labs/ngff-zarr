<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🔖 Methods

To avoid [aliasing artifacts] when generating a multiscale representation, it is
crucial to smooth the input image before downsampling. Different downsampling
methods are available, primarily differing in the smoothing method that is
applied prior to resampling. There can also be differences due to the
interpolator used to resample, but these tend to be negligible since we are
downsampling.

These methods vary in their:

- Amount of artifacts
- Speed
- Hardware requirements and portability
- Appropriateness for intensity or label images

The `ngff_zarr.to_multiscales` function accepts the following methods,
enumerated in `ngff_zarr.Methods`.

## `ITKWASM_GAUSSIAN`

Smoothed with a discrete gaussian filter to generate a [scale space], ideal for
intensity images.

[ITK-Wasm] implementation is extremely portable. [SIMD] accelerated.

The default method.

To use an NVIDIA CUDA GPU-accelerated version, install the
`itkwasm-downsample-cucim` package:

[Install cuCIM](https://github.com/rapidsai/cucim?tab=readme-ov-file#install-cucim),
then:

```sh
pip install itkwasm-downsample-cucim
```

And GPU-accelerated filtering is applied by default after installation.

## `ITKWASM_BIN_SHRINK`

Uses the [local mean] for the output value. [WebAssembly] build.

Fast but generates more artifacts than gaussian-based methods.

Appropriate for intensity images.

An NVIDIA CUDA GPU-accelerated version can be installed similar to
`ITKWASM_GAUSSIAN` above.

## `ITKWASM_LABEL_IMAGE`

A sample is the mode of the linearly weighted [local labels] in the image.

Fast and minimal artifacts. For label images.

## `ITK_GAUSSIAN`

Similar to `ITKWASM_GAUSSIAN`, but the package is built to native binaries.

Install required dependencies with:

```sh
pip install "ngff-zarr[itk]"
```

To use a GPU-accelerated version, install the `itk-vkfft` package:

```sh
pip install itk-vkfft
```

And GPU-accelerated, FFT-based filtering is applied by default after
installation.

## `ITK_BIN_SHRINK`

Uses the [local mean] for the output value. Native binary build.

Fast but generates more artifacts than gaussian-based methods.

Appropriate for intensity images.

Install required dependencies with:

```sh
pip install "ngff-zarr[itk]"
```

## `DASK_IMAGE_GAUSSIAN`

Smoothed with a discrete gaussian filter to generate a [scale space], ideal for
intensity images.

[dask-image] implementation based on [scipy].

Install required dependencies with:

```sh
pip install "ngff-zarr[dask-image]"
```

## `DASK_IMAGE_MODE`

Local mode for label images.

Fewer artifacts than simple nearest neighbor interpolation.

Slower.

## `DASK_IMAGE_NEAREST`

Nearest neighbor for label images.

Will have many artifacts for high-frequecy content and / or multiple scales.

Install required dependencies with:

```sh
pip install "ngff-zarr[dask-image]"
```

[aliasing artifacts]:
  https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
[dask-image]: https://image.dask.org/
[ITK-Wasm]: https://wasm.itk.orghttps://image.dask.org/
[local mean]: https://doi.org/10.54294/p39qox
[local labels]: https://doi.org/10.54294/nr6iii
[SIMD]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[scale space]: https://en.wikipedia.org/wiki/Scale_space
[scipy]: https://scipy.org/
[WebAssembly]: https://webassembly.org/
