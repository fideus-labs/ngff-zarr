# Changelog

All notable changes to ngff-zarr will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.22.0] - 2026-02-02

### Added

- Initial changelog setup with Commitizen

## py-v0.23.0 (2026-02-03)

### Bug Fixes

- **py**: handle unknown axis fields in Zarr metadata gracefully
- **ts**: bump @zarrita/storage for ozx sharded store support
- **ts**: add many channel to_multiscales support
- **py**: better downsampling with many-channel inputs
- **ts**: write omero metadata inside ome namespace for OME-Zarr 0.5
- **py**: fix cwd for pixi build-docs command
- **py**: omero metadata with OME-Zarr version 0.5
- **cli**: only process spatial dims with IMAGEIO backend
- update dask kwargs
- improve error message for RFC-9 zipped OME-Zarr version check
- prevent wasm imread error on windows
- **py**: Remove debug print statements from _itkwasm.py
- **ts**: detect ome property with version 0.5 data
- Always use written data for next pyramid level
- Always use written data for next pyramid level
- **py**: remove duplicate .scale_factors check in to_ngff_zarr
- **deps**: pin dask to <2025.11.0
- **ts**: add back full component type support
- **ts**: time and channel handling
- **ts**: fix vector order indexing and bin-shrink isotroic scaling
- incremental downsampling to achieve exact 1x, 2x, 3x, 4x sizes
- **ts**: fix chunk size downsample rounding
- **ts**: itk ngff image shape corrections
- **ts**: add missing translation offset
- **ts**: use a crop radius of zeros
- **ts**: compute sigma correctly for gaussian filtering
- **ts**: support all types, shape order in itkImageToZarr
- **ts**: write images to `scale${index}`
- **hcs**: better 0.5 metadata support
- **cli**: default to OME-Zarr 0.5 for .ozx
- **ts**: more direct typing for publish to JSR
- **ts**: export explicit metadata types
- **py**: improve sharding chunk shape
- **py**: improved chunk sizes for dask/zarr
- **ts**: bump downsample dep for Windows path support
- **py**: multi-channel / component gaussian downsampling
- **py**: update testing data hashes for nifti image input
- **nibabel**: add 'c' scale_dict translation_dict values
- **nibabel**: remove zero shear criteria to add anatomical orientation
- **cli**: add check that input and output dirs are not the same
- **hcs**: address writing to sparse plate layouts
- **hcs.py**: use zarr format 2 with NGFF version 0.4
- **hcs.py**: add missing to_ngff_zarr import
- **ts**: use AnatomicalOrientation in OrientationSchema
- **ts**: use anatomical value enum
- **ts**: use zod anatomical literal for RFC-4
- **verify_against_baseline.ts**: remove duplicate code
- **verify_against_baseline.ts**: improve path normalization
- **verify_against_baseline.ts**: use deno's path utilities
- **rfc4**: correct LPS RAS direction
- **to_ngff_zarr.ts**: remove unused targetTypedArrayConstructor
- **py**: lps uses superior-to-inferior
- **mcp**: remove memory_target option
- **py**: sharding, compression codec, tensorstore
- **py**: use _handle_large_array_writing without scale_factors
- **py**: support compressor kwargs with _write_with_tensorstore
- **mcp**: add ngff-zarr tensorstore dep
- **mcp**: fix shadowing of zarr module in utils.py
- **mcp**: workaround dandi omero compatibility
- **from_ngff_zarr**: handle invalid OMERO metadata
- **to_ngff_zarr**: throw with unknown dtype
- **to_ngff_zarr**: use null to write full array
- **rfc.py**: add missing values
- **ts**: Windows FileSystemStore path normalization
- **ts**: add zod schema types
- **ts**: remove unused playwright.config.js
- **ts**: remove unused dask_array.ts
- **ts**: remove invalid methods from MethodsSchema
- **ts**: fix deno task build
- **ts**: remove ITK_* and DASK_IMAGE_* methods

### Features

- **ci**: add automated GitHub Release workflow with Commitizen changelog integration
- **py,ts**: add OMERO metadata computation from NgffImage
- add input validation for OMERO metadata computation
- **ts**: add RFC-9 (.ozx) zipped OME-Zarr support
- **py**: jsonFirst flag in the ZIP comment for RFC-9 OZX files
- add Leica Image Format (LIF) support
- **ts**: add rfc4 support
- **ts**: add RFC 4 validation
- add support for dask>=2025.12.0
- **hcs**: add helpers and docs for writing .ozx
- **ts**: add path option to ItkImageToNgffImageOptions
- **py**: implement RFC-9 Zip file support
- **ts**: add support for downsampling more types
- **ts**: add downsampleItkWasm method
- **ts**: initial toMultiscales
- **nibabel**: extract omero metadata from nibabel
- **nibabel**: extract orientation from nifti header
- **nibabel**: better pixel data typing
- **py**: nibabel io backend
- add write_hcs_well_image
- **ts**: add basic hcs support
- **ts**: add zod schemas based on spec json schemas
- **ts**: add RFC-4 zod schema support
- **ts**: add ngffImageToItkImage
- **ts**: itk_image_to_ngff_image
- **py**: add hcs caching support
- **py**: add high content screening support
- **py**: add RFC-4 validation
- **ts**: write array data
- **ts**: add BigInt64Array, BigUint64Array to TypedArray
- **mcp**: populate method_type, method_metadata
- **mcp**: convert_to_ome_zarr support for store inputs
- **mcp**: updates for ngff-zarr 0.14, 0.15 features
- **ts**: initial toNgffZarr implementation
- **fromNgffZarr**: support store type inputs
- add method metadata support
- **py**: add method type metadata
- provide RAS, LPS convenience mappings
- **ts**: bump deno and zarrita
- **from_ngff_zarr**: add support for storage_options
- **from_ngff_zarr**: add support for storage_options

### Performance

- **ts**: use reservoir sampling for memory-efficient quantile computation
- **py**: avoid loading data into memory for omero computation

### Refactoring

- move re imports to module level for better performance
- **ts**: switch to glasbey_hv for omero colors
- deduplicate dask.array.to_zarr
- **ts**: metadata extract per Python PR #250
- refactor zarr_kwargs checks prior to da.array.to_zarr
- add metadata reading into class methods
- version converters for metadata classes
- **ts**: factor out calculateIncrementalFactor
- remove LazyArray
- **ts**: use zarr.Array for NgffImage.data
- **ts**: fromNgffZarr simplifications
- use descriptive CI workflow filenames
- **ts**: simplify toNgffZarr, fromNgffZarr
- **ts**: rename ZarrReader to OMEZarrReader
- **ts**: use @std/cli package
- **ts**: rename DaskArray to LazyArray
- **ts**: update package org/name to @fideus-labs/ngff-zarr
