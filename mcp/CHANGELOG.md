<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# Changelog

All notable changes to the ngff-zarr-mcp package will be documented in this
file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-07-23

### Removed

- **Memory Target Option**: Removed `memory_target` parameter from MCP interface
  - Removed from `ConversionOptions` model
  - Removed from `convert_images_to_ome_zarr` tool function
  - Removed from `setup_dask_config` utility function
  - Memory management now handled automatically by ngff-zarr library defaults

### Changed

- **Simplified Memory Management**: Memory usage is now managed automatically
  - Removed manual memory limit calculations for Dask LocalCluster
  - Simplified caching logic to rely on system defaults
  - LocalCluster now uses default memory management without explicit limits

### Technical Details

- **Breaking Change**: Applications using `memory_target` parameter will need to
  be updated
- Memory management delegated to underlying ngff-zarr library configuration
- Improved code maintainability by removing manual memory management complexity
- All existing tests updated and passing without memory_target references

## [0.4.0] - 2025-07-22

### Added

- **Store Input Support**: Enhanced `convert_to_ome_zarr` tool to accept
  existing OME-Zarr stores as input
  - Enables conversion and optimization of existing OME-Zarr datasets
  - Supports both local and remote store inputs with storage options
- **Enhanced Metadata Population**: Improved metadata extraction and analysis
  - Populate `method_type` and `method_metadata` fields in store analysis
  - Better detection of multiscale generation methods
- **TensorStore Integration**: Added comprehensive TensorStore support
  - Added `tensorstore` dependency for enhanced performance
  - Compression testing for TensorStore-based operations

### Changed

- **Dependency Updates**: Bumped ngff-zarr dependency to 0.15.2
  - Compatibility with latest ngff-zarr features and improvements
  - Enhanced support for RFC 4 and advanced storage options
- **Improved Documentation**: Enhanced README and examples
  - Updated usage patterns for new store input functionality

### Fixed

- **DANDI OMERO Compatibility**: Added workaround for DANDI OMERO dataset
  compatibility issues
  - Improved handling of legacy OMERO metadata structures
  - Better error handling for malformed metadata
- **Module Import Issues**: Fixed zarr module shadowing in utils.py
  - Resolved import conflicts that could cause runtime errors
  - Improved code clarity and maintainability
- **Dependency Resolution**: Added missing remote storage dependencies
  - Ensures proper functionality with cloud storage backends

### Technical Details

- Compatible with ngff-zarr Python package versions 0.15.0 and 0.15.2
- Enhanced test coverage with TensorStore compression testing
- Improved error handling and logging throughout the codebase
- Better support for legacy OME-Zarr formats and OMERO datasets

## [0.3.0] - 2025-01-18

### Added

- **RFC 4 - Anatomical Orientation Support**: Added parameters for medical
  imaging orientation systems
  - `anatomical_orientation` parameter for LPS/RAS coordinate system presets
  - `enable_rfc4` parameter to enable RFC 4 support
  - `enabled_rfcs` parameter to enable specific RFC features
- **Enhanced Storage Support**: Added `storage_options` parameter for cloud
  storage authentication
  - Support for AWS S3, Google Cloud Storage, Azure Blob Storage
  - Authentication and configuration options for remote storage
- **New Tool**: `read_ome_zarr_store` for reading OME-Zarr data with remote
  storage support
- **Enhanced Metadata**: Extended StoreInfo model with new fields:
  - `method_type`: Type of multiscale method used
  - `method_metadata`: Detailed method information
  - `anatomical_orientation`: Anatomical orientation information
  - `rfc_support`: List of enabled RFC features

### Changed

- Updated `convert_images_to_ome_zarr` tool with new parameters for RFC 4 and
  storage options
- Enhanced documentation in README with new features
- Improved type annotations and error handling

### Technical Details

- Compatible with ngff-zarr Python package versions 0.14.0 and 0.15.0
- Maintains backward compatibility with existing MCP tool calls
- Prepared infrastructure for future RFC 4 and method metadata integration
- Added graceful fallbacks for features not yet available in the core library

## [0.2.1] - Previous Release

- Previous MCP server functionality
