<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ✨ Specification Features

This page describes the features of the OME-Zarr specification that are
supported by `ngff-zarr`.

## Features Overview

- **Multiscales**: Support for multiscale representations of images.
- **Multiscales generation**: Ability to generate multiscale representations
  from single-scale images.
- **Chunking**: Customizable chunking strategies for efficient data access.
- **Compression**: Support for various compression algorithms to reduce storage
  requirements.
- **Metadata**: Rich metadata support, including spatial metadata.
- **Anatomical Orientation**: Support for anatomical orientation metadata
  (RFC-4).
- **Coordinate Systems and Transformations**: Support for RFC-5 coordinate
  systems and transformations in OME-Zarr v0.6.
- **High Content Screening (HCS)**: Complete support for plate and well data
  structures.
- **Sharded Zarr**: Support for sharded Zarr stores, allowing for scalable data
  management.
- **Format conversion**: Conversion of most bioimaging file formats to OME-Zarr.
- **Zarrista-backed I/O**: Zarr reading and writing via
  [zarrista](https://github.com/developmentseed/zarrista), a Python Zarr
  implementation built on the Rust [zarrs](https://zarrs.dev/) library;
  outputs remain readable by zarr-python 2 and 3 and other OME-Zarr
  implementations.
- **Model Context Protocol (MCP)**: Integration with the Model Context Protocol
  for AI agent interaction.

## OME-Zarr Versions

- **OME-Zarr v0.1 to v0.6**: Reads OME-Zarr versions 0.1 to 0.6 into simple
  Python data classes with Dask arrays.
- **OME-Zarr v0.4 to v0.6**: Writes OME-Zarr versions 0.4 to 0.6, including
  RFC-4 anatomical orientation. v0.6 additionally adds RFC-5 coordinate systems
  and transformations.
- **OME-Zarr 0.9.dev1**: Reads and writes the development version that adopts
  RFC-3, which extends support for the number, names, types and order
  of axes. It is opt-in: pass `version="0.9.dev1"` explicitly, the default
  target is unchanged. The schemas of the `0.9.dev1` release are bundled, so
  `validate=True` checks a store at that version as it does at any other.
- **OME-Zarr 0.9.dev3**: The development version that adopts
  [RFC-8 collections](./rfc8.md). Collection documents are read and written
  (`from_collection_zarr` / `to_collection_zarr` and the standalone-JSON
  equivalents); the RFC-8 image (multiscale node) model is not implemented
  yet, so images are written at `0.9.dev1` and referenced from `0.9.dev3`
  collections.

## High Content Screening (HCS)

Complete implementation of the HCS specification defined in OME-Zarr v0.4+:

- **Plate Metadata**: Support for plate-level metadata including rows, columns,
  wells, and acquisitions
- **Well Structure**: Hierarchical well organization with multiple fields per
  well
- **Multi-field Imaging**: Support for multiple fields of view within each well
- **Time Series**: Support for acquisition metadata and time series data
- **Validation**: HCS-specific metadata validation using the appropriate JSON
  schema

See the [HCS documentation](./hcs.md) for detailed usage examples and
implementation details.

## RFCs Supported

- **RFC-1**: We support and contribute to the NGFF specification via the
  "Request for Comments" (RFC) process described in
  [RFC-1](https://ngff.openmicroscopy.org/rfc/1/index.html).
- **RFC-2**: Support for Zarr v3,
  including[Sharded Zarr](https://zarr.dev/zeps/accepted/ZEP0002.html) stores,
  allowing for scalable data management.
- **RFC-4**: [Anatomical orientation support](./rfc4.md), allowing images to
  include metadata about their anatomical orientation.
- **RFC-5**: [Coordinate systems and transformations](./rfc5.md) for OME-Zarr
  v0.6, including affine transforms and displacement/coordinate fields.
- **RFC-9**: [OME-Zarr Zip (.ozx) format support](https://ngff.openmicroscopy.org/rfc/9/index.html),
  enabling single-file distribution of OME-Zarr datasets.

## OME-Zarr Zip Format (.ozx)

[RFC-9](https://ngff.openmicroscopy.org/rfc/9/index.html) defines the OME-Zarr Zip (.ozx) format, which packages complete OME-Zarr hierarchies into single ZIP archives. Key features include:

- **Portable Distribution**: Share entire multiscale datasets as single files
- **Version Detection**: OME-Zarr version stored in ZIP file comment for automatic format detection
- **Lazy Remote Access**: Efficient HTTP byte-range requests for remote .ozx files
- **Compression**: Individual files compressed within the ZIP archive
- **Standard Format**: Based on standard ZIP file format for broad compatibility

All required and recommended RFC-9 practices are implemented in ngff-zarr.
