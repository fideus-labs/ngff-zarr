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
- **High Content Screening (HCS)**: Complete support for plate and well data
  structures.
- **Sharded Zarr**: Support for sharded Zarr stores, allowing for scalable data
  management.
- **Format conversion**: Conversion of most bioimaging file formats to OME-Zarr.
- **Tensorstore Writing**: Optional writing via [tensorstore] for advanced use
  cases.
- **Model Context Protocol (MCP)**: Integration with the Model Context Protocol
  for AI agent interaction.

## OME-Zarr Versions

- **OME-Zarr v0.1 to v0.5**: Reads OME-Zarr versions 0.1 to 0.5 into simple
  Python data classes with Dask arrays.
- **OME-Zarr v0.4 to v0.5**: Writes OME-Zarr versions 0.4 to 0.5, including
  support for RFC 4.

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
