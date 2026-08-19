<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ngff-zarr

[![PyPI - Version](https://img.shields.io/pypi/v/ngff-zarr.svg)](https://pypi.org/project/ngff-zarr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ngff-zarr.svg)](https://pypi.org/project/ngff-zarr)
[![Python CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/python.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/python.yml)
[![npm - Version](https://img.shields.io/npm/v/@fideus-labs/ngff-zarr.svg)](https://www.npmjs.com/package/@fideus-labs/ngff-zarr)
[![Typescript CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/typescript.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/typescript.yml)
[![MCP CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/mcp-ci.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/mcp-ci.yml)
[![DOI](https://zenodo.org/badge/541840158.svg)](https://zenodo.org/badge/latestdoi/541840158)
[![Documentation Status](https://readthedocs.org/projects/ngff-zarr/badge/?version=latest)](https://ngff-zarr.readthedocs.io/en/latest/?badge=latest)
[![Give it a star!](https://img.shields.io/github/stars/fideus-labs/ngff-zarr?style=social)](https://github.com/fideus-labs/ngff-zarr)

A lean and kind
[Open Microscopy Environment (OME) Next Generation File Format (NGFF) Zarr implementation](https://ngff.openmicroscopy.org/latest/),
[OME-Zarr](https://link.springer.com/article/10.1007/s00418-023-02209-1).

## ✨ Features

- Minimal dependencies
- Work with arbitrary Zarr store types
- Lazy, parallel, and web ready -- no local filesystem required
- Process extremely large datasets
- Conversion of most bioimaging file formats
- Multiple downscaling methods
- Supports Python>=3.11
- Reads OME-Zarr v0.1 to v0.6 into simple Python data classes with Dask arrays
- Optional OME-Zarr data model validation during reading
- Writes OME-Zarr v0.4 to v0.6
- v0.6 adds RFC-5 coordinate systems and transformations
- [Sharded Zarr] stores
- Writing backed by [zarrista] (Python) or zarrita (TypeScript); outputs stay
  readable by zarr-python 2 and 3
- [Anatomical orientation metadata](./rfc4.md) (RFC-4)
- [Coordinate systems and transformations](./rfc5.md) (RFC-5)
- **OME-Zarr Zip (.ozx) file support** for single-file OME-Zarr datasets (RFC-9)
- **High Content Screening (HCS) support** for plate and well data
- **TIFF/OME-TIFF support** with automatic metadata extraction and multi-series conversion
- **Leica Image Format (LIF) support** for Leica microscopy data
- **Model Context Protocol (MCP) server** for AI agent integration

```{toctree}
:maxdepth: 2

quick_start.md
installation.md
python.md
typescript.md
cli.md
mcp.md
rfc4.md
rfc5.md
hcs.md
spec_features.md
itk.md
methods.md
faq.md
development.md
```

```{toctree}
:maxdepth: 3
:caption: 📖 Reference

apidocs/index.rst
```

[Sharded Zarr]: https://zarr.dev/zeps/accepted/ZEP0002.html
[zarrista]: https://github.com/developmentseed/zarrista
