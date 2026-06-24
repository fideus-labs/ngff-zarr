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

---

A multi-language implementation of the
[Open Microscopy Environment (OME) Next Generation File Format (NGFF) Zarr](https://ngff.openmicroscopy.org)
specification.

## Repository Structure

This repository contains multiple packages implementing NGFF-Zarr support:

- **[`py/`](./py/)** - Python package (ngff-zarr) - A lean and kind NGFF-Zarr
  implementation
- **[`ts/`](./ts/)** - TypeScript/JavaScript package (@fideus-labs/ngff-zarr) -
  Universal OME-Zarr for Deno, Node.js, and browsers
- **[`mcp/`](./mcp/)** - Model Context Protocol (MCP) server (ngff-zarr-mcp) for
  AI integration

## Python Package (`py/`)

The main Python package provides:

✨ **Features**

- Minimal dependencies
- Work with arbitrary Zarr store types
- Lazy, parallel, and web ready -- no local filesystem required
- Process extremely large datasets
- Conversion of most bioimaging file formats
- Multiple downscaling methods
- Supports Python>=3.10
- Reads OME-Zarr v0.1 to v0.5 into simple Python data classes with Dask arrays
- Optional OME-Zarr data model validation during reading
- Writes OME-Zarr v0.4 to v0.5
- [Sharded Zarr] stores
- Optional writing via zarr-python 2, zarr-python 3, [tensorstore] or zarrita (TypeScript)
- [Anatomical orientation metadata](./docs/rfc4.md) (RFC-4)
- **OME-Zarr Zip (.ozx) file support** for single-file OME-Zarr datasets (RFC-9)
- **High Content Screening (HCS) support** for plate and well data
- **Model Context Protocol (MCP) server** for AI agent integration

📖 **Documentation**

More information about command line usage, the Python API, library features, and
how to contribute can be found in
[our documentation](https://ngff-zarr.readthedocs.io/).

## TypeScript/JavaScript Package (`ts/`)

The TypeScript package provides universal OME-Zarr support for modern JavaScript environments:

✨ **Features**

- 🦕 **Deno-first** with first-class TypeScript support
- 📦 **Universal compatibility** - Works in Deno, Node.js, and browsers
- 🔍 **Type-safe** with Zod schema validation
- 🗂️ **OME-Zarr v0.4, v0.5, and v0.6** support (v0.6 adds RFC-5 coordinate systems and transformations)
- 🌐 **Web ready** - No filesystem dependencies, works with remote stores
- 🏗️ **Mirrors Python API** - Familiar interfaces for Python users
- 📚 **Lazy loading** - Efficient handling of large datasets

📦 **Installation**

```bash
# Deno
import * as ngffZarr from "jsr:@fideus-labs/ngff-zarr";

# Node.js / npm
npm install @fideus-labs/ngff-zarr

# Browser (CDN)
import * as ngffZarr from "https://esm.sh/@fideus-labs/ngff-zarr";
```

📖 **Documentation**

See the [TypeScript documentation](https://ngff-zarr.readthedocs.io/en/latest/typescript.html)
for detailed usage, API reference, and examples.

## MCP Server (`mcp/`)

The Model Context Protocol server enables AI assistants like Claude to interact
with NGFF-Zarr files. See the [MCP documentation](./mcp/README.md) for setup and
usage instructions.

## See also

- [ome-zarr-py](https://github.com/ome/ome-zarr-py)
- [multiscale-spatial-image](https://github.com/spatial-image/multiscale-spatial-image)
- [itk-ioomezarrngff](https://github.com/InsightSoftwareConsortium/ITKIOOMEZarrNGFF)
- [iohub](https://czbiohub-sf.github.io/iohub/)
- [pydantic-ome-ngff](https://janeliascicomp.github.io/pydantic-ome-ngff/)
- [aicsimageio](https://allencellmodeling.github.io/aicsimageio/)
- [bfio](https://bfio.readthedocs.io/)
- [zod-ome-ngff](https://github.com/hms-dbmi/zod-ome-ngff)

## License

`ngff-zarr` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.

[Sharded Zarr]: https://zarr.dev/zeps/accepted/ZEP0002.html
[tensorstore]: https://google.github.io/tensorstore/
