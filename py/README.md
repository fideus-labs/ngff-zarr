<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ngff-zarr

[![PyPI - Version](https://img.shields.io/pypi/v/ngff-zarr.svg)](https://pypi.org/project/ngff-zarr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ngff-zarr.svg)](https://pypi.org/project/ngff-zarr)
[![Python CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/python.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/python.yml)
[![DOI](https://zenodo.org/badge/541840158.svg)](https://zenodo.org/badge/latestdoi/541840158)
[![Documentation Status](https://readthedocs.org/projects/ngff-zarr/badge/?version=latest)](https://ngff-zarr.readthedocs.io/en/latest/?badge=latest)

---

A lean and kind
[Open Microscopy Environment (OME) Next Generation File Format (NGFF) Zarr](https://ngff.openmicroscopy.org)
implementation.

## ✨ Features

- Minimal dependencies
- Work with arbitrary Zarr store types
- Lazy, parallel, and web ready -- no local filesystem required
- Process extremely large datasets
- Conversion of most bioimaging file formats
- Multiple downscaling methods
- Supports Python>=3.10
- Reads OME-Zarr v0.1 to v0.6 into simple Python data classes with Dask arrays
- Optional OME-Zarr data model validation during reading
- Writes OME-Zarr v0.4 to v0.6
- v0.6 adds RFC-5 coordinate systems and transformations
- [Sharded Zarr] stores
- Optional writing via zarr-python 2, zarr-python 3, [zarrista] or zarrita (TypeScript)
- [Anatomical orientation metadata][rfc4] (RFC-4)
- **OME-Zarr Zip (.ozx) file support** for single-file OME-Zarr datasets (RFC-9)
- **High Content Screening (HCS) support** for plate and well data
- **Model Context Protocol (MCP) server** for AI agent integration

## Documentation

More information about command line usage, the Python API, library features, and
how to contribute can be found in
[our documentation](https://ngff-zarr.readthedocs.io/).

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
[zarrista]: https://github.com/developmentseed/zarrista
[rfc4]: https://ngff-zarr.readthedocs.io/en/latest/rfc4.html
