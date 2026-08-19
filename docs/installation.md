<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 💾 Installation

::::{tab-set}

:::{tab-item} System

```shell
pip install ngff-zarr
```

:::

:::{tab-item} Browser via Pyodide

With, for example, the [Pyodide REPL] or [JupyterLite]

```python
import micropip
await micropip.install('ngff-zarr')
```

:::

::::

## Optional dependencies

Optional extras dependencies include:

`cli` : Additional IO libraries for
[file conversion programmically or via the command line interface (CLI)](./cli.md).

`dask-image` : Support multiscale generation with [dask-image]
[methods](./methods.md).

`itk` : Support multiscale generation with [itk] [methods](./methods.md).

`remote` : Read OME-Zarr from remote stores -- http(s), S3, Google Cloud
Storage, and Azure -- via the [obstore] backend (requires Python >= 3.11).
Remote writes are not supported: write to a local directory and upload
afterwards. See
[Network Storage and Authentication](./faq.md#network-storage-and-authentication).

`validate` : Support OME-Zarr data model metadata validation when reading.

[Pyodide REPL]: https://pyodide.org/en/stable/console.html
[JupyterLite]: https://jupyterlite.readthedocs.io/en/latest/try/lab
[dask-image]: https://image.dask.org/en/latest/
[itk]: https://docs.itk.org/en/latest/learn/python_quick_start.html
[obstore]: https://developmentseed.org/obstore/latest/

### Install all optional dependencies

To install all optional dependencies:

```shell
pip install "ngff-zarr[cli,dask-image,itk,remote,validate]"
```

which is equivalent to:

```shell
pip install "ngff-zarr[all]"
```
