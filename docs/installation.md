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

`tensorstore` : Support writing with [tensorstore].

`validate` : Support OME-Zarr data model metadata validation when reading.

[Pyodide REPL]: https://pyodide.org/en/stable/console.html
[JupyterLite]: https://jupyterlite.readthedocs.io/en/latest/try/lab
[dask-image]: https://image.dask.org/en/latest/
[itk]: https://docs.itk.org/en/latest/learn/python_quick_start.html
[tensorstore]: https://google.github.io/tensorstore/

### Install all optional dependencies

To install all optional dependencies:

```shell
pip install "ngff-zarr[cli,dask-image,itk,tensorstore,validate]"
```

which is equivalent to:

```shell
pip install "ngff-zarr[all]"
```
