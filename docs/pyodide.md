<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🌐 Pyodide

`ngff-zarr` runs inside [Pyodide], Python compiled to WebAssembly, in browsers
and in Node. The public API is the same as on a native install: `from_ome_zarr`,
`to_multiscales`, `to_ome_zarr`, `from_hcs_zarr`, `open_array` and the rest.
Underneath, chunk encoding, decoding and sharding run in the
[`@fideus-labs/ngff-zarr`](./typescript.md) JavaScript package, because the
Rust-backed [zarrista] engine has no Pyodide build.

```python
import micropip

await micropip.install("ngff-zarr")

from ngff_zarr import from_ome_zarr, to_multiscales, to_ome_zarr

image = from_ome_zarr("https://example.org/image.ome.zarr").images[0]
multiscales = to_multiscales(image, scale_factors=[2, 4])
to_ome_zarr("/tmp/image.ome.zarr", multiscales)
```

micropip installs every dependency, including the Emscripten build of
`itkwasm-downsample` that the default downsampling method uses.

## Requirements

Python waits for JavaScript promises with JSPI (JavaScript Promise
Integration, `pyodide.ffi.run_sync`). Two things have to hold:

- **A JSPI-capable runtime:** Chrome and Edge 137+, Firefox 153+, Safari 27+,
  Node 25+. Node 24 needs `--experimental-wasm-jspi`.
- **A promising entry point:** the code must be entered through
  `pyodide.runPythonAsync`, an `async` Python function, or
  `PyProxy.callPromising()`. The [Pyodide REPL] and [JupyterLite]'s Pyodide
  kernel run every cell this way, so nothing is needed there. A plain
  `pyodide.runPython` call raises a `RuntimeError` naming these entry points.

## What works

- Reading OME-Zarr 0.4 and 0.5 (zarr v2 and v3, sharded or not) from URLs, from
  the Emscripten filesystem, and from `.ozx` archives.
- Writing OME-Zarr 0.4 and 0.5, sharded or not, to the Emscripten filesystem
  as a directory or an `.ozx` archive, including `metadata_only=True` stores
  filled later through `open_array`.
- `to_multiscales` with the `ITKWASM_GAUSSIAN`, `ITKWASM_BIN_SHRINK` and
  `ITKWASM_LABEL_IMAGE` methods. dask computes on its synchronous scheduler.
- HCS plates, read by path or URL, and written with `to_hcs_zarr` and
  `write_hcs_well_image`.
- Integer, floating point and `bool` data types; the test harness round-trips
  `uint8`, `uint16`, `int16`, `int32`, `int64`, `float32`, `float64` and
  `bool` in both OME-Zarr versions.

## What does not

- The command line interface.
- Remote writes. Write to the Emscripten filesystem and upload afterwards, as
  on other platforms.
- `storage_options` on reads. URLs are fetched as they are, so the server has
  to allow the request (CORS in browsers); passing options emits a warning.
- Writing sharded `bool` arrays; unsharded ones work.
- Listing the children of a URL store. Readers that navigate by the paths in
  the OME metadata, multiscales and HCS included, are unaffected.
- The `ITK_*` downsampling methods: native ITK has no Pyodide build.

`.ozx` archives are read through whole-member decompression per request, so a
large sharded archive reads slower than the same store as a directory.

## Where the JavaScript comes from

The first zarr operation imports the JavaScript package once, from one of two
places:

1. A module the host registered as `ngff_zarr_js`. Applications that bundle
   their own JavaScript, and Node (which cannot `import()` an `https://` URL),
   use this route:

   ```js
   import * as ngffZarr from "@fideus-labs/ngff-zarr";
   import * as zarr from "zarrita";

   pyodide.registerJsModule("ngff_zarr_js", { ...ngffZarr, zarr });
   ```

   `zarr` is the [zarrita] instance the package itself depends on; register
   the same one so arrays opened by one are understood by the other.
   `py/test/pyodide/run.mjs` in the repository is a complete Node example,
   including the codec worker setting Node needs.

2. Otherwise, `ngff_zarr.config.js_package_url`, which defaults to the
   package's browser entry on esm.sh, pinned to the release the Python package
   was tested with. Point it at a copy you host before the first zarr call:

   ```python
   from ngff_zarr import config

   config.js_package_url = "https://static.example.org/ngff-zarr/browser.js"
   ```

## Downsampling in a browser

The itkwasm pipelines run through the `itkwasm-downsample-emscripten` wheel,
which executes each pipeline in a web worker. The Node harness in the
repository swaps in the package's Node build, so the browser configuration has
not been exercised end to end yet. If a browser raises a `Worker` error on
the first `to_multiscales` call, `itkwasm.pyodide` offers two switches: set
`itkwasm.pyodide.js_resources.web_worker = False` to run the pipelines on the
main thread, or give the wrapper an explicit worker URL with
`itkwasm_downsample_emscripten.js_package.js_package.config = JsPackageConfig(module_url, pipeline_worker_url=...)`.

[Pyodide]: https://pyodide.org
[Pyodide REPL]: https://pyodide.org/en/stable/console.html
[JupyterLite]: https://jupyterlite.readthedocs.io/en/latest/try/lab
[zarrista]: https://github.com/developmentseed/zarrista
