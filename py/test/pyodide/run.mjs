// Runs test_pyodide.py inside Pyodide hosted by Node, with the
// @fideus-labs/ngff-zarr JS package registered as the `ngff_zarr_js` module.
//
//   pip wheel py/ -w py/dist --no-deps        # from the repository root
//   cd py/test/pyodide && npm ci && node run.mjs [path/to/ngff_zarr-*.whl] [script.py]
//
// Needs Node >= 25 (JSPI on by default); Node 24 needs --experimental-wasm-jspi.
//
// With NGFF_ZARR_POOL=<path to python-wasm-parallelism's dist/index.js> the
// downsampling runs on a pyodide-pool worker pool of NGFF_ZARR_POOL_SIZE
// (default 4) workers, registered for Python as `js_pyodide_pool`.
import {
  copyFileSync,
  createReadStream,
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
} from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as ngffZarr from "@fideus-labs/ngff-zarr";
import { loadPyodide } from "pyodide";
import * as zarr from "zarrita";

const here = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(here, "../../dist");
const newestWheel = () => {
  const wheels = readdirSync(distDir).filter((f) => f.endsWith(".whl"));
  if (!wheels.length) {
    throw new Error(`no wheel in ${distDir}; run: pip wheel py/ -w py/dist --no-deps`);
  }
  const mtime = (f) => statSync(join(distDir, f)).mtimeMs;
  return join(distDir, wheels.sort((a, b) => mtime(b) - mtime(a))[0]);
};
const wheel = process.argv[2] ? resolve(process.argv[2]) : newestWheel();
// The itkwasm wrapper imports the package's browser build; Node runs its Node
// build instead through this shim.
const shimUrl = pathToFileURL(resolve(here, "itkwasm-downsample-node.mjs")).href;

// A host directory, mounted at /out in Pyodide, that this static server also
// serves: what Python writes there, Python can read back over HTTP. zarrita
// fetches shard indexes with Range requests, so 206 responses are needed.
const outDir = mkdtempSync(join(tmpdir(), "ngff-zarr-pyodide-"));
const server = createServer((req, res) => {
  const file = join(outDir, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!file.startsWith(outDir) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end();
    return;
  }
  const size = statSync(file).size;
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
  let [start, end] = [0, size - 1];
  if (range && size) {
    if (range[1] === "") start = Math.max(size - Number(range[2]), 0);
    else [start, end] = [Number(range[1]), range[2] ? Math.min(Number(range[2]), end) : end];
    res.writeHead(206, { "Content-Range": `bytes ${start}-${end}/${size}` });
  } else {
    res.writeHead(200, { "Content-Length": size });
  }
  if (!size) return res.end();
  createReadStream(file, { start, end }).pipe(res);
});
await new Promise((listening) => server.listen(0, "127.0.0.1", listening));
const httpBase = `http://127.0.0.1:${server.address().port}`;

// A pyodide-pool PyodidePool whose workers hold the driver's packages and
// the itkwasm shim, plus a count of the tasks Python sent it.
async function createPool(wheelUrl) {
  const indexUrl = pathToFileURL(resolve(process.env.NGFF_ZARR_POOL)).href;
  const { PyodidePool } = await import(indexUrl);
  const poolSize = Number(process.env.NGFF_ZARR_POOL_SIZE ?? 4);
  const pool = new PyodidePool({ poolSize, workerUrl: new URL("pyodide-worker.js", indexUrl) });
  // One map run of poolSize tasks lands on poolSize distinct workers.
  const setup = `
import micropip
await micropip.install([${JSON.stringify(wheelUrl)}, "zarr"])
from itkwasm.pyodide import JsPackageConfig
from itkwasm_downsample_emscripten.js_package import js_package
js_package.config = JsPackageConfig(${JSON.stringify(shimUrl)})
`;
  const slots = Array.from({ length: poolSize }, (_, index) => index);
  await pool.map(setup, slots, { packages: ["micropip"] }).promise;
  let submits = 0;
  const counted = Object.create(pool, {
    runPickled: {
      value: (...args) => {
        submits += 1;
        return pool.runPickled(...args);
      },
    },
  });
  return { pool, module: { pool: counted, submits: () => submits } };
}

let status = 0;
let pool;
try {
  const pyodide = await loadPyodide({
    env: { NGFF_ZARR_HTTP_BASE: httpBase, NGFF_ZARR_ITKWASM_MODULE: shimUrl },
  });
  pyodide.mountNodeFS("/out", outDir);
  // The npm build inlines its codec worker as a blob: URL, which Node's
  // worker_threads refuse; an undefined workerUrl makes fizarrita pick its
  // packaged node worker entry instead.
  const nodeWorker = { workerUrl: undefined };
  pyodide.registerJsModule("ngff_zarr_js", {
    ...ngffZarr,
    zarr,
    zarrGet: (arr, sel, opts) =>
      ngffZarr.zarrGet(arr, sel, { ...nodeWorker, ...opts }),
    zarrSet: (arr, sel, value, opts) =>
      ngffZarr.zarrSet(arr, sel, value, { ...nodeWorker, ...opts }),
  });
  // Served, not mounted: pool workers install the driver's packages by URL.
  copyFileSync(wheel, join(outDir, basename(wheel)));
  const wheelUrl = `${httpBase}/${basename(wheel)}`;
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(wheelUrl);
  await micropip.install("zarr"); // reference reader for the bridge's output
  if (process.env.NGFF_ZARR_POOL) {
    const created = await createPool(wheelUrl);
    pool = created.pool;
    pyodide.registerJsModule("js_pyodide_pool", created.module);
  }
  await pyodide.runPythonAsync(
    readFileSync(resolve(here, process.argv[3] ?? "test_pyodide.py"), "utf8"),
  );
  console.log("pyodide harness: OK");
} catch (error) {
  // The Python traceback is the useful part; Node's default handler would
  // also dump pyodide.asm.mjs's minified source line.
  console.error(error.message);
  status = 1;
} finally {
  pool?.terminate();
  server.close();
  rmSync(outDir, { recursive: true, force: true });
}
process.exit(status); // the codec worker threads would otherwise keep Node alive
