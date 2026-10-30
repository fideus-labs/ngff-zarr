// Runs test_pyodide.py inside Pyodide hosted by Node, with the
// @fideus-labs/ngff-zarr JS package registered as the `ngff_zarr_js` module.
//
//   pip wheel py/ -w py/dist --no-deps        # from the repository root
//   cd py/test/pyodide && npm ci && node run.mjs [path/to/ngff_zarr-*.whl] [script.py]
//
// Needs Node >= 25 (JSPI on by default); Node 24 needs --experimental-wasm-jspi.
import {
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
import { fileURLToPath } from "node:url";

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

const pyodide = await loadPyodide({ env: { NGFF_ZARR_HTTP_BASE: httpBase } });
pyodide.mountNodeFS("/mnt", dirname(wheel));
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
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");
await micropip.install(`emfs:/mnt/${basename(wheel)}`);
await micropip.install("zarr"); // reference reader for the bridge's output
try {
  await pyodide.runPythonAsync(
    readFileSync(resolve(here, process.argv[3] ?? "test_pyodide.py"), "utf8"),
  );
} catch (error) {
  // The Python traceback is the useful part; Node's default handler would
  // also dump pyodide.asm.mjs's minified source line.
  console.error(error.message);
  process.exit(1);
} finally {
  server.close();
  rmSync(outDir, { recursive: true, force: true });
}
console.log("pyodide harness: OK");
process.exit(0); // the codec worker threads would otherwise keep Node alive
