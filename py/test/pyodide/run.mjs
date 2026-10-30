// Runs test_pyodide.py inside Pyodide hosted by Node, with the
// @fideus-labs/ngff-zarr JS package registered as the `ngff_zarr_js` module.
//
//   pip wheel py/ -w py/dist --no-deps        # from the repository root
//   cd py/test/pyodide && npm ci && node run.mjs [path/to/ngff_zarr-*.whl] [script.py]
//
// Needs Node >= 25 (JSPI on by default); Node 24 needs --experimental-wasm-jspi.
import { readdirSync, readFileSync, statSync } from "node:fs";
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

const pyodide = await loadPyodide();
pyodide.mountNodeFS("/mnt", dirname(wheel));
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
}
console.log("pyodide harness: OK");
process.exit(0); // the codec worker threads would otherwise keep Node alive
