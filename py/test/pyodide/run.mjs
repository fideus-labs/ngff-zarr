// Runs test_pyodide.py inside Pyodide hosted by Node, with the
// @fideus-labs/ngff-zarr JS package registered as the `ngff_zarr_js` module.
//
//   pip wheel py/ -w py/dist --no-deps        # from the repository root
//   cd py/test/pyodide && npm ci && node run.mjs [path/to/ngff_zarr-*.whl]
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
pyodide.registerJsModule("ngff_zarr_js", { ...ngffZarr, zarr });
await pyodide.loadPackage("micropip");
await pyodide.pyimport("micropip").install(`emfs:/mnt/${basename(wheel)}`);
try {
  await pyodide.runPythonAsync(
    readFileSync(join(here, "test_pyodide.py"), "utf8"),
  );
} catch (error) {
  // The Python traceback is the useful part; Node's default handler would
  // also dump pyodide.asm.mjs's minified source line.
  console.error(error.message);
  process.exit(1);
}
console.log("pyodide harness: OK");
