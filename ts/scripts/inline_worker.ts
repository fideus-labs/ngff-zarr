#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Inline the codec worker into `dist/` for the self-contained browser bundle
 * the Playwright tests load (`test/browser-npm/ngff-zarr.bundle.js`).
 *
 * esbuild does not follow `new Worker(new URL(..., import.meta.url))`, so the
 * worker is bundled separately here and handed to the package's one spawn
 * site, `createCodecWorker()` in `utils/worker_pool.js`, as a Blob URL.
 *
 * This edits `dist/` in place and is for the test bundle only. The published
 * package is the clean `dist/`, with the worker as a file for consumers'
 * bundlers to pick up: `npm pack` and `npm publish` run `npm run build` first
 * (the `prepack` script), so whatever this left behind is rebuilt.
 */

import { readFile, writeFile } from "node:fs/promises";
import { build } from "esbuild";

const SPAWN_MODULE = "dist/utils/worker_pool.js";
// After the build, createCodecWorker() spawns
//   new Worker(new URL("../workers/omero_codec_worker.js", import.meta.url), { type: "module" })
const SPAWN_PATTERN =
  /new\s+Worker\(\s*new\s+URL\(\s*["']\.\.\/workers\/omero_codec_worker\.js["']\s*,\s*import\.meta\.url\s*\)/;

/** Bundle the worker and its dependencies for inline use. */
async function bundleWorker(entryPoint: string): Promise<string> {
  const result = await build({
    entryPoints: [entryPoint],
    bundle: true,
    format: "esm",
    platform: "browser",
    minify: true,
    write: false,
    logLevel: "warning",
  });
  const [output] = result.outputFiles;
  if (!output) throw new Error("esbuild produced no output for the worker");
  return output.text;
}

const workerCode = await bundleWorker("dist/workers/omero_codec_worker.js");
console.log(`[inline_worker] worker bundle size: ${workerCode.length} bytes`);

const content = await readFile(SPAWN_MODULE, "utf8");
if (!SPAWN_PATTERN.test(content)) {
  throw new Error(`[inline_worker] worker spawn not found in ${SPAWN_MODULE}`);
}
// A replacer function keeps String.replace() from interpreting "$&" and
// friends in the replacement.
await writeFile(
  SPAWN_MODULE,
  `const codecWorkerUrl = URL.createObjectURL(new Blob([${JSON.stringify(workerCode)}], { type: "application/javascript" }));\n` +
    content.replace(SPAWN_PATTERN, () => "new Worker(codecWorkerUrl"),
);
console.log(`[inline_worker] ${SPAWN_MODULE}`);
