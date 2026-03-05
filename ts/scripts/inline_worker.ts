#!/usr/bin/env -S deno run --allow-all

// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Script to inline the omero_codec_worker.ts code into compute_omero.js
 * for bundled browser builds.
 *
 * This script:
 * 1. Bundles the worker code using esbuild
 * 2. Replaces the Worker URL constructor with a Blob URL in compute_omero.js
 *
 * Run this after the npm build but before creating the final bundle.
 */

import { join } from "jsr:@std/path@^1";

const NPM_DIR = "./npm";
const ESM_DIR = join(NPM_DIR, "esm");

/**
 * Bundle the worker code for inline use.
 */
async function bundleWorker(): Promise<string> {
  // Use esbuild to bundle the worker with all its dependencies
  // Note: Run from NPM_DIR, use relative path to worker
  // Using pinned esbuild version for supply-chain security
  const command = new Deno.Command("npx", {
    args: [
      "esbuild@0.24.2",
      "esm/workers/omero_codec_worker.js",
      "--bundle",
      "--format=esm",
      "--platform=browser",
      "--minify",
      "--target=es2020",
    ],
    cwd: NPM_DIR,
    stdout: "piped",
    stderr: "inherit",
  });

  const result = await command.output();
  if (!result.success) {
    throw new Error(`esbuild failed with code ${result.code}`);
  }

  return new TextDecoder().decode(result.stdout);
}

/**
 * Escape a string for use in a JavaScript template literal.
 */
function escapeForTemplateLiteral(code: string): string {
  return code.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$");
}

/**
 * Replace the Worker constructor with an inline Blob URL in compute_omero.js
 */
async function inlineWorkerCode(workerCode: string): Promise<void> {
  const targetPath = join(ESM_DIR, "utils", "compute_omero.js");

  let content = await Deno.readTextFile(targetPath);

  // Find and replace the createOmeroWorker function body.
  // After tsc compilation, createOmeroWorker() contains:
  //   new Worker(new URL("../workers/omero_codec_worker.js", import.meta.url), { type: "module" })
  const workerPattern =
    /new\s+Worker\(\s*new\s+URL\(\s*["']\.\.\/workers\/omero_codec_worker\.js["']\s*,\s*import\.meta\.url\s*\)\s*,\s*\{\s*type\s*:\s*["']module["']\s*\}\s*\)/g;

  if (!workerPattern.test(content)) {
    console.log(
      "[inline_worker] Worker pattern not found in file, skipping inline step",
    );
    console.log("[inline_worker] Content preview:", content.substring(0, 2000));
    return;
  }

  // Create the inline worker code
  const escapedWorkerCode = escapeForTemplateLiteral(workerCode);
  const inlineWorkerCreation = `(() => {
    const workerCode = \`${escapedWorkerCode}\`;
    const blob = new Blob([workerCode], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    workerBlobUrl = url;
    return new Worker(url, { type: "module" });
  })()`;

  // Replace the pattern using a replacer function to avoid
  // String.replace() interpreting "$&", "$`", etc. in the
  // replacement string as special back-reference patterns.
  content = content.replace(workerPattern, () => inlineWorkerCreation);

  await Deno.writeTextFile(targetPath, content);
  console.log("[inline_worker] Successfully inlined worker code");
}

// Main process
console.log("[inline_worker] Starting worker inlining...");

try {
  console.log("[inline_worker] Bundling worker code...");
  const workerCode = await bundleWorker();
  console.log(`[inline_worker] Worker bundle size: ${workerCode.length} bytes`);

  console.log("[inline_worker] Inlining worker into compute_omero module...");
  await inlineWorkerCode(workerCode);

  console.log("[inline_worker] Complete!");
} catch (error) {
  console.error("[inline_worker] Error:", error);
  Deno.exit(1);
}
