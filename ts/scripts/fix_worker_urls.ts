#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Point the emitted package's worker URLs at the emitted worker.
 *
 * The build rewrites relative import specifiers from `.ts` to `.js`, but not
 * the quoted path `"../workers/omero_codec_worker.ts"` that the codec and
 * OMERO pools pass to `new URL(..., import.meta.url)` to locate their worker.
 * This runs after `vp pack` and rewrites that path wherever it appears in
 * `dist/`.
 */

import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const WORKER_PATH_PATTERN = /(["'])\.\.\/workers\/omero_codec_worker\.ts\1/g;

/** Rewrite every quoted `../workers/omero_codec_worker.ts` to `.js`. */
export function rewriteWorkerUrls(source: string): string {
  return source.replace(
    WORKER_PATH_PATTERN,
    (_match, quote: string) =>
      `${quote}../workers/omero_codec_worker.js${quote}`,
  );
}

async function main(): Promise<void> {
  const rewritten: string[] = [];
  for (const name of await readdir("dist", { recursive: true })) {
    if (!name.endsWith(".js")) continue;
    const path = join("dist", name);
    const source = await readFile(path, "utf8");
    const result = rewriteWorkerUrls(source);
    if (result === source) continue;
    await writeFile(path, result);
    rewritten.push(path);
  }
  if (rewritten.length === 0) {
    throw new Error(
      "no codec worker URL found in dist/; has the worker spawn site moved?",
    );
  }
  console.log(`[fix_worker_urls] ${rewritten.join(", ")}`);
}

if (import.meta.main) await main();
