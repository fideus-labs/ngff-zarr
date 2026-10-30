// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Invariants of the published package: the post-build worker URL rewrite,
 * and the dependency declarations in `package.json`.
 */

import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "vite-plus/test";

import { assert, assertEquals } from "./_assert.ts";
import { readTextFileSync } from "./_fs.ts";
import { rewriteWorkerUrls } from "../scripts/fix_worker_urls.ts";

const ZOD_COMPILE_MINIMUM = [4, 5, 0] as const;

const packageJson = JSON.parse(
  readTextFileSync(new URL("../package.json", import.meta.url)),
) as {
  name: string;
  dependencies: Record<string, string>;
  devDependencies: Record<string, string>;
};

// ---------------------------------------------------------------------------
// Worker URL rewrite
// ---------------------------------------------------------------------------

test("rewriteWorkerUrls points the codec worker URL at the emitted .js", () => {
  assertEquals(
    rewriteWorkerUrls(
      `new URL("../workers/omero_codec_worker.ts", import.meta.url)`,
    ),
    `new URL("../workers/omero_codec_worker.js", import.meta.url)`,
  );
  assertEquals(
    rewriteWorkerUrls(
      `new Worker(new URL('../workers/omero_codec_worker.ts', import.meta.url), { type: "module" })`,
    ),
    `new Worker(new URL('../workers/omero_codec_worker.js', import.meta.url), { type: "module" })`,
  );
});

// ---------------------------------------------------------------------------
// Dependency declarations
// ---------------------------------------------------------------------------

/** Every `.ts` file under `dir`, recursively. */
function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return entry.name.endsWith(".ts") ? [path] : [];
  });
}

/** The package name of a bare import specifier: "@s/p/sub" -> "@s/p". */
function packageName(specifier: string): string {
  const parts = specifier.split("/");
  return specifier.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0]!;
}

/** Bare (non-relative, non-builtin) package names imported by `src/`. */
function importedPackages(): Set<string> {
  const names = new Set<string>();
  const srcDir = fileURLToPath(new URL("../src/", import.meta.url));
  for (const file of sourceFiles(srcDir)) {
    for (const line of readTextFileSync(file).split("\n")) {
      // Doc comments quote consumer-side imports of this very package.
      if (/^\s*(\/\/|\*|\/\*)/.test(line)) continue;
      const match = /(?:from|import)\s*\(?\s*["']([^."'][^"']*)["']/.exec(line);
      if (!match) continue;
      const specifier = match[1]!;
      if (specifier.startsWith("node:")) continue;
      names.add(packageName(specifier));
    }
  }
  names.delete(packageJson.name);
  return names;
}

test("package.json declares every package that src imports", () => {
  // A dependency resolved only from this repo's own node_modules installs
  // fine here and is missing for every consumer; `fflate` shipped that way.
  const declared = new Set(Object.keys(packageJson.dependencies));
  const missing = [...importedPackages()].filter((n) => !declared.has(n));
  assertEquals(missing, [], `src imports undeclared packages: ${missing}`);
});

test("package.json declares no runtime dependency that src never imports", () => {
  const imported = importedPackages();
  const unused = Object.keys(packageJson.dependencies).filter(
    (n) => !imported.has(n),
  );
  assertEquals(unused, [], `unused dependencies: ${unused}`);
});

test("first-party @fideus-labs packages share one worker-pool", () => {
  // fizarrita re-exports worker-pool's types and runs its codec tasks on the
  // pool this package hands it, so two worker-pool copies in a consumer's tree
  // are incompatible. What has to hold is a single resolved copy: were
  // fizarrita to need a worker-pool this package's range excludes, the
  // lockfile would carry a nested second copy.
  const lock = JSON.parse(
    readTextFileSync(new URL("../package-lock.json", import.meta.url)),
  ) as { packages: Record<string, { version: string }> };
  const copies = Object.keys(lock.packages).filter((key) =>
    key.endsWith("node_modules/@fideus-labs/worker-pool"),
  );
  assertEquals(copies, ["node_modules/@fideus-labs/worker-pool"]);
});

test("published Zod range floors at schema compilation support", () => {
  const range = packageJson.dependencies["zod"];
  const match = /^\^(\d+)\.(\d+)\.(\d+)$/.exec(range ?? "");
  assert(match !== null, `zod range "${range}" is not a caret range`);
  const floor = match.slice(1, 4).map(Number) as [number, number, number];
  assertEquals(floor[0], ZOD_COMPILE_MINIMUM[0]);
  assert(
    floor[1] > ZOD_COMPILE_MINIMUM[1] ||
      (floor[1] === ZOD_COMPILE_MINIMUM[1] &&
        floor[2] >= ZOD_COMPILE_MINIMUM[2]),
    `zod floor ${floor.join(".")} is below ${ZOD_COMPILE_MINIMUM.join(".")}`,
  );
});
