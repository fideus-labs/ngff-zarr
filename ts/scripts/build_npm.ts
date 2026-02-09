#!/usr/bin/env -S deno run --allow-all
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Build script for npm package using tsc.
 *
 * This script:
 * 1. Copies source files to a staging directory
 * 2. Rewrites imports from Deno-style to Node-style
 * 3. Runs tsc to compile TypeScript
 * 4. Generates package.json with proper exports
 */

import { walk } from "jsr:@std/fs@^1/walk";
import { emptyDir, ensureDir } from "jsr:@std/fs@^1";
import { dirname, join, relative } from "jsr:@std/path@^1";

const SRC_DIR = "./src";
const STAGING_DIR = "./npm/src";
const NPM_DIR = "./npm";

/**
 * Rewrite imports in a TypeScript file from Deno-style to Node-style.
 */
function rewriteImports(content: string): string {
  // Replace .ts extensions with .js in relative imports
  content = content.replace(
    /(from\s+["'])(\.[^"']+)\.ts(["'])/g,
    "$1$2.js$3",
  );

  // Replace import() with .ts extensions
  content = content.replace(
    /(import\s*\(\s*["'])(\.[^"']+)\.ts(["']\s*\))/g,
    "$1$2.js$3",
  );

  // Replace export from with .ts extensions
  content = content.replace(
    /(export\s+\*\s+from\s+["'])(\.[^"']+)\.ts(["'])/g,
    "$1$2.js$3",
  );
  content = content.replace(
    /(export\s+\{[^}]*\}\s+from\s+["'])(\.[^"']+)\.ts(["'])/g,
    "$1$2.js$3",
  );

  // Replace .ts extensions in new URL() calls (for Worker imports)
  content = content.replace(
    /(new\s+URL\s*\(\s*["'])(\.[^"']+)\.ts(["'])/g,
    "$1$2.js$3",
  );

  // Replace jsr: imports with node-style module imports
  content = content.replace(
    /from\s+["']jsr:@std\/fs@[^"']*["']/g,
    'from "node:fs/promises"',
  );
  content = content.replace(
    /from\s+["']jsr:@std\/path@[^"']*["']/g,
    'from "node:path"',
  );

  return content;
}

/**
 * Copy and transform source files to staging directory.
 */
async function copyAndTransformSources(): Promise<void> {
  await emptyDir(NPM_DIR);
  await ensureDir(STAGING_DIR);

  for await (
    const entry of walk(SRC_DIR, { exts: [".ts"], includeDirs: false })
  ) {
    const relativePath = relative(SRC_DIR, entry.path);
    const destPath = join(STAGING_DIR, relativePath);

    await ensureDir(dirname(destPath));

    let content = await Deno.readTextFile(entry.path);
    content = rewriteImports(content);

    await Deno.writeTextFile(destPath, content);
  }
}

/**
 * Create tsconfig.json for the npm build.
 */
async function createTsConfig(): Promise<void> {
  const tsconfig = {
    compilerOptions: {
      target: "ES2022",
      module: "NodeNext",
      moduleResolution: "NodeNext",
      lib: ["ES2022", "DOM"],
      declaration: true,
      declarationMap: true,
      sourceMap: true,
      outDir: "../esm",
      rootDir: ".",
      strict: true,
      esModuleInterop: true,
      skipLibCheck: true,
      forceConsistentCasingInFileNames: true,
      resolveJsonModule: true,
      isolatedModules: true,
    },
    include: ["./**/*.ts"],
    exclude: ["node_modules"],
  };

  await Deno.writeTextFile(
    join(STAGING_DIR, "tsconfig.json"),
    JSON.stringify(tsconfig, null, 2),
  );
}

/**
 * Create package.json for the npm package.
 */
async function createPackageJson(): Promise<void> {
  const packageJson = {
    name: "@fideus-labs/ngff-zarr",
    version: "0.7.2",
    description:
      "TypeScript implementation of ngff-zarr for reading and writing OME-Zarr files",
    license: "MIT",
    type: "module",
    main: "./esm/mod.js",
    types: "./esm/mod.d.ts",
    repository: {
      type: "git",
      url: "git+https://github.com/fideus-labs/ngff-zarr.git",
    },
    bugs: {
      url: "https://github.com/fideus-labs/ngff-zarr/issues",
    },
    homepage: "https://github.com/fideus-labs/ngff-zarr#readme",
    keywords: [
      "ome-zarr",
      "zarr",
      "microscopy",
      "imaging",
      "ngff",
      "typescript",
      "deno",
    ],
    author: "ngff-zarr contributors",
    exports: {
      ".": {
        types: "./esm/mod.d.ts",
        browser: "./esm/browser-mod.js",
        node: "./esm/mod.js",
        import: "./esm/mod.js",
        default: "./esm/mod.js",
      },
      "./browser": {
        types: "./esm/browser-mod.d.ts",
        import: "./esm/browser-mod.js",
        default: "./esm/browser-mod.js",
      },
      "./methods/itkwasm-browser": {
        types: "./esm/methods/itkwasm-browser.d.ts",
        import: "./esm/methods/itkwasm-browser.js",
        default: "./esm/methods/itkwasm-browser.js",
      },
      "./methods/itkwasm-node": {
        types: "./esm/methods/itkwasm-node.d.ts",
        import: "./esm/methods/itkwasm-node.js",
        default: "./esm/methods/itkwasm-node.js",
      },
      "./process/to_multiscales-browser": {
        types: "./esm/process/to_multiscales-browser.d.ts",
        import: "./esm/process/to_multiscales-browser.js",
        default: "./esm/process/to_multiscales-browser.js",
      },
      "./process/to_multiscales-node": {
        types: "./esm/process/to_multiscales-node.d.ts",
        import: "./esm/process/to_multiscales-node.js",
        default: "./esm/process/to_multiscales-node.js",
      },
    },
    browser: {
      // Redirect itkwasm imports to browser versions
      "./esm/methods/itkwasm.js": "./esm/methods/itkwasm-browser.js",
      "./esm/methods/itkwasm-node.js": "./esm/methods/itkwasm-browser.js",
      // Redirect to_multiscales imports to browser versions
      "./esm/process/to_multiscales-node.js":
        "./esm/process/to_multiscales-browser.js",
    },
    files: ["esm/", "README.md", "LICENSE.txt"],
    dependencies: {
      "@fideus-labs/fizarrita": "^1.1.0",
      "@fideus-labs/worker-pool": "^1.0.0",
      "@itk-wasm/downsample": "^1.8.1",
      comlink: "^4.4.2",
      "itk-wasm": "^1.0.0-b.196",
      "@zarrita/storage": "^0.1.4",
      zod: "^4.0.2",
      zarrita: "^0.6.1",
    },
    devDependencies: {
      "@itk-wasm/image-io": "^1.6.0",
      typescript: "^5.7.2",
    },
  };

  await Deno.writeTextFile(
    join(NPM_DIR, "package.json"),
    JSON.stringify(packageJson, null, 2),
  );
}

/**
 * Run tsc to compile TypeScript.
 */
async function runTsc(): Promise<void> {
  const command = new Deno.Command("npx", {
    args: ["tsc", "-p", "src/tsconfig.json"],
    cwd: NPM_DIR,
    stdout: "inherit",
    stderr: "inherit",
  });

  const result = await command.output();
  if (!result.success) {
    throw new Error(`tsc failed with code ${result.code}`);
  }
}

/**
 * Install dependencies and run tsc.
 */
async function installAndBuild(): Promise<void> {
  // Install dependencies
  console.log("[build] Installing dependencies...");
  const installCmd = new Deno.Command("npm", {
    args: ["install"],
    cwd: NPM_DIR,
    stdout: "inherit",
    stderr: "inherit",
  });

  const installResult = await installCmd.output();
  if (!installResult.success) {
    throw new Error(`npm install failed with code ${installResult.code}`);
  }

  // Run tsc
  console.log("[build] Compiling TypeScript...");
  await runTsc();
}

/**
 * Copy static files to npm directory.
 */
async function copyStaticFiles(): Promise<void> {
  await Deno.copyFile("LICENSE.txt", join(NPM_DIR, "LICENSE.txt"));
  await Deno.copyFile("README.md", join(NPM_DIR, "README.md"));
}

/**
 * Clean up staging directory.
 */
async function cleanup(): Promise<void> {
  // Keep source files for source maps, but remove tsconfig
  try {
    await Deno.remove(join(STAGING_DIR, "tsconfig.json"));
  } catch {
    // Ignore if doesn't exist
  }
}

// Main build process
console.log("[build] Starting npm build with tsc...");

console.log("[build] Copying and transforming sources...");
await copyAndTransformSources();

console.log("[build] Creating tsconfig.json...");
await createTsConfig();

console.log("[build] Creating package.json...");
await createPackageJson();

console.log("[build] Installing dependencies and compiling...");
await installAndBuild();

console.log("[build] Copying static files...");
await copyStaticFiles();

console.log("[build] Cleaning up...");
await cleanup();

console.log("[build] Complete!");
