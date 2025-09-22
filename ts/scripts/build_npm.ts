#!/usr/bin/env -S deno run --allow-all
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { build, emptyDir } from "dnt";

await emptyDir("./npm");

await build({
  entryPoints: ["./src/mod.ts"],
  outDir: "./npm",
  shims: {
    deno: false, // Disable Deno shims for browser compatibility
  },
  test: false,
  typeCheck: false,
  package: {
    name: "@fideus-labs/ngff-zarr",
    version: "0.0.2",
    description:
      "TypeScript implementation of ngff-zarr for reading and writing OME-Zarr files",
    license: "MIT",
    repository: {
      type: "git",
      url: "git+https://github.com/thewtex/ngff-zarr.git",
    },
    bugs: {
      url: "https://github.com/thewtex/ngff-zarr/issues",
    },
    homepage: "https://github.com/thewtex/ngff-zarr#readme",
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
    main: "./esm/mod.js",
    types: "./esm/mod.d.ts",
    exports: {
      ".": {
        import: "./esm/mod.js",
        require: "./script/mod.js",
        types: "./esm/mod.d.ts",
      },
    },
    files: ["esm/", "script/", "types/", "README.md", "LICENSE"],
    dependencies: {
      "itk-wasm": "^1.0.0-b.195",
      "p-queue": "^8.1.0",
      "@zarrita/storage": "^0.1.1",
      zod: "^4.0.2",
      zarrita: "^0.5.2",
    },
  },
  postBuild() {
    Deno.copyFileSync("../README.md", "npm/README.md");
    Deno.copyFileSync("../LICENSE.txt", "npm/LICENSE");
  },
  compilerOptions: {
    lib: ["ES2022", "DOM"],
    target: "ES2022",
  },
});
