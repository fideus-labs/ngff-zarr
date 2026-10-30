// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { defineConfig } from "vite-plus";

export default defineConfig({
  test: {
    include: ["test/**/*_test.ts"],
    // Several files share `test/output/`, so run files one after another
    // until they get private directories.
    fileParallelism: false,
    // Downsampling and worker-pool tests take well over Vitest's 5 s default.
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
  lint: {
    rules: {
      // `new Array(n)` is this codebase's idiom for preallocating a
      // fixed-length array.
      "unicorn/no-new-array": "off",
    },
    overrides: [
      {
        files: ["test/**"],
        rules: {
          // Tests spell index arithmetic out (`0 * 6 + 5`) for readability.
          "oxc/erasing-op": "off",
        },
      },
    ],
    ignorePatterns: [
      "dist/**",
      "test/browser-npm/ngff-zarr.bundle.js",
      "test/browser-npm/test-results/**",
    ],
  },
  fmt: {
    printWidth: 80,
    semi: true,
    singleQuote: false,
    ignorePatterns: [
      "**/*.toml",
      "**/*.lock",
      "**/*.yml",
      "**/*.yaml",
      "package-lock.json",
      "dist/**",
      "test/browser-npm/*.html",
      "test/browser-npm/ngff-zarr.bundle.js",
      "test/browser-npm/test-results/**",
      "test/output/**",
    ],
  },
  pack: {
    entry: ["src/**/*.ts"],
    unbundle: true,
    format: ["esm"],
    platform: "node",
    dts: true,
    sourcemap: true,
    outDir: "dist",
    outExtensions: () => ({ js: ".js" }),
  },
});
