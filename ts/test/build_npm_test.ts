// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Tests for the npm build's import rewriting.
 *
 * `rewriteImports` runs over every source file on its way into the npm
 * package, so a regex that over- or under-matches silently ships a broken
 * build. These pin the specifier forms this repo actually uses.
 */

import { assertEquals } from "@std/assert";

import { rewriteImports } from "../scripts/build_npm.ts";

// ---------------------------------------------------------------------------
// Relative .ts → .js
// ---------------------------------------------------------------------------

Deno.test("rewriteImports rewrites relative .ts specifiers to .js", () => {
  assertEquals(
    rewriteImports(`import { a } from "./utils/codecs.ts";`),
    `import { a } from "./utils/codecs.js";`,
  );
  assertEquals(
    rewriteImports(`export * from "../types/units.ts";`),
    `export * from "../types/units.js";`,
  );
  assertEquals(
    rewriteImports(`export { b } from "./io/rfc9_zip.ts";`),
    `export { b } from "./io/rfc9_zip.js";`,
  );
  assertEquals(
    rewriteImports(`const m = await import("./config.ts");`),
    `const m = await import("./config.js");`,
  );
});

Deno.test("rewriteImports rewrites worker URLs to .js", () => {
  assertEquals(
    rewriteImports(
      `new URL("../workers/omero_codec_worker.ts", import.meta.url)`,
    ),
    `new URL("../workers/omero_codec_worker.js", import.meta.url)`,
  );
});

// ---------------------------------------------------------------------------
// npm: specifiers
// ---------------------------------------------------------------------------

Deno.test("rewriteImports strips the npm: prefix and version range", () => {
  assertEquals(
    rewriteImports(`import { registry } from "npm:zarrita@^0.6.1";`),
    `import { registry } from "zarrita";`,
  );
  assertEquals(
    rewriteImports(`import x from "npm:zarrita";`),
    `import x from "zarrita";`,
  );
});

Deno.test("rewriteImports handles scoped npm packages", () => {
  assertEquals(
    rewriteImports(`import x from "npm:@fideus-labs/fizarrita@^1.3.0";`),
    `import x from "@fideus-labs/fizarrita";`,
  );
  assertEquals(
    rewriteImports(`import x from "npm:@scope/pkg";`),
    `import x from "@scope/pkg";`,
  );
});

Deno.test("rewriteImports preserves npm subpaths", () => {
  // Dropping the subpath would resolve to the package root and silently
  // ship a build importing the wrong module.
  assertEquals(
    rewriteImports(
      `import x from "npm:@fideus-labs/fizarrita@^1.3.0/codec-worker";`,
    ),
    `import x from "@fideus-labs/fizarrita/codec-worker";`,
  );
  assertEquals(
    rewriteImports(`import x from "npm:zarrita@^0.6.1/internals/util";`),
    `import x from "zarrita/internals/util";`,
  );
  assertEquals(
    rewriteImports(`import x from "npm:pkg/sub";`),
    `import x from "pkg/sub";`,
  );
});

Deno.test("rewriteImports handles prerelease and exact versions", () => {
  assertEquals(
    rewriteImports(`import x from "npm:itk-wasm@1.0.0-b.196";`),
    `import x from "itk-wasm";`,
  );
  assertEquals(
    rewriteImports(`import x from "npm:pkg@~1.2.3/a/b";`),
    `import x from "pkg/a/b";`,
  );
});

Deno.test("rewriteImports leaves bare npm package names alone", () => {
  const bare = `import { setWorker } from "@fideus-labs/fizarrita";`;
  assertEquals(rewriteImports(bare), bare);
});

Deno.test("rewriteImports rewrites side-effect and dynamic imports", () => {
  assertEquals(
    rewriteImports(`import "npm:some-polyfill@1.0.0";`),
    `import "some-polyfill";`,
  );
  assertEquals(
    rewriteImports(`const m = await import("npm:zarrita@^0.6.1");`),
    `const m = await import("zarrita");`,
  );
});

Deno.test("rewriteImports leaves non-specifier npm: strings alone", () => {
  // Only module-specifier positions are rewritten. A string that merely
  // starts with `npm:` — an error message, a label — is data, not an import.
  for (
    const src of [
      `const label = "npm:zarrita@^0.6.1";`,
      `throw new Error("install npm:pkg@1.0.0/sub first");`,
      `const specifier = "npm:zarrita@^0.6.1";`,
    ]
  ) {
    assertEquals(rewriteImports(src), src);
  }
});

Deno.test("rewriteImports leaves commented-out imports alone", () => {
  // Several source comments discuss `npm:` versus `jsr:` resolution, so a
  // quoted specifier inside one is a realistic way to corrupt the build.
  for (
    const src of [
      `// import "npm:pkg@1.0.0"`,
      `// see: from "npm:zarrita@^0.6.1"`,
      `/* import "npm:pkg@1.0.0" */`,
    ]
  ) {
    assertEquals(rewriteImports(src), src);
  }
});

Deno.test("rewriteImports leaves a specifier nested in a string alone", () => {
  // The inner quotes belong to the outer literal, not to an import.
  const src = `const text = 'from "npm:pkg@1.0.0"';`;
  assertEquals(rewriteImports(src), src);
});

Deno.test("rewriteImports rewrites imports but not surrounding prose", () => {
  // A whole-file shape: a JSDoc block mentioning specifiers, a real import,
  // a commented-out one, and a specifier embedded in string data.
  const src = [
    `/**`,
    ` * Use import "npm:pkg@1.0.0" to load it.`,
    ` * Also: from "npm:zarrita@^0.6.1" resolves differently.`,
    ` */`,
    `import x from "npm:zarrita@^0.6.1";`,
    `// import "npm:other@2.0.0"`,
    `const s = 'from "npm:inner@1.0.0"';`,
  ].join("\n");

  const expected = [
    `/**`,
    ` * Use import "npm:pkg@1.0.0" to load it.`,
    ` * Also: from "npm:zarrita@^0.6.1" resolves differently.`,
    ` */`,
    `import x from "zarrita";`,
    `// import "npm:other@2.0.0"`,
    `const s = 'from "npm:inner@1.0.0"';`,
  ].join("\n");

  assertEquals(rewriteImports(src), expected);
});

// ---------------------------------------------------------------------------
// jsr: specifiers
// ---------------------------------------------------------------------------

Deno.test("rewriteImports maps @std/fs and @std/path to node builtins", () => {
  assertEquals(
    rewriteImports(`import { walk } from "jsr:@std/fs@^1/walk";`),
    `import { walk } from "node:fs/promises";`,
  );
  assertEquals(
    rewriteImports(`import { join } from "jsr:@std/path@^1";`),
    `import { join } from "node:path";`,
  );
});

// ---------------------------------------------------------------------------
// Non-specifier text must survive untouched
// ---------------------------------------------------------------------------

Deno.test("rewriteImports leaves unrelated .ts mentions alone", () => {
  // Only relative specifiers are rewritten — a .ts inside a doc comment or
  // a non-import string is not an import.
  const prose = `// see codecs.ts for details`;
  assertEquals(rewriteImports(prose), prose);
});

Deno.test("rewriteImports is idempotent", () => {
  const source = [
    `import { registry } from "npm:zarrita@^0.6.1";`,
    `import { a } from "./utils/codecs.ts";`,
    `import { join } from "jsr:@std/path@^1";`,
  ].join("\n");

  const once = rewriteImports(source);
  assertEquals(rewriteImports(once), once);
});
