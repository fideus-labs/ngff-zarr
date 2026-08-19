// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Tests for the npm build's import rewriting and declared dependencies.
 *
 * `rewriteImports` runs over every source file on its way into the npm
 * package, so a regex that over- or under-matches silently ships a broken
 * build. These pin the specifier forms this repo actually uses.
 *
 * The dependency tests guard the other half of the published package: the
 * version ranges written into package.json.
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import { format, parse, parseRange, satisfies } from "@std/semver";

import {
  NPM_DEPENDENCIES,
  NPM_DEV_DEPENDENCIES,
  rewriteImports,
} from "../scripts/build_npm.ts";

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

Deno.test("rewriteImports rewrites across a comment before the specifier", () => {
  // A comment is whitespace to the JS grammar, so it does not stop the
  // literal after it from being a module specifier.
  assertEquals(
    rewriteImports(`import /* c */ "npm:pkg@1.0.0";`),
    `import /* c */ "pkg";`,
  );
  assertEquals(
    rewriteImports(`import x from /* c */ "npm:zarrita@^0.6.1";`),
    `import x from /* c */ "zarrita";`,
  );
  assertEquals(
    rewriteImports(`import x from // c\n  "npm:zarrita@^0.6.1";`),
    `import x from // c\n  "zarrita";`,
  );
  assertEquals(
    rewriteImports(`const m = await import(/* c */ "npm:zarrita@^0.6.1");`),
    `const m = await import(/* c */ "zarrita");`,
  );
});

Deno.test("rewriteImports does not let comment text fake a specifier", () => {
  // Skipped comments contribute a placeholder space, never their own text —
  // otherwise a comment ending in `import` would capture the next literal.
  for (
    const src of [
      `/* import */ "npm:pkg@1.0.0"`,
      `// the word from\n"npm:pkg@1.0.0"`,
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

// ---------------------------------------------------------------------------
// Published dependency ranges vs. the versions we actually test
// ---------------------------------------------------------------------------

/**
 * The published package.json declares its own dependency ranges, separate from
 * the `deno.json` import map. Nothing in the build wires the two together, so
 * bumping one and forgetting the other is silent: `deno task test` keeps
 * passing against the locked version while npm consumers resolve a version the
 * suite never exercised. These tests pin the invariant instead.
 */

/**
 * Packages published as npm dependencies that this repo resolves only from
 * JSR. Their npm range is checked against the JSR-resolved version, which
 * holds only because these packages publish the same version to both
 * registries. Adding to this list weakens the check — prefer an npm
 * resolution when one exists.
 */
const JSR_RESOLVED_DEPENDENCIES = new Set(["@zarrita/storage"]);

/**
 * Resolved versions from `deno.lock`, keyed by bare package name, kept
 * separate per registry.
 *
 * npm and JSR are independent registries: `jsr:@scope/pkg@1.2.3` and
 * `npm:@scope/pkg@1.2.3` are different artifacts that need not agree. Merging
 * them would let a JSR version vouch for an npm dependency range.
 */
function lockedVersions(): {
  npm: Map<string, Set<string>>;
  jsr: Map<string, Set<string>>;
} {
  // Resolved against this module, not the cwd, so the test works under
  // `deno test` from either the repo root or ts/.
  const lockPath = new URL("../deno.lock", import.meta.url);
  const lock = JSON.parse(Deno.readTextFileSync(lockPath)) as {
    specifiers?: Record<string, string>;
  };
  const resolved = {
    npm: new Map<string, Set<string>>(),
    jsr: new Map<string, Set<string>>(),
  };

  for (const [specifier, version] of Object.entries(lock.specifiers ?? {})) {
    const match = /^(npm|jsr):(.+)$/.exec(specifier);
    if (!match) continue;
    const [, registry, bare] = match;

    // "@scope/pkg@^1.2.3" -> "@scope/pkg"; "pkg@~1.2.3" -> "pkg"
    const at = bare.lastIndexOf("@");
    if (at <= 0) continue;
    const name = bare.slice(0, at);
    // A peer-dependency suffix ("2.1.0_zarrita@0.6.1") is not part of the
    // version.
    const [plain] = version.split("_");

    // Every distinct resolution is kept. Collapsing them (say, to the highest)
    // could let a version the source never builds against satisfy the check.
    const bucket = resolved[registry as "npm" | "jsr"];
    const versions = bucket.get(name) ?? new Set<string>();
    versions.add(plain);
    bucket.set(name, versions);
  }

  return resolved;
}

/** Every locked version a published dependency's range is checked against. */
function lockedVersionsFor(
  name: string,
  resolved: ReturnType<typeof lockedVersions>,
): Set<string> | undefined {
  return resolved.npm.get(name) ??
    (JSR_RESOLVED_DEPENDENCIES.has(name) ? resolved.jsr.get(name) : undefined);
}

Deno.test("every published dependency resolves in deno.lock", () => {
  const resolved = lockedVersions();
  const missing = Object.keys(NPM_DEPENDENCIES).filter(
    (n) => lockedVersionsFor(n, resolved) === undefined,
  );

  assertEquals(
    missing,
    [],
    `package.json declares dependencies absent from deno.lock: ${
      missing.join(", ")
    }. They are published to consumers but never resolved here, so nothing ` +
      `tests them.`,
  );
});

Deno.test("JSR-resolved exceptions really are absent from npm resolution", () => {
  // Each exception weakens the check above, so it has to keep earning its
  // place: once a package resolves from npm, drop it from the set.
  const resolved = lockedVersions();
  const stale = [...JSR_RESOLVED_DEPENDENCIES].filter((n) =>
    resolved.npm.has(n)
  );

  assertEquals(
    stale,
    [],
    `${stale.join(", ")} now resolves from npm in deno.lock; remove it from ` +
      `JSR_RESOLVED_DEPENDENCIES so its npm version is checked directly.`,
  );
});

Deno.test("published dependency ranges are satisfied by locked versions", () => {
  const resolved = lockedVersions();

  // A range may sit *below* the locked version on purpose — publishing a
  // lower supported floor. What it must never do is require a version this
  // repo has not resolved and tested. When a package is locked at several
  // versions, every one of them must qualify: picking a single "winner" would
  // let a version the source never builds against carry the check.
  const violations: string[] = [];
  for (const [name, range] of Object.entries(NPM_DEPENDENCIES)) {
    const locked = lockedVersionsFor(name, resolved);
    if (!locked) continue; // reported by the test above

    const unsatisfied = [...locked]
      .filter((v) => !satisfies(parse(v), parseRange(range)))
      .map((v) => format(parse(v)));

    if (unsatisfied.length > 0) {
      violations.push(
        `${name}: package.json requires "${range}" but deno.lock pins ` +
          unsatisfied.join(", "),
      );
    }
  }

  assertEquals(violations, [], violations.join("\n"));
});

Deno.test("dependency ranges are well-formed semver ranges", () => {
  for (
    const [name, range] of [
      ...Object.entries(NPM_DEPENDENCIES),
      ...Object.entries(NPM_DEV_DEPENDENCIES),
    ]
  ) {
    // parseRange throws on a malformed range; a typo'd range would otherwise
    // only surface at `npm install` time in a consumer's project.
    parseRange(range);
    assertEquals(typeof range, "string", `${name} range must be a string`);
  }
});

Deno.test("first-party @fideus-labs packages stay in lockstep", () => {
  // fizarrita re-exports worker-pool's types; a split-major pair puts two
  // incompatible copies in a consumer's tree.
  const fizarrita = NPM_DEPENDENCIES["@fideus-labs/fizarrita"];
  const workerPool = NPM_DEPENDENCIES["@fideus-labs/worker-pool"];

  // Assert presence first, so dropping both does not pass as undefined ===
  // undefined.
  assertNotEquals(
    fizarrita,
    undefined,
    "NPM_DEPENDENCIES must declare @fideus-labs/fizarrita",
  );
  assertNotEquals(
    workerPool,
    undefined,
    "NPM_DEPENDENCIES must declare @fideus-labs/worker-pool",
  );

  assertEquals(
    fizarrita,
    workerPool,
    `@fideus-labs/fizarrita ("${fizarrita}") and @fideus-labs/worker-pool ` +
      `("${workerPool}") must be bumped together`,
  );
});
