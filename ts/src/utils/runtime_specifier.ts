// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Keep a module specifier or worker path out of bundlers' reach.
 *
 * Bundlers resolve string literals in `import()` and in
 * `new URL(..., import.meta.url)`. For the Node-only branches of this package
 * that is wrong twice over: a browser build cannot resolve
 * `node:worker_threads`, and a worker path would be emitted as an asset.
 * Holding the literal in a `const` does not help, because the build inlines
 * single-use constants into `dist/`. A call into another module is opaque to
 * that, so the string reaches the runtime untouched.
 */
export function runtimeSpecifier(specifier: string): string {
  return specifier;
}
