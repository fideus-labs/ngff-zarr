// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Python-style value formatting helpers shared across the validation modules.
 *
 * These keep `ValidationError` message bytes identical to the Python port,
 * which the cross-language parity tests assert.
 */

/**
 * Render a list of names as a Python-style list literal, e.g. `['z', 'y']`.
 *
 * Mirrors Python's `str(list_of_str)` / f-string rendering: bracketed,
 * single-quoted elements, comma-space separated. Both the structural
 * spatial-axis-order message and the RFC 4 axis-orientation-completeness
 * message interpolate this verbatim, so the two ports stay byte-for-byte
 * identical. Axis names reaching these rules are single-quote-free, so the
 * plain single-quoted element form is exact.
 */
export function formatNameList(names: readonly string[]): string {
  return `[${names.map((name) => `'${name}'`).join(", ")}]`;
}
