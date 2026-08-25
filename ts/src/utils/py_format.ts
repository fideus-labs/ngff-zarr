// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Python-style value formatting helpers shared across the validation modules.
 *
 * These keep `ValidationError` message bytes identical to the Python port,
 * which the cross-language parity tests assert.
 */

/**
 * Render a string as Python's `repr()` renders it.
 *
 * A single-quoted literal, switching to double quotes when the value contains a
 * single quote but no double quote. RFC-3 puts arbitrary strings in `name` and
 * `type`, and no schema excludes a quote, so the switch is reachable.
 */
export function pyRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escaped = value.replaceAll("\\", "\\\\").replaceAll(
    quote,
    `\\${quote}`,
  );
  return `${quote}${escaped}${quote}`;
}

/**
 * Render a list of names as a Python-style list literal, e.g. `['z', 'y']`.
 *
 * Mirrors Python's `str(list_of_str)` / f-string rendering: bracketed,
 * `repr`-rendered elements, comma-space separated. Both the structural
 * spatial-axis-order message and the RFC 4 axis-orientation-on-non-space
 * message interpolate this verbatim, so the two ports stay byte-for-byte
 * identical.
 */
export function formatNameList(names: readonly string[]): string {
  return `[${names.map(pyRepr).join(", ")}]`;
}

/**
 * Render an optional string the way Python renders `{value!r}`.
 *
 * An axis may declare no `type`; Python renders that `None`, TypeScript would
 * render `undefined` (or `null` when the JSON carried an explicit null).
 */
export function pyReprOptional(value: string | null | undefined): string {
  return value === undefined || value === null ? "None" : pyRepr(value);
}
