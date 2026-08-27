// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The incremental factor keeps the requested one when it reaches the target.
 *
 * Mirrors `py/test/test_incremental_scale_factors.py`, so the two ports pick
 * the same factor for the same ladder.
 */

import { assertEquals } from "@std/assert";
import { calculateIncrementalFactor } from "../src/methods/itkwasm-shared.ts";

Deno.test("the requested factor is kept when it reaches the target", () => {
  // 15 / 4 floors to 3, and 4 already gets there: 5 would too, but was not
  // asked for.
  assertEquals(calculateIncrementalFactor(15, 3, 4), 4);
  assertEquals(calculateIncrementalFactor(61, 15, 4), 4);
  assertEquals(calculateIncrementalFactor(514, 128, 4), 4);
});

Deno.test("a floored ladder step is used only when it lands on the target", () => {
  // A [2, 5] ladder steps by 2.5 and floors to 2. A floored step is smaller
  // than the exact ratio, so it can only leave the level larger than the
  // target; it is used only when it lands on the target exactly, and the
  // search answers whenever it misses.
  for (let original = 4; original < 400; original++) {
    for (
      const [previousAbsolute, absolute] of [[2, 5], [3, 7], [4, 10], [
        2,
        11,
      ]]
    ) {
      const previous = Math.floor(original / previousAbsolute);
      const target = Math.floor(original / absolute);
      if (previous < 1 || target < 1) continue;
      const nominal = Math.floor(absolute / previousAbsolute);
      if (Math.floor(previous / nominal) === target) {
        assertEquals(
          calculateIncrementalFactor(previous, target, nominal),
          nominal,
        );
      }
    }
  }
});

Deno.test("without a nominal factor the search still lands on the target", () => {
  for (const [previous, target] of [[15, 3], [61, 15], [514, 128], [100, 33]]) {
    const factor = calculateIncrementalFactor(previous, target);
    assertEquals(Math.floor(previous / factor), target);
  }
});
