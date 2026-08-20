// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The RFC 4 vocabulary is written down more than once; pin the copies together.
 *
 * The 24 anatomical orientation values are transcribed independently as the
 * `AnatomicalOrientationValues` enum, the Zod schema the validator checks
 * against, and the antonym-pair table. Nothing derives one from another, so
 * these tests are what keeps them from drifting. The Python twin is
 * `py/test/test_rfc4_vocabulary.py`.
 */

import { assertEquals } from "@std/assert";
import { AnatomicalOrientationValues } from "../src/types/rfc4.ts";
import { AnatomicalOrientationValuesSchema } from "../src/schemas/rfc4.ts";
import { ANATOMICAL_AXIS_OF } from "../src/utils/rfc4_validation.ts";

const EXPECTED_COUNT = 24;

Deno.test("the enum and the pair table hold the same values", () => {
  const fromEnum = new Set<string>(Object.values(AnatomicalOrientationValues));
  const fromTable = new Set(Object.keys(ANATOMICAL_AXIS_OF));
  assertEquals([...fromEnum].sort(), [...fromTable].sort());
});

Deno.test("the schema accepts every value of the pair table", () => {
  for (const value of Object.keys(ANATOMICAL_AXIS_OF)) {
    assertEquals(
      AnatomicalOrientationValuesSchema.safeParse(value).success,
      true,
      value,
    );
  }
  assertEquals(
    AnatomicalOrientationValuesSchema.safeParse("north-to-south").success,
    false,
  );
});

Deno.test("the vocabulary has not changed size", () => {
  assertEquals(
    Object.values(AnatomicalOrientationValues).length,
    EXPECTED_COUNT,
  );
  assertEquals(Object.keys(ANATOMICAL_AXIS_OF).length, EXPECTED_COUNT);
});

Deno.test("every anatomical axis carries exactly two directions", () => {
  const perAxis = new Map<string, string[]>();
  for (const [value, axis] of Object.entries(ANATOMICAL_AXIS_OF)) {
    perAxis.set(axis, [...(perAxis.get(axis) ?? []), value]);
  }
  assertEquals(perAxis.size, EXPECTED_COUNT / 2);
  for (const [axis, values] of perAxis) {
    assertEquals(values.length, 2, axis);
  }
});
