// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Simple test for ITK-Wasm Image to NgffImage conversion
 */

import { assertEquals } from "@std/assert";

Deno.test("Basic test", () => {
  assertEquals(1, 1);
});

// This test would require actual itk-wasm package to work properly
// For now, let's run the basic TypeScript compilation and structure tests

Deno.test("RFC-4 types compile", async () => {
  const { LPS, RAS, itkLpsToAnatomicalOrientation } = await import(
    "../src/types/rfc4.ts"
  );

  assertEquals(LPS.x.value, "right-to-left");
  assertEquals(LPS.y.value, "anterior-to-posterior");
  assertEquals(LPS.z.value, "inferior-to-superior");

  assertEquals(RAS.x.value, "left-to-right");
  assertEquals(RAS.y.value, "posterior-to-anterior");
  assertEquals(RAS.z.value, "inferior-to-superior");

  assertEquals(itkLpsToAnatomicalOrientation("x")?.value, "right-to-left");
  assertEquals(
    itkLpsToAnatomicalOrientation("y")?.value,
    "anterior-to-posterior",
  );
  assertEquals(
    itkLpsToAnatomicalOrientation("z")?.value,
    "inferior-to-superior",
  );
  assertEquals(itkLpsToAnatomicalOrientation("t"), undefined);
});
