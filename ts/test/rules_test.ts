// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// The rule table, the Python package's rules.json, which both ports read.
import { test } from "vite-plus/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { assert, assertEquals } from "./_assert.ts";
import { NGFF_RULES } from "../src/schemas/ngff_rules.ts";
import { groupProblems, schemaProblems } from "../src/utils/validate.ts";

const bundled: {
  code: string;
  versions: string[];
  example: Record<string, unknown>;
}[] = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../py/ngff_zarr/rules.json", import.meta.url)),
    "utf8",
  ),
);

test("the rule table is the one the Python package bundles", () => {
  // Run `node scripts/sync_ome_zarr_schemas.mjs` when this fails.
  assertEquals(
    NGFF_RULES,
    bundled.map(({ code, versions }) => ({ code, versions })),
  );
});

for (const rule of bundled) {
  test(`${rule.code} fires on its example`, async () => {
    const version = (rule.example.ome as Record<string, unknown>)
      .version as string;
    assertEquals(await schemaProblems(rule.example, version), []);
    const codes = (await groupProblems(rule.example, version)).map(
      (problem) => problem.code,
    );
    assert(codes.includes(rule.code), `${rule.code} not in ${codes}`);
  });
}
