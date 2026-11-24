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
  examples: { ome: { version: string } }[];
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

for (const { code, examples } of bundled) {
  for (const [index, example] of examples.entries()) {
    test(`${code} fires on its example ${index}`, async () => {
      const version = example.ome.version;
      assertEquals(await schemaProblems(example, version), []);
      const codes = (await groupProblems(example, version)).map(
        (problem) => problem.code,
      );
      assert(codes.includes(code), `${code} not in ${codes}`);
    });
  }
}

const OPEN: [string, unknown][] = [
  ["coordinateSystems", null],
  ["coordinateSystems", [{}]],
  ["coordinateSystems", [null]],
  ["coordinateTransformations", null],
  ["coordinateTransformations", [null]],
  ["coordinateTransformations", [{ type: "sequence", transformations: null }]],
  ["datasets", null],
  ["datasets", [null]],
];

for (const [key, value] of OPEN) {
  test(`${key} ${JSON.stringify(value)} beside a scene does not break the rules`, async () => {
    // `ome` is open to properties the schema does not define; the rules check
    // only what has the shape they expect, and still report the scene's finding.
    const example = bundled.find((rule) => rule.code === "NGFF-CT-015")!
      .examples[0];
    const attributes = { ome: { ...example.ome, [key]: value } };
    assertEquals(await schemaProblems(attributes, "0.6"), []);
    assertEquals(
      (await groupProblems(attributes, "0.6")).map((problem) => problem.code),
      ["NGFF-CT-015"],
    );
  });
}

test("series metadata is checked against its schema", async () => {
  assert((await schemaProblems({ series: 42 }, "0.4")).length > 0);
  assert(
    (await schemaProblems({ ome: { version: "0.5", series: 42 } }, "0.5"))
      .length > 0,
  );
});

test("a group read as a scene is checked against the scene schema", async () => {
  const attributes = { ome: { version: "0.6" } };
  assertEquals(await schemaProblems(attributes, "0.6"), []);
  assert(
    (await schemaProblems(attributes, "0.6", "", "scene")).length > 0,
    "a scene root with no scene must be refused",
  );
});

test("a kind its version does not define is a problem", async () => {
  const attributes = { ome: { version: "0.5" } };
  assert(
    (await schemaProblems(attributes, "0.5", "", "scene")).length > 0,
    "a 0.5 scene must be refused",
  );
});
