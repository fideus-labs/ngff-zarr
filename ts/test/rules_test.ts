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

test("a rule reaches the transformations of a byDimension", async () => {
  const example = bundled.find((rule) => rule.code === "NGFF-CT-027")!.example;
  const ome = example.ome as Record<string, unknown>;
  const scene = ome.scene as Record<string, unknown>;
  const rotation = (
    scene.coordinateTransformations as Record<string, unknown>[]
  )[0];
  const nested = {
    ome: {
      ...ome,
      scene: {
        ...scene,
        coordinateTransformations: [
          {
            input: rotation.input,
            output: rotation.output,
            type: "byDimension",
            transformations: [
              {
                inputAxes: [0, 1],
                outputAxes: [0, 1],
                transformation: {
                  type: "rotation",
                  rotation: rotation.rotation,
                },
              },
            ],
          },
        ],
      },
    },
  };
  assertEquals(await schemaProblems(nested, "0.6"), []);
  const codes = (await groupProblems(nested, "0.6")).map(
    (problem) => problem.code,
  );
  assert(codes.includes("NGFF-CT-027"), `NGFF-CT-027 not in ${codes}`);
});

test("a group read as a scene is checked against the scene schema", async () => {
  const attributes = { ome: { version: "0.6" } };
  assertEquals(await schemaProblems(attributes, "0.6"), []);
  assert(
    (await schemaProblems(attributes, "0.6", "", "scene")).length > 0,
    "a scene root with no scene must be refused",
  );
});
