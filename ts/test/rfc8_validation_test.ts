// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-8 structural rules, driven by the shared cross-language case table.
 *
 * `py/test/rfc8_collection_cases.json` pins the rule, message and location
 * bytes both ports must produce; `py/test/test_rfc8_validation.py` drives
 * the identical file.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  validateCollection,
  ValidationError,
  ValidationLevel,
} from "../src/mod.ts";
import { isRfc8NodeModel } from "../src/types/supported_versions.ts";

interface SharedCase {
  name: string;
  ome: Record<string, unknown>;
  version?: string;
  ok?: boolean;
  rule?: string;
  message?: string;
  location?: string;
}

const casesUrl = new URL(
  "../../py/test/rfc8_collection_cases.json",
  import.meta.url,
);
const CASES: SharedCase[] = JSON.parse(await Deno.readTextFile(casesUrl)).cases;

for (const testCase of CASES) {
  Deno.test(`shared case: ${testCase.name}`, () => {
    if (testCase.ok) {
      validateCollection(testCase.ome, testCase.version);
      return;
    }
    const error = assertThrows(
      () => validateCollection(testCase.ome, testCase.version),
      ValidationError,
    );
    assertEquals(error.rule, testCase.rule);
    assertEquals(String(error.message), testCase.message);
    assertEquals(error.location, testCase.location);
  });
}

Deno.test("every rule is exercised by the shared cases", () => {
  const exercised = new Set(
    CASES.filter((testCase) => testCase.rule !== undefined).map(
      (testCase) => testCase.rule,
    ),
  );
  assertEquals(
    exercised,
    new Set([
      "node-type-required",
      "node-name-required",
      "node-name-unique",
      "node-id-format",
      "node-id-unique",
      "node-nodes-xor-path",
      "path-type-known",
      "coordinate-system-id-required",
      "reference-id-required",
      "reference-path-required",
      "label-value-required",
      "label-color-format",
      "scene-transformations-required",
    ]),
  );
});

Deno.test("isRfc8NodeModel is a dispatch predicate", () => {
  // Unlike the RFC-4 gate, no version is not node-shaped: the predicate says
  // which document shape a version stores, and the orchestrator handles the
  // no-version strictness default itself.
  assertEquals(isRfc8NodeModel("0.9.dev3"), true);
  assertEquals(isRfc8NodeModel("0.9.dev1"), false);
  assertEquals(isRfc8NodeModel("0.6"), false);
  assertEquals(isRfc8NodeModel(undefined), false);
});

Deno.test("validateCollection accepts a parsed node tree", () => {
  const error = assertThrows(
    () =>
      validateCollection(
        {
          type: "collection",
          name: "c",
          nodes: [
            { type: "multiscale", name: "img" },
            { type: "multiscale", name: "img" },
          ],
        },
        "0.9.dev3",
      ),
    ValidationError,
  );
  assertEquals(error.rule, "node-name-unique");
  assertEquals(error.location, "ome.nodes[1]");
});

Deno.test("schema_only level skips the rules", () => {
  const invalid = { type: "collection", name: "c" };
  validateCollection(invalid, undefined, {
    level: ValidationLevel.SchemaOnly,
  });
  assertThrows(() => validateCollection(invalid), ValidationError);
});
