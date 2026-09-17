// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-8 node model: parsing and serialization round trips, mirroring
 * `py/test/test_rfc8_model.py` over the same fixture documents.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  isCollectionNode,
  isPrefixedIdentifier,
  NgffCollection,
  nodeFromOmeDict,
  nodeToOmeDict,
  RFC8_ID_PATTERN,
} from "../src/mod.ts";

const FIXTURES = new URL("../../py/test/fixtures/rfc8/", import.meta.url);

const FIXTURE_FILES = [
  "base_multiscale_collection.json",
  "nested_collection.json",
  "external_collection/parent_collection.json",
  "external_collection/child_collection.json",
  "external_collection/resolved_collection.json",
  "webknossos_inline_multiscale.json",
  "mobie_image_labels_table.json",
];

function omeValue(document: Record<string, unknown>): Record<string, unknown> {
  if ("attributes" in document) {
    return (document.attributes as Record<string, unknown>).ome as Record<
      string,
      unknown
    >;
  }
  return document.ome as Record<string, unknown>;
}

Deno.test("the id pattern matches the RFC production", () => {
  for (const accepted of ["s0", "A-1_b.2", "7fa2f064-62c5-4d55-9e6c"]) {
    assertEquals(RFC8_ID_PATTERN.test(accepted), true, accepted);
  }
  for (const rejected of ["a b", "a/b", "", "é"]) {
    assertEquals(RFC8_ID_PATTERN.test(rejected), false, rejected);
  }
});

Deno.test("prefixed identifier detection", () => {
  assertEquals(isPrefixedIdentifier("myorg:zip"), true);
  assertEquals(isPrefixedIdentifier("webknossos:settings"), true);
  assertEquals(isPrefixedIdentifier("zip"), false);
  assertEquals(isPrefixedIdentifier(":zip"), false);
  assertEquals(isPrefixedIdentifier("myorg:"), false);
});

Deno.test("a collection document parses and reports its version", () => {
  const { node, version } = nodeFromOmeDict({
    version: "0.9.dev3",
    type: "collection",
    name: "c",
    nodes: [],
  });
  assertEquals(version, "0.9.dev3");
  assertEquals(isCollectionNode(node), true);
  assertEquals(node.nodes, []);
});

Deno.test("unknown keys round-trip through extra", () => {
  const document = {
    type: "collection",
    name: "c",
    nodes: [],
    "future-key": { nested: [1, null, "x"] },
  };
  const { node } = nodeFromOmeDict(document);
  assertEquals(node.extra, { "future-key": { nested: [1, null, "x"] } });
  assertEquals(nodeToOmeDict(node), document);
});

Deno.test("attributes are opaque and deep-copied", () => {
  const attributes = { "mobie:grid": "true", nested: { value: null } };
  const { node } = nodeFromOmeDict({
    type: "collection",
    name: "c",
    nodes: [],
    attributes,
  });
  assertEquals(node.attributes, attributes);
  (node.attributes as { nested: { value: unknown } }).nested.value = 1;
  assertEquals(attributes.nested.value, null);
});

Deno.test("the version is not stamped on children", () => {
  const document = nodeToOmeDict(
    {
      type: "collection",
      name: "c",
      nodes: [{ type: "collection", name: "sub", nodes: [] }],
    },
    "0.9.dev3",
  );
  assertEquals(document.version, "0.9.dev3");
  assertEquals(
    "version" in (document.nodes as Record<string, unknown>[])[0],
    false,
  );
});

Deno.test("the parser reports malformed core keys", () => {
  assertThrows(() => nodeFromOmeDict({ name: "c" }), Error, "'type'");
  assertThrows(() => nodeFromOmeDict({ type: "collection" }), Error, "'name'");
  assertThrows(
    () => nodeFromOmeDict({ type: "collection", name: "c", id: 7 }),
    Error,
    "'id'",
  );
  assertThrows(
    () =>
      nodeFromOmeDict({
        type: "collection",
        name: "c",
        nodes: [{ name: "x" }],
      }),
    Error,
    "ome.nodes[0]",
  );
  assertThrows(
    () => nodeFromOmeDict({ type: "collection", name: "c", version: 9 }),
    Error,
    "'version'",
  );
});

Deno.test("node lookup prefers ids", () => {
  const collection = new NgffCollection({
    root: {
      type: "collection",
      name: "c",
      nodes: [
        { type: "multiscale", name: "a", id: "b" },
        { type: "multiscale", name: "b", id: "c" },
      ],
    },
  });
  assertEquals(collection.node("b").name, "a");
  assertEquals(collection.node("a").name, "a");
  assertThrows(() => collection.node("missing"), Error, "missing");
});

for (const fixture of FIXTURE_FILES) {
  Deno.test(`fixture document round-trips: ${fixture}`, async () => {
    const document = JSON.parse(
      await Deno.readTextFile(new URL(fixture, FIXTURES)),
    );
    const ome = omeValue(document);
    const { node, version } = nodeFromOmeDict(ome);
    assertEquals(version, "0.9.dev3");
    assertEquals(nodeToOmeDict(node, version), ome);
  });
}
