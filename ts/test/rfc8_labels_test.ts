// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-8 labels attribute, mirroring `py/test/test_rfc8_labels.py`.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  isLabelMap,
  labels,
  nodeFromOmeDict,
  nodeToOmeDict,
  type OmeNode,
  setLabels,
  validateCollection,
  ValidationError,
} from "../src/mod.ts";

function labelMap(): OmeNode {
  const node: OmeNode = {
    type: "multiscale",
    name: "nuclei",
    path: { type: "zarr", path: "./nuclei" },
  };
  setLabels(node, {
    labelAttributes: [
      { labelValue: 1, color: [255, 0, 0, 255] },
      { labelValue: 2, extra: { "ngio:class": "nucleus" } },
    ],
    source: [{ id: "raw" }],
  });
  return node;
}

Deno.test("labels round-trip through the document", () => {
  const node = labelMap();
  const serialized = node.attributes?.labels as Record<string, unknown>;
  assertEquals((serialized.labelAttributes as unknown[])[0], {
    labelValue: 1,
    color: [255, 0, 0, 255],
  });
  assertEquals((serialized.labelAttributes as unknown[])[1], {
    labelValue: 2,
    "ngio:class": "nucleus",
  });
  assertEquals(serialized.source, [{ id: "raw" }]);

  const { node: parsed } = nodeFromOmeDict(nodeToOmeDict(node));
  assertEquals(isLabelMap(parsed), true);
  const value = labels(parsed);
  assertEquals(value?.labelAttributes?.[0].color, [255, 0, 0, 255]);
  assertEquals(value?.labelAttributes?.[1].extra, {
    "ngio:class": "nucleus",
  });
  assertEquals(value?.source, [{ id: "raw" }]);
});

Deno.test("an empty labels object denotes a label map", () => {
  const node: OmeNode = {
    type: "multiscale",
    name: "seg",
    path: { type: "zarr", path: "./seg" },
  };
  assertEquals(isLabelMap(node), false);
  assertEquals(labels(node), undefined);
  setLabels(node, {});
  assertEquals(node.attributes?.labels, {});
  assertEquals(isLabelMap(node), true);
  assertEquals(labels(node), {});

  setLabels(node, undefined);
  assertEquals(isLabelMap(node), false);
});

Deno.test("m:n label relations validate", () => {
  const nodes: OmeNode[] = [];
  for (const index of [0, 1]) {
    nodes.push({
      type: "multiscale",
      name: `img${index}`,
      id: `img${index}`,
      path: { type: "zarr", path: `./img${index}` },
    });
  }
  for (const index of [0, 1]) {
    const node: OmeNode = {
      type: "multiscale",
      name: `seg${index}`,
      id: `seg${index}`,
      path: { type: "zarr", path: `./seg${index}` },
    };
    setLabels(node, { source: [{ id: "img0" }, { id: "img1" }] });
    nodes.push(node);
  }
  validateCollection(
    { type: "collection", name: "study", nodes: nodes.map((node) => node) },
    "0.9.dev3",
  );
});

Deno.test("label rules fire", () => {
  const node: OmeNode = {
    type: "multiscale",
    name: "seg",
    path: { type: "zarr", path: "./seg" },
    attributes: { labels: { labelAttributes: [{ color: [1, 2, 3, 4] }] } },
  };
  let error = assertThrows(
    () => validateCollection(node, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "label-value-required");

  node.attributes = {
    labels: { labelAttributes: [{ labelValue: 1, color: [1, 2, 3] }] },
  };
  error = assertThrows(
    () => validateCollection(node, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "label-color-format");

  node.attributes = { labels: { source: [{ id: "elsewhere" }] } };
  error = assertThrows(
    () => validateCollection(node, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "reference-path-required");
});

Deno.test("a null labels value is rejected", () => {
  const node: OmeNode = {
    type: "multiscale",
    name: "nuclei",
    nodes: [],
    attributes: { labels: null },
  };
  assertEquals(isLabelMap(node), true);
  assertThrows(() => labels(node), Error, "must be an object");
});

Deno.test("extra keys do not override typed fields", () => {
  const node: OmeNode = { type: "multiscale", name: "nuclei", nodes: [] };
  setLabels(node, {
    labelAttributes: [
      {
        labelValue: 1,
        color: [255, 0, 0, 255],
        extra: { labelValue: 99, "myorg:note": "kept" },
      },
    ],
  });
  const serialized = (
    (node.attributes?.labels as Record<string, unknown>)
      .labelAttributes as Record<string, unknown>[]
  )[0];
  assertEquals(serialized.labelValue, 1);
  assertEquals(serialized.color, [255, 0, 0, 255]);
  assertEquals(serialized["myorg:note"], "kept");
});
