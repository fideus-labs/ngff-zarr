// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-8 coordinate attribute views and id-based references, mirroring
 * `py/test/test_rfc8_coordinates.py`.
 */

import { assertEquals } from "@std/assert";
import {
  coordinateSystems,
  coordinateTransformations,
  nodeFromOmeDict,
  nodeToOmeDict,
  type OmeNode,
  setCoordinateSystems,
  setCoordinateTransformations,
} from "../src/mod.ts";
import type { Scale } from "../src/types/zarr_metadata.ts";

function multiscaleNode(): OmeNode {
  const node: OmeNode = { type: "multiscale", name: "img", id: "img" };
  setCoordinateSystems(node, [
    {
      name: "physical",
      id: "physical",
      axes: [
        { name: "y", type: "space", unit: "micrometer" },
        { name: "x", type: "space", unit: "micrometer" },
      ],
    },
  ]);
  setCoordinateTransformations(node, [
    {
      type: "scale",
      scale: [1.0, 2.0],
      input: { id: "s0" },
      output: { id: "physical" },
    },
  ]);
  return node;
}

Deno.test("coordinate attributes round-trip through the document form", () => {
  const node = multiscaleNode();
  const document = nodeToOmeDict(node, "0.9.dev3");
  const attributes = document.attributes as Record<string, unknown>;
  const systems = attributes.coordinateSystems as Record<string, unknown>[];
  assertEquals(systems[0].id, "physical");
  const transforms = attributes.coordinateTransformations as Record<
    string,
    unknown
  >[];
  assertEquals(transforms[0].input, { id: "s0" });
  assertEquals(transforms[0].output, { id: "physical" });

  const { node: parsed } = nodeFromOmeDict(document);
  const parsedSystems = coordinateSystems(parsed);
  assertEquals(parsedSystems[0].id, "physical");
  assertEquals(parsedSystems[0].axes[0].unit, "micrometer");
  const parsedTransforms = coordinateTransformations(parsed);
  assertEquals((parsedTransforms[0] as Scale).scale, [1.0, 2.0]);
  assertEquals(parsedTransforms[0].input?.id, "s0");
  assertEquals(parsedTransforms[0].output?.id, "physical");
});

Deno.test("a nameless coordinate system round-trips", () => {
  const node: OmeNode = { type: "multiscale", name: "img" };
  setCoordinateSystems(node, [
    {
      name: "",
      id: "physical",
      axes: [{ name: "x", type: "space", unit: undefined }],
    },
  ]);
  const serialized = (node.attributes?.coordinateSystems as Record<
    string,
    unknown
  >[])[0];
  assertEquals("name" in serialized, false);
  const systems = coordinateSystems(node);
  assertEquals(systems[0].id, "physical");
  assertEquals(systems[0].name, "");
});

Deno.test("sequence transformations parse from attributes", () => {
  const node: OmeNode = {
    type: "singlescale",
    name: "s0",
    id: "s0",
    attributes: {
      coordinateTransformations: [
        {
          type: "sequence",
          transformations: [
            { type: "scale", scale: [1.0, 1.0] },
            { type: "translation", translation: [0.0, 5.0] },
          ],
          input: { id: "s0" },
          output: { id: "physical" },
        },
      ],
    },
  };
  const systems = [
    {
      name: "physical",
      id: "physical",
      axes: [
        { name: "y", type: "space" as const, unit: undefined },
        { name: "x", type: "space" as const, unit: undefined },
      ],
    },
  ];
  const transforms = coordinateTransformations(node, systems);
  assertEquals(transforms[0].type, "sequence");
  assertEquals(transforms[0].input?.id, "s0");
  assertEquals(transforms[0].output?.id, "physical");
});
