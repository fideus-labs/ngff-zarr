// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Scene metadata, mirroring `py/test/test_rfc8_scene.py` over the same
 * fixture documents.
 */

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import {
  type OmeNode,
  readSceneFromAttrs,
  type Scene,
  scene,
  setScene,
  validateCollection,
  ValidationError,
} from "../src/mod.ts";
import { sceneFromOmeValue, sceneToOmeValue } from "../src/utils/rfc8_scene.ts";

const FIXTURES = new URL(
  "../../py/test/fixtures/rfc8/scene/",
  import.meta.url,
);

function tilesScene(): Scene {
  return {
    coordinateSystems: [
      {
        name: "world",
        id: "world",
        axes: [
          { name: "y", type: "space", unit: "micrometer" },
          { name: "x", type: "space", unit: "micrometer" },
        ],
      },
    ],
    coordinateTransformations: [
      {
        type: "translation",
        translation: [0.0, 100.0],
        input: { id: "physical", path: "./tile_0.zarr" },
        output: { id: "world" },
      },
      {
        type: "translation",
        translation: [100.0, 0.0],
        input: { id: "physical", path: "./tile_1.zarr" },
        output: { id: "world" },
      },
    ],
  };
}

Deno.test("the scene attribute round-trips", () => {
  const collection: OmeNode = { type: "collection", name: "tiles", nodes: [] };
  setScene(collection, tilesScene());
  const serialized = collection.attributes?.scene as Record<string, unknown>;
  assertEquals(
    (serialized.coordinateSystems as Record<string, unknown>[])[0].id,
    "world",
  );
  assertEquals(
    (serialized.coordinateTransformations as Record<string, unknown>[])[0]
      .input,
    { path: "./tile_0.zarr", id: "physical" },
  );

  const parsed = scene(collection);
  assertEquals(parsed?.coordinateSystems?.[0].id, "world");
  assertEquals(parsed?.coordinateTransformations[0].input?.id, "physical");
  assertEquals(
    parsed?.coordinateTransformations[0].input?.path,
    "./tile_0.zarr",
  );

  setScene(collection, undefined);
  assertEquals(scene(collection), undefined);
});

Deno.test("a valid scene passes validation", () => {
  const collection: OmeNode = { type: "collection", name: "tiles", nodes: [] };
  setScene(collection, tilesScene());
  validateCollection(collection, "0.9.dev3");
});

Deno.test("scene validation rules fire", () => {
  const collection: OmeNode = {
    type: "collection",
    name: "tiles",
    nodes: [],
    attributes: { scene: {} },
  };
  let error = assertThrows(
    () => validateCollection(collection, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "scene-transformations-required");

  collection.attributes = {
    scene: {
      coordinateTransformations: [
        {
          type: "identity",
          input: { name: "physical" },
          output: { id: "world" },
        },
      ],
    },
  };
  error = assertThrows(
    () => validateCollection(collection, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "reference-id-required");

  collection.attributes = {
    scene: {
      coordinateTransformations: [
        {
          type: "identity",
          input: { id: "physical" },
          output: { id: "world" },
        },
      ],
    },
  };
  error = assertThrows(
    () => validateCollection(collection, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "reference-path-required");
});

Deno.test("both storage shapes read through readSceneFromAttrs", () => {
  // The RFC-5 0.6-family shape: ome.scene.
  const v06Attrs = {
    ome: {
      version: "0.9.dev1",
      scene: {
        coordinateTransformations: [
          {
            type: "translation",
            translation: [0.0, 100.0],
            input: { name: "physical", path: "tile_0" },
            output: { name: "world" },
          },
        ],
      },
    },
  };
  const fromV06 = readSceneFromAttrs(v06Attrs);
  assertEquals(fromV06?.coordinateTransformations[0].input?.name, "physical");

  // The RFC-8 shape: the scene attribute of a node document.
  const collection: OmeNode = { type: "collection", name: "tiles", nodes: [] };
  setScene(collection, tilesScene());
  const dev3Attrs = {
    ome: {
      version: "0.9.dev3",
      type: "collection",
      name: "tiles",
      nodes: [],
      attributes: collection.attributes,
    },
  };
  const fromDev2 = readSceneFromAttrs(dev3Attrs);
  assertEquals(
    fromDev2?.coordinateTransformations[1].input?.path,
    "./tile_1.zarr",
  );
});

for (const fixture of ["scene_registration.json", "scene_stitching.json"]) {
  Deno.test(`the upstream scene example parses: ${fixture}`, async () => {
    const document = JSON.parse(
      await Deno.readTextFile(new URL(fixture, FIXTURES)),
    );
    const parsed = readSceneFromAttrs(document.attributes);
    assertEquals(parsed !== undefined, true);
    const first = parsed!.coordinateTransformations[0];
    assertEquals(typeof first.input?.name, "string");
  });
}

Deno.test("typed reference paths survive scene parsing", () => {
  const parsed = sceneFromOmeValue({
    coordinateTransformations: [
      {
        type: "translation",
        translation: [0.0, 100.0],
        input: {
          id: "physical",
          path: { type: "zarr", path: "./tile_0.zarr" },
        },
        output: { id: "world" },
      },
    ],
  }, "0.9.dev3");
  assertEquals(parsed.coordinateTransformations[0].input?.path, {
    type: "zarr",
    path: "./tile_0.zarr",
  });
  const serialized = sceneToOmeValue(parsed);
  const transforms = serialized.coordinateTransformations as Record<
    string,
    unknown
  >[];
  assertEquals((transforms[0].input as Record<string, unknown>).path, {
    type: "zarr",
    path: "./tile_0.zarr",
  });
});

Deno.test("id-only references refuse the 0.6 scene shape", () => {
  const tiles = sceneFromOmeValue({
    coordinateTransformations: [
      {
        type: "translation",
        translation: [0.0, 100.0],
        input: { id: "physical", path: "./tile_0.zarr" },
        output: { id: "world" },
      },
    ],
  }, "0.9.dev3");
  let message = "";
  try {
    sceneToOmeValue(tiles, "0.6");
  } catch (error) {
    message = String(error);
  }
  assertStringIncludes(message, '"0.9.dev3"');
});

Deno.test("typed reference paths refuse the 0.6 scene shape", () => {
  const tiles = sceneFromOmeValue({
    coordinateTransformations: [
      {
        type: "translation",
        translation: [0.0, 100.0],
        input: {
          name: "physical",
          path: { type: "zarr", path: "./tile_0.zarr" },
        },
        output: { name: "world" },
      },
    ],
  }, "0.9.dev3");
  let message = "";
  try {
    sceneToOmeValue(tiles, "0.6");
  } catch (error) {
    message = String(error);
  }
  assertStringIncludes(message, "typed path object");
});
