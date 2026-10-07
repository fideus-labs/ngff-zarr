// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Scenes: images that share a spatial relationship (OME-Zarr 0.6, GH #563).
 * Mirrors `py/test/test_scene.py`.
 */
import { assertEquals, assertRejects } from "@std/assert";
import * as zarr from "zarrita";
import { FileSystemStore } from "@zarrita/storage";
import {
  type Bijection,
  type CoordinateSystem,
  type Displacements,
  fromOmeZarr,
  NgffMultiscales,
  NgffScene,
  toMultiscales,
  toNgffImage,
  toOmeZarr,
  type TransformSequence,
  type Translation,
  type V06Transform,
} from "../src/mod.ts";
import { sceneFromOmeValue } from "../src/io/scene.ts";

const FIXTURES = new URL("../../py/test/fixtures/scene/", import.meta.url);

async function tile(
  seed: number,
): Promise<{ data: Uint8Array; multiscales: NgffMultiscales }> {
  const data = new Uint8Array(16 * 16);
  for (let i = 0; i < data.length; i++) {
    data[i] = (i * 31 + seed * 17) % 256;
  }
  const image = await toNgffImage(data, {
    dims: ["y", "x"],
    shape: [16, 16],
    scale: { y: 0.5, x: 0.5 },
  });
  return {
    data,
    multiscales: await toMultiscales(image, { scaleFactors: [2] }),
  };
}

function world(): CoordinateSystem {
  return {
    name: "world",
    axes: [
      { name: "y", type: "space", unit: "micrometer" },
      { name: "x", type: "space", unit: "micrometer" },
    ],
  };
}

function toWorld(path: string, offset: [number, number]): Translation {
  return {
    type: "translation",
    translation: offset,
    input: { path, name: "intrinsic" },
    output: { name: "world" },
    name: `${path} to world`,
  };
}

async function tilesScene(
  paths: string[] = ["tile_0", "tile_1"],
): Promise<{ pixels: Record<string, Uint8Array>; scene: NgffScene }> {
  const pixels: Record<string, Uint8Array> = {};
  const images: Record<string, NgffMultiscales> = {};
  const transforms: V06Transform[] = [];
  for (const [index, path] of paths.entries()) {
    const { data, multiscales } = await tile(index);
    pixels[path] = data;
    images[path] = multiscales;
    transforms.push(toWorld(path, [0, 8 * index]));
  }
  return {
    pixels,
    scene: new NgffScene({
      images,
      coordinateTransformations: transforms,
      coordinateSystems: [world()],
    }),
  };
}

async function rootDocument(store: string): Promise<Record<string, unknown>> {
  return JSON.parse(await Deno.readTextFile(`${store}/zarr.json`));
}

async function withTempDir(
  body: (dir: string) => Promise<void>,
): Promise<void> {
  const dir = await Deno.makeTempDir();
  try {
    await body(dir);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
}

async function pixelsOf(multiscales: NgffMultiscales): Promise<Uint8Array> {
  const chunk = await zarr.get(multiscales.images[0].data);
  return chunk.data as Uint8Array;
}

Deno.test("scene round trip", async () => {
  await withTempDir(async (dir) => {
    const { pixels, scene } = await tilesScene();
    const store = `${dir}/tiles.ome.zarr`;
    await toOmeZarr(store, scene);

    const document = await rootDocument(store);
    const attributes = document.attributes as Record<string, unknown>;
    const ome = attributes.ome as Record<string, unknown>;
    assertEquals(ome.version, "0.6");
    const written = ome.scene as Record<string, unknown>;
    assertEquals(
      (written.coordinateSystems as Array<{ name: string }>)[0].name,
      "world",
    );
    assertEquals((written.coordinateTransformations as unknown[])[1], {
      type: "translation",
      name: "tile_1 to world",
      input: { path: "tile_1", name: "intrinsic" },
      output: { name: "world" },
      translation: [0, 8],
    });
    const consolidated = document.consolidated_metadata as {
      metadata: Record<string, unknown>;
    };
    const datasetPath = scene.images.tile_0.metadata.datasets[0].path;
    assertEquals(`tile_0/${datasetPath}` in consolidated.metadata, true);

    const read = await fromOmeZarr(store, { kind: "scene", validate: true });
    assertEquals(Object.keys(read.images), ["tile_0", "tile_1"]);
    for (const [path, data] of Object.entries(pixels)) {
      assertEquals(await pixelsOf(read.images[path]), data);
      assertEquals(
        read.images[path].metadata.coordinateSystems![0].name,
        "intrinsic",
      );
    }
    assertEquals(read.coordinateSystems, [world()]);
    assertEquals(
      read.coordinateTransformations,
      scene.coordinateTransformations,
    );
  });
});

Deno.test("nested image paths get their ancestor groups", async () => {
  await withTempDir(async (dir) => {
    const { pixels, scene } = await tilesScene([
      "sample/instrument1",
      "sample/instrument2",
    ]);
    const store = `${dir}/scene.ome.zarr`;
    await toOmeZarr(store, scene);

    const root = zarr.root(new FileSystemStore(store));
    const sample = await zarr.open(root.resolve("sample"), { kind: "group" });
    assertEquals(sample.attrs, {});
    const read = await fromOmeZarr(store, { kind: "scene" });
    assertEquals(Object.keys(read.images), Object.keys(pixels));
    assertEquals(
      await pixelsOf(read.images["sample/instrument1"]),
      pixels["sample/instrument1"],
    );
  });
});

Deno.test("overwrite false keeps the root attributes", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    const store = `${dir}/tiles.ome.zarr`;
    await zarr.create(zarr.root(new FileSystemStore(store)), {
      attributes: { "myorg:note": "kept" },
    });
    await toOmeZarr(store, scene, { overwrite: false });
    const attributes = (await rootDocument(store)).attributes as Record<
      string,
      unknown
    >;
    assertEquals(attributes["myorg:note"], "kept");
    assertEquals("scene" in (attributes.ome as Record<string, unknown>), true);
  });
});

function tile0To(name: string, path?: string): Translation {
  return {
    type: "translation",
    translation: [0, 0],
    input: { path: "tile_0", name: "intrinsic" },
    output: path === undefined ? { name } : { path, name },
  };
}

const REFUSALS: Array<[string, V06Transform[], string]> = [
  ["no transformation", [], "at least one coordinate transformation"],
  ["unknown image", [toWorld("tile_9", [0, 0])], "does not contain"],
  ["unknown scene system", [tile0To("universe")], "does not declare"],
  ["unknown image system", [tile0To("physical", "tile_1")], "which declares"],
  [
    "nameless reference",
    [{
      type: "translation",
      translation: [0, 0],
      input: { path: "tile_0" },
      output: { name: "world" },
    }],
    "must name a coordinate system",
  ],
  ["disconnected graph", [toWorld("tile_0", [0, 0])], "unconnected groups"],
];

for (const [label, transforms, message] of REFUSALS) {
  Deno.test(`write refuses a scene with ${label}`, async () => {
    await withTempDir(async (dir) => {
      const { scene } = await tilesScene();
      scene.coordinateTransformations = transforms;
      const store = `${dir}/tiles.ome.zarr`;
      await assertRejects(() => toOmeZarr(store, scene), Error, message);
      await assertRejects(() => Deno.stat(store), Deno.errors.NotFound);
    });
  });
}

Deno.test("write refuses duplicate or empty system names", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    scene.coordinateSystems = [world(), world()];
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ome.zarr`, scene),
      Error,
      "non-empty and unique",
    );
  });
});

Deno.test("scene axis model follows the version", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    scene.coordinateSystems![0].axes = [..."abcdef"].map((name) => ({
      name,
      type: "space",
      unit: undefined,
    }));
    await assertRejects(
      () => toOmeZarr(`${dir}/v06.ome.zarr`, scene),
      Error,
      "scene.coordinateSystems[0].axes",
    );
    await toOmeZarr(`${dir}/dev1.ome.zarr`, scene, { version: "0.9.dev1" });
    const read = await fromOmeZarr(`${dir}/dev1.ome.zarr`, {
      kind: "scene",
      validate: true,
    });
    assertEquals(
      read.coordinateSystems![0].axes.map((axis) => axis.name),
      [..."abcdef"],
    );
  });
});

Deno.test("write refuses vectors that do not span their systems", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    (scene.coordinateTransformations[0] as Translation).translation = [0, 0, 0];
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ome.zarr`, scene),
      Error,
      "3 translation values for the 2 axes",
    );
  });
});

Deno.test("write refuses transforms the reader rejects", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    scene.coordinateTransformations[0] = {
      type: "mapAxis",
      mapAxis: [0, 1, 2],
      input: { path: "tile_0", name: "intrinsic" },
      output: { name: "world" },
    };
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ome.zarr`, scene),
      Error,
      "cannot read back",
    );
  });
});

Deno.test("scene with a displacement field between two images", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    const fieldPath = "coordinateTransformations/dfield";
    const warp: Displacements = {
      type: "displacements",
      path: fieldPath,
      interpolation: "linear",
      input: { path: "tile_0", name: "intrinsic" },
      output: { path: "tile_1", name: "intrinsic" },
    };
    scene.coordinateTransformations.push(warp);
    const store = `${dir}/registered.ome.zarr`;

    await assertRejects(
      () => toOmeZarr(store, scene),
      Error,
      "write those first",
    );
    await assertRejects(
      () => toOmeZarr(store, scene, { overwrite: false }),
      Error,
      "which the store does not hold",
    );
    await assertRejects(() => Deno.stat(store), Deno.errors.NotFound);

    const field = await toNgffImage(new Float32Array(2 * 16 * 16), {
      dims: ["c", "y", "x"],
      shape: [2, 16, 16],
      scale: { c: 1, y: 0.5, x: 0.5 },
      translation: { c: 0, y: 0, x: 0 },
      axesTypes: { c: "displacement" },
    });
    await toOmeZarr(
      `${store}/${fieldPath}`,
      await toMultiscales(field, {
        scaleFactors: [],
      }),
      { version: "0.6" },
    );
    await toOmeZarr(store, scene, { overwrite: false });
    const consolidated = (await rootDocument(store)).consolidated_metadata as {
      metadata: Record<string, unknown>;
    };
    assertEquals("coordinateTransformations" in consolidated.metadata, true);
    assertEquals(fieldPath in consolidated.metadata, true);

    const read = await fromOmeZarr(store, { kind: "scene", validate: true });
    assertEquals(read.coordinateTransformations[2], warp);
    const fieldRead = await fromOmeZarr(`${store}/${fieldPath}`, {
      version: "0.6",
    });
    assertEquals(
      fieldRead.metadata.coordinateSystems![0].axes.map((axis) => axis.type),
      ["displacement", "space", "space"],
    );
  });
});

Deno.test("write refuses paths outside the scene", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene(["../tile_0", "tile_1"]);
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ome.zarr`, scene),
      Error,
      "relative path below the scene group",
    );
  });
});

Deno.test("write refuses versions before 0.6", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ome.zarr`, scene, { version: "0.5" }),
      Error,
      "0.6",
    );
  });
});

Deno.test("read validates references against the store", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    const store = `${dir}/tiles.ome.zarr`;
    await toOmeZarr(store, scene, { consolidateMetadata: false });
    const document = await rootDocument(store);
    const ome = (document.attributes as Record<string, unknown>).ome as Record<
      string,
      unknown
    >;
    delete (ome.scene as Record<string, unknown>).coordinateSystems;
    await Deno.writeTextFile(`${store}/zarr.json`, JSON.stringify(document));

    const read = await fromOmeZarr(store, { kind: "scene" });
    assertEquals(read.coordinateSystems, undefined);
    await assertRejects(
      () => fromOmeZarr(store, { kind: "scene", validate: true }),
      Error,
      "does not declare",
    );
  });
});

for (const path of ["../tile_0", "..\\tile_0", "%2e%2e/tile_0", "/tile_0"]) {
  Deno.test(`read refuses the image path ${path} outside the scene`, async () => {
    await withTempDir(async (dir) => {
      const { scene } = await tilesScene();
      const store = `${dir}/tiles.ome.zarr`;
      await toOmeZarr(store, scene, { consolidateMetadata: false });
      const document = await rootDocument(store);
      const ome = (document.attributes as Record<string, unknown>)
        .ome as Record<string, unknown>;
      const transforms = (ome.scene as Record<string, unknown>)
        .coordinateTransformations as Array<{ input: { path: string } }>;
      transforms[0].input.path = path;
      await Deno.writeTextFile(`${store}/zarr.json`, JSON.stringify(document));
      await assertRejects(
        () => fromOmeZarr(store, { kind: "scene" }),
        Error,
        "relative path below the scene group",
      );
    });
  });
}

Deno.test("the kind option selects what a store holds", async () => {
  await withTempDir(async (dir) => {
    const { multiscales } = await tile(0);
    const image = `${dir}/image.ome.zarr`;
    await toOmeZarr(image, multiscales, { version: "0.6" });
    assertEquals((await fromOmeZarr(image)) instanceof NgffMultiscales, true);
    await assertRejects(
      () => fromOmeZarr(image, { kind: "scene" }),
      Error,
      "holds no scene",
    );

    const { pixels, scene } = await tilesScene();
    const store = `${dir}/tiles.ome.zarr`;
    await toOmeZarr(store, scene);
    await assertRejects(() => fromOmeZarr(store), Error, 'kind: "scene"');
    const tile1 = await fromOmeZarr(`${store}/tile_1`);
    assertEquals(tile1 instanceof NgffMultiscales, true);
    assertEquals(await pixelsOf(tile1), pixels.tile_1);
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ozx`, scene),
      Error,
      "directory path",
    );
  });
});

async function upstream(name: string): Promise<Record<string, unknown>> {
  const document = JSON.parse(
    await Deno.readTextFile(new URL(name, FIXTURES)),
  );
  const attributes = document.attributes as Record<string, unknown>;
  return attributes.ome as Record<string, unknown>;
}

Deno.test("upstream stitching example", async () => {
  const ome = await upstream("scene_stitching.json");
  const { coordinateSystems, coordinateTransformations } = sceneFromOmeValue(
    ome.scene as Record<string, unknown>,
    "0.6",
  );
  assertEquals(coordinateSystems!.map((system) => system.name), ["world"]);
  assertEquals(
    coordinateSystems![0].axes.map((axis) => axis.unit),
    ["micrometer", "micrometer"],
  );
  assertEquals(
    coordinateTransformations.map((transform) => transform.input!.path),
    ["tile_0", "tile_1", "tile_2", "tile_3"],
  );
  assertEquals(
    coordinateTransformations.every((t) =>
      t.input!.name === "physical" && t.output!.name === "world"
    ),
    true,
  );
  assertEquals(
    (coordinateTransformations[3] as Translation).translation,
    [276, 348],
  );
});

Deno.test("upstream registration example", async () => {
  const ome = await upstream("scene_registration.json");
  const { coordinateSystems, coordinateTransformations } = sceneFromOmeValue(
    ome.scene as Record<string, unknown>,
    "0.6",
  );
  assertEquals(coordinateSystems, undefined);
  assertEquals(coordinateTransformations.length, 1);
  const bijection = coordinateTransformations[0] as Bijection;
  assertEquals(bijection.type, "bijection");
  assertEquals(bijection.input, { path: "JRC2018F", name: "physical" });
  assertEquals(bijection.output, { path: "FCWB", name: "physical" });
  const forward = bijection.forward as TransformSequence;
  assertEquals(forward.type, "sequence");
  const field = forward.transformations[0] as Displacements;
  assertEquals(field.type, "displacements");
  assertEquals(field.path, "coordinateTransformations/dfield");
  assertEquals(field.interpolation, "linear");
});
