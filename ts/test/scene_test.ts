// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Scenes: images that share a spatial relationship (OME-Zarr 0.6, GH #563).
 * Mirrors `py/test/test_scene.py`.
 */
import { assertEquals, assertRejects } from "@std/assert";
import * as zarr from "zarrita";
import { FileSystemStore, ZipFileStore } from "@zarrita/storage";
import {
  type Affine,
  type Bijection,
  type CoordinateSystem,
  type Displacements,
  fromOmeZarr,
  getZipFileList,
  NgffMultiscales,
  NgffScene,
  readOzxVersion,
  storeToZip,
  toMultiscales,
  toNgffImage,
  toOmeZarr,
  toOmeZarrOzxData,
  type TransformSequence,
  type Translation,
  type V06Transform,
} from "../src/mod.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { fromOmeZarr as fromOmeZarrBrowser } from "../src/io/from_ngff_zarr-browser.ts";
import {
  storeToZip as storeToZipBrowser,
  toOmeZarr as toOmeZarrBrowser,
  toOmeZarrOzx as toOmeZarrOzxBrowser,
} from "../src/io/to_ngff_zarr-browser.ts";
import { sceneFromOmeValue } from "../src/io/scene_common.ts";

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
    // The field's own arrays too, as the Python writer, which consolidates
    // the whole store, lists them.
    assertEquals(`${fieldPath}/scale0` in consolidated.metadata, true);

    const read = await fromOmeZarr(store, { kind: "scene", validate: true });
    assertEquals(read.coordinateTransformations[2], warp);
    const fieldRead = await fromOmeZarr(`${store}/${fieldPath}`, {
      version: "0.6",
    });
    assertEquals(
      fieldRead.metadata.coordinateSystems![0].axes.map((axis) => axis.type),
      ["displacement", "space", "space"],
    );

    // The staged directory packs into an archive that reads back the same.
    const archivePath = `${dir}/registered.ozx`;
    await storeToZip(store, archivePath);
    const zip = await Deno.readFile(archivePath);
    assertEquals(readOzxVersion(zip), "0.6");
    const archive = ZipFileStore.fromBlob(new Blob([zip as BlobPart]));
    const fromArchive = await fromOmeZarr(archive, {
      kind: "scene",
      validate: true,
    });
    assertEquals(fromArchive.coordinateTransformations[2], warp);
    assertEquals(
      (await fromOmeZarr(archive, { path: fieldPath })).images[0].data.shape,
      [2, 16, 16],
    );
  });
});

/** The 2 x 16 x 16 displacement field the tile scenes' warp references. */
async function displacementField(): Promise<NgffMultiscales> {
  const field = await toNgffImage(
    new Float32Array(2 * 16 * 16).map((_, i) => i / 8),
    {
      dims: ["c", "y", "x"],
      shape: [2, 16, 16],
      scale: { c: 1, y: 0.5, x: 0.5 },
      translation: { c: 0, y: 0, x: 0 },
      axesTypes: { c: "displacement" },
    },
  );
  return await toMultiscales(field, { scaleFactors: [] });
}

const STAGING = [
  ["Node", toOmeZarr, storeToZip],
  ["browser", toOmeZarrBrowser, storeToZipBrowser],
] as const;

for (const [module, write, pack] of STAGING) {
  Deno.test(`${module}: a scene with a displacement field is staged in a MemoryStore and packed`, async () => {
    const { pixels, scene } = await tilesScene();
    const fieldPath = "coordinateTransformations/dfield";
    const warp: Displacements = {
      type: "displacements",
      path: fieldPath,
      interpolation: "linear",
      input: { path: "tile_0", name: "intrinsic" },
      output: { path: "tile_1", name: "intrinsic" },
    };
    scene.coordinateTransformations.push(warp);
    const field = await displacementField();

    const store: MemoryStore = new Map();
    await write(store, field, { version: "0.6", path: fieldPath });
    await write(store, scene, { overwrite: false });
    const zip = pack(store);
    assertEquals(readOzxVersion(zip), "0.6");
    assertEquals(getZipFileList(zip)[0], "zarr.json");

    const archive = ZipFileStore.fromBlob(new Blob([zip as BlobPart]));
    const nodes = consolidatedNodes(
      await archiveDocument(archive, "/zarr.json"),
    );
    const fieldDataset = field.metadata.datasets[0].path;
    for (
      const node of [
        "coordinateTransformations",
        fieldPath,
        `${fieldPath}/${fieldDataset}`,
        "tile_0",
      ]
    ) {
      assertEquals(nodes.includes(node), true, node);
    }
    for (const read of [fromOmeZarr, fromOmeZarrBrowser]) {
      const readScene = await read(archive, { kind: "scene", validate: true });
      await assertScenePixels(readScene, pixels);
      assertEquals(readScene.coordinateTransformations[2], warp);
      const readField = await read(archive, { path: warp.path });
      assertEquals(
        readField.metadata.coordinateSystems![0].axes.map((axis) => axis.type),
        ["displacement", "space", "space"],
      );
      assertEquals(
        (await zarr.get(readField.images[0].data)).data,
        (await zarr.get(field.images[0].data)).data,
      );
    }
  });
}

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

Deno.test("a requested version is checked against the stored one", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    const store = `${dir}/tiles.ome.zarr`;
    await toOmeZarr(store, scene);
    await assertRejects(
      () =>
        fromOmeZarr(store, { kind: "scene", validate: true, version: "0.5" }),
      Error,
      "Expected OME-Zarr version 0.5, but found 0.6",
    );
    const read = await fromOmeZarr(store, {
      kind: "scene",
      validate: true,
      version: "0.6",
    });
    assertEquals(Object.keys(read.images), ["tile_0", "tile_1"]);
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

for (
  const path of [
    "../tile_0",
    "..\\tile_0",
    "%2e%2e/tile_0",
    "%2e%2e%2ftile_0",
    "/tile_0",
  ]
) {
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
  });
});

/** Each image of `read` holds the pixels written at its path, and no other image is there. */
async function assertScenePixels(
  read: NgffScene,
  pixels: Record<string, Uint8Array>,
): Promise<void> {
  assertEquals(Object.keys(read.images).sort(), Object.keys(pixels).sort());
  for (const [path, data] of Object.entries(pixels)) {
    assertEquals(await pixelsOf(read.images[path]), data);
  }
}

function documentAt(
  store: MemoryStore,
  key: string,
): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(store.get(key)));
}

async function archiveDocument(
  archive: ZipFileStore,
  key: `/${string}`,
): Promise<Record<string, unknown>> {
  return JSON.parse(new TextDecoder().decode(await archive.get(key)));
}

function consolidatedNodes(document: Record<string, unknown>): string[] {
  return Object.keys(
    (document.consolidated_metadata as { metadata: Record<string, unknown> })
      .metadata,
  );
}

/** Records every progress report, for the last one and the total. */
function progressLog(): {
  reports: Array<[number, number]>;
  onProgress: (completed: number, total: number) => void;
} {
  const reports: Array<[number, number]> = [];
  return {
    reports,
    onProgress: (completed, total) => reports.push([completed, total]),
  };
}

Deno.test("the browser writer writes a scene to a MemoryStore", async () => {
  const { pixels, scene } = await tilesScene(["tile_0", "sample/tile_1"]);
  const store: MemoryStore = new Map();
  const progress = progressLog();
  await toOmeZarrBrowser(store, scene, { onProgress: progress.onProgress });

  const document = documentAt(store, "/zarr.json");
  const ome = (document.attributes as Record<string, unknown>).ome as Record<
    string,
    unknown
  >;
  assertEquals(ome.version, "0.6");
  assertEquals(
    (ome.scene as Record<string, unknown>).coordinateTransformations,
    scene.coordinateTransformations.map((transform) => ({ ...transform })),
  );
  assertEquals(documentAt(store, "/sample/zarr.json").attributes, {});
  const datasetPath = scene.images.tile_0.metadata.datasets[0].path;
  const nodes = consolidatedNodes(document);
  for (
    const node of [
      "tile_0",
      `tile_0/${datasetPath}`,
      "sample",
      "sample/tile_1",
      `sample/tile_1/${datasetPath}`,
    ]
  ) {
    assertEquals(nodes.includes(node), true, node);
  }

  // One count across both images: every report has the same total, which
  // the last one reaches.
  const [, total] = progress.reports.at(-1)!;
  assertEquals(progress.reports.at(-1), [total, total]);
  assertEquals(progress.reports.every(([, each]) => each === total), true);

  for (const read of [fromOmeZarrBrowser, fromOmeZarr]) {
    const scene_ = await read(store, { kind: "scene", validate: true });
    await assertScenePixels(scene_, pixels);
    assertEquals(
      scene_.coordinateTransformations,
      scene.coordinateTransformations,
    );
  }
});

Deno.test("toOmeZarrOzx zips a scene with its images", async () => {
  const { pixels, scene } = await tilesScene();
  // The scene's progress counts what the images count on their own.
  let imageWrites = 0;
  for (const multiscales of Object.values(scene.images)) {
    const progress = progressLog();
    await toOmeZarrOzxBrowser(multiscales, { onProgress: progress.onProgress });
    imageWrites += progress.reports.at(-1)![1];
  }

  for (const zipScene of [toOmeZarrOzxBrowser, toOmeZarrOzxData]) {
    const progress = progressLog();
    const zip = await zipScene(scene, { onProgress: progress.onProgress });
    assertEquals(readOzxVersion(zip), "0.6");
    assertEquals(progress.reports.at(-1), [imageWrites, imageWrites]);
    // RFC-9: the zarr.json documents lead, breadth first, the root's first.
    assertEquals(getZipFileList(zip).slice(0, 3), [
      "zarr.json",
      "tile_0/zarr.json",
      "tile_1/zarr.json",
    ]);

    const archive = ZipFileStore.fromBlob(new Blob([zip as BlobPart]));
    const datasetPath = scene.images.tile_0.metadata.datasets[0].path;
    assertEquals(
      consolidatedNodes(await archiveDocument(archive, "/zarr.json"))
        .includes(`tile_1/${datasetPath}`),
      true,
    );
    // Sharded, as an image's .ozx is by default.
    const array = await archiveDocument(
      archive,
      `/tile_0/${datasetPath}/zarr.json`,
    );
    assertEquals(
      (array.codecs as Array<{ name: string }>)[0].name,
      "sharding_indexed",
    );
    for (const read of [fromOmeZarrBrowser, fromOmeZarr]) {
      await assertScenePixels(
        await read(archive, { kind: "scene", validate: true }),
        pixels,
      );
    }
  }
});

Deno.test("toOmeZarr writes a scene to an .ozx path and to a MemoryStore", async () => {
  await withTempDir(async (dir) => {
    const { pixels, scene } = await tilesScene();
    const path = `${dir}/tiles.ozx`;
    await toOmeZarr(path, scene);
    const zip = await Deno.readFile(path);
    assertEquals(readOzxVersion(zip), "0.6");
    await assertScenePixels(
      await fromOmeZarr(ZipFileStore.fromBlob(new Blob([zip as BlobPart])), {
        kind: "scene",
        validate: true,
      }),
      pixels,
    );

    const memory: MemoryStore = new Map();
    await toOmeZarr(memory, scene);
    await assertScenePixels(
      await fromOmeZarr(memory, { kind: "scene", validate: true }),
      pixels,
    );
  });
});

Deno.test("a scene reads from a store object", async () => {
  await withTempDir(async (dir) => {
    const { pixels, scene } = await tilesScene(["tile_0", "sample/tile_1"]);
    const store = `${dir}/tiles.ome.zarr`;
    await toOmeZarr(store, scene);
    await assertScenePixels(
      await fromOmeZarr(new FileSystemStore(store), {
        kind: "scene",
        validate: true,
      }),
      pixels,
    );
  });
});

Deno.test("the matrix arrays of a scene's images are written and consolidated", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    const tile0 = scene.images.tile_0.metadata;
    const toPhysical: Affine = {
      type: "affine",
      name: "to_physical",
      affine: [[1, 0, 0.5], [0, 1, 0.25]],
      input: { name: tile0.coordinateSystems![0].name },
      output: { name: "physical" },
    };
    tile0.coordinateSystems!.push({ name: "physical", axes: tile0.axes });
    tile0.coordinateTransformations = [toPhysical];
    const arrayNodes = [
      "tile_0/coordinateTransformations",
      "tile_0/coordinateTransformations/to_physical",
    ];

    const directory = `${dir}/tiles.ome.zarr`;
    await toOmeZarr(directory, scene);
    const memory: MemoryStore = new Map();
    await toOmeZarrBrowser(memory, scene);
    const roots = [
      await rootDocument(directory),
      documentAt(memory, "/zarr.json"),
    ];
    for (const document of roots) {
      const nodes = consolidatedNodes(document);
      for (const node of arrayNodes) {
        assertEquals(nodes.includes(node), true, node);
      }
    }
    for (const store of [directory, memory]) {
      const read = await fromOmeZarr(store, { kind: "scene", validate: true });
      const [affine] = read.images.tile_0.metadata.coordinateTransformations!;
      assertEquals((affine as Affine).affine, toPhysical.affine);
    }
  });
});

Deno.test("a scene is written where it can be held", async () => {
  await withTempDir(async (dir) => {
    const { scene } = await tilesScene();
    await assertRejects(
      () => toOmeZarrBrowser(`${dir}/tiles.ome.zarr`, scene),
      Error,
      "MemoryStore",
    );
    await assertRejects(
      () => toOmeZarr("https://example.org/tiles.ome.zarr", scene),
      Error,
      "read-only",
    );
    for (const zip of [toOmeZarrOzxBrowser, toOmeZarrOzxData]) {
      await assertRejects(
        () => zip(scene, { version: "0.5" }),
        Error,
        "defined from OME-Zarr 0.6",
      );
    }
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ozx`, scene, { version: "0.4" }),
      Error,
      "RFC-9",
    );

    scene.coordinateTransformations.push({
      type: "displacements",
      path: "coordinateTransformations/dfield",
      interpolation: "linear",
      input: { path: "tile_0", name: "intrinsic" },
      output: { path: "tile_1", name: "intrinsic" },
    });
    for (const zip of [toOmeZarrOzxBrowser, toOmeZarrOzxData]) {
      await assertRejects(() => zip(scene), Error, "written in one piece");
    }
    await assertRejects(
      () => toOmeZarr(`${dir}/tiles.ozx`, scene),
      Error,
      "written in one piece",
    );
    await assertRejects(
      () => Deno.stat(`${dir}/tiles.ozx`),
      Deno.errors.NotFound,
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
