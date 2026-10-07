// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Rotation and affine matrices are written to, and read from, Zarr arrays.
 *
 * Mirrors `py/test/test_matrix_transform_arrays.py`.
 */

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { ZipFileStore } from "@zarrita/storage";
import * as zarr from "zarrita";
import {
  createAffine,
  createAxis,
  createBijection,
  createByDimension,
  createCoordinateSystem,
  createRotation,
  createScale,
  createTransformSequence,
  fromOmeZarr,
  type MemoryStore,
  type NgffMultiscales,
  toOmeZarr,
  toOmeZarrOzxData,
  upgradeOmeZarr,
  type V06Transform,
} from "../src/mod.ts";
import { fromOmeZarr as fromOmeZarrBrowser } from "../src/io/from_ngff_zarr-browser.ts";
import { toOmeZarr as toOmeZarrBrowser } from "../src/io/to_ngff_zarr-browser.ts";
import {
  toMultiscales,
  toNgffImage,
} from "../src/process/to_multiscales-node.ts";
import {
  externalizeMatrixTransforms,
  resolveMatrixTransforms,
  writeMatrixArrays,
} from "../src/utils/matrix_transform_arrays.ts";

/**
 * Entries a JSON round trip through a lossy tool would round: no short decimal
 * spells them, and -0 compares equal to 0 unless the bits are checked.
 */
const AFFINE = [
  [0.1, 1 / 3, 2 / 3, Math.PI],
  [1e-17, 1.0, 0.30000000000000004, Math.E],
  [-0, 0, 1.0, -7.5],
];

const ANGLE = 0.1;
const ROTATION = [
  [Math.cos(ANGLE), -Math.sin(ANGLE), 0.0],
  [Math.sin(ANGLE), Math.cos(ANGLE), 0.0],
  [0.0, 0.0, 1.0],
];

const TRANSPOSED_ROTATION = ROTATION.map((_, i) =>
  ROTATION.map((row) => row[i])
);

function bits(matrix: number[][]): number[] {
  return [...new Uint8Array(Float64Array.from(matrix.flat()).buffer)];
}

async function buildMultiscales(
  transforms: V06Transform[],
): Promise<NgffMultiscales> {
  const shape = [4, 4, 4];
  const image = await toNgffImage(new Float32Array(64), {
    dims: ["z", "y", "x"],
    shape,
  });
  const multiscales = await toMultiscales(image, { scaleFactors: [] });
  const intrinsic = multiscales.metadata.coordinateSystems![0].name;
  multiscales.metadata.coordinateSystems!.push(
    createCoordinateSystem(
      "output",
      (["z", "y", "x"] as const).map((name) => createAxis(name, "space")),
    ),
  );
  for (const transform of transforms) {
    transform.input = { name: intrinsic };
    transform.output = { name: "output" };
  }
  multiscales.metadata.coordinateTransformations = transforms;
  return multiscales;
}

function readJson(store: MemoryStore, key: string): Record<string, unknown> {
  const bytes = store.get(key);
  assert(bytes !== undefined, `no document at ${key}`);
  return JSON.parse(new TextDecoder().decode(bytes));
}

function writeJson(
  store: MemoryStore,
  key: string,
  document: Record<string, unknown>,
): void {
  store.set(key, new TextEncoder().encode(JSON.stringify(document)));
}

function rootEntry(root: Record<string, unknown>): Record<string, unknown> {
  const attributes = root.attributes as {
    ome: { multiscales: Array<Record<string, unknown>> };
  };
  return attributes.ome.multiscales[0];
}

function writtenTransforms(
  store: MemoryStore,
): Array<Record<string, unknown>> {
  return rootEntry(readJson(store, "/zarr.json"))
    .coordinateTransformations as Array<Record<string, unknown>>;
}

function deleteNode(store: MemoryStore, path: string): void {
  for (const key of [...store.keys()]) {
    if (key.startsWith(`/${path}/`)) {
      store.delete(key);
    }
  }
}

async function storedValues(
  store: MemoryStore,
  path: string,
): Promise<number[][]> {
  const array = await zarr.open(zarr.root(store).resolve(path), {
    kind: "array",
  });
  const { data, shape } = await zarr.get(array);
  const values = data as Float64Array;
  return Array.from(
    { length: shape[0] },
    (_, i) => [...values.subarray(i * shape[1], (i + 1) * shape[1])],
  );
}

for (const version of ["0.6", "0.9.dev1"] as const) {
  Deno.test(`matrices are written as float64 arrays at ${version}`, async () => {
    const affine = createAffine(AFFINE);
    affine.name = "to_output";
    const sequence = createTransformSequence([
      createScale([1.0, 2.0, 3.0]),
      createRotation(ROTATION),
    ]);
    sequence.name = "rotate";
    const multiscales = await buildMultiscales([affine, sequence]);
    const store: MemoryStore = new Map();

    await toOmeZarr(store, multiscales, { version });

    const [writtenAffine, writtenSequence] = writtenTransforms(store);
    assertEquals(writtenAffine.path, "coordinateTransformations/to_output");
    assertEquals("affine" in writtenAffine, false);
    const rotation = (writtenSequence.transformations as Array<
      Record<string, unknown>
    >)[1];
    assertEquals(rotation, {
      type: "rotation",
      path: "coordinateTransformations/rotation",
    });

    const document = readJson(
      store,
      "/coordinateTransformations/to_output/zarr.json",
    );
    assertEquals(document.data_type, "float64");
    assertEquals(document.shape, [3, 4]);
    assertEquals(
      (document.chunk_grid as { configuration: { chunk_shape: number[] } })
        .configuration.chunk_shape,
      [3, 4],
    );
    assertEquals(document.codecs, [
      { name: "bytes", configuration: { endian: "little" } },
    ]);
    assertEquals(
      readJson(store, "/coordinateTransformations/rotation/zarr.json").shape,
      [3, 3],
    );
    assertEquals(
      readJson(store, "/coordinateTransformations/zarr.json").node_type,
      "group",
    );
    assertEquals(
      bits(await storedValues(store, "coordinateTransformations/to_output")),
      bits(AFFINE),
    );

    // The consolidated block lists the matrix arrays and their group.
    const consolidated = (readJson(store, "/zarr.json")
      .consolidated_metadata as { metadata: Record<string, unknown> })
      .metadata;
    for (
      const key of [
        "coordinateTransformations",
        "coordinateTransformations/to_output",
        "coordinateTransformations/rotation",
      ]
    ) {
      assert(key in consolidated, `${key} is not consolidated`);
    }

    // The caller's transforms are serialized, not edited.
    assertEquals(affine.path, undefined);
    assertEquals(affine.affine, AFFINE);
  });
}

Deno.test("matrices round-trip bit for bit", async () => {
  const affine = createAffine(AFFINE);
  affine.name = "to_output";
  const bijection = createBijection(
    createRotation(ROTATION),
    createRotation(TRANSPOSED_ROTATION),
  );
  bijection.name = "rotate";
  const store: MemoryStore = new Map();
  await toOmeZarr(store, await buildMultiscales([affine, bijection]), {
    version: "0.6",
  });

  const imported = await fromOmeZarr(store, { validate: true });

  const [readAffine, readBijection] = imported.metadata
    .coordinateTransformations!;
  if (readAffine.type !== "affine" || readBijection.type !== "bijection") {
    throw new Error("expected an affine then a bijection");
  }
  assertEquals(readAffine.path, "coordinateTransformations/to_output");
  assertEquals(bits(readAffine.affine), bits(AFFINE));
  const { forward, inverse } = readBijection;
  if (forward.type !== "rotation" || inverse.type !== "rotation") {
    throw new Error("expected a bijection of rotations");
  }
  assertEquals(forward.path, "coordinateTransformations/rotation");
  assertEquals(bits(forward.rotation), bits(ROTATION));
  assertEquals(inverse.path, "coordinateTransformations/rotation_1");
  assertEquals(bits(inverse.rotation), bits(TRANSPOSED_ROTATION));

  // A rewrite of what was read keeps each matrix at the path it was read from.
  const copy: MemoryStore = new Map();
  await toOmeZarr(copy, imported, { version: "0.6" });
  assertEquals(writtenTransforms(copy), writtenTransforms(store));
  const reread = await fromOmeZarr(copy, { validate: true });
  const rereadAffine = reread.metadata.coordinateTransformations![0];
  if (rereadAffine.type !== "affine") {
    throw new Error("expected an affine");
  }
  assertEquals(bits(rereadAffine.affine), bits(AFFINE));
});

Deno.test("matrices round-trip through a directory store", async () => {
  const directory = await Deno.makeTempDir();
  try {
    const store = `${directory}/image.ome.zarr`;
    const affine = createAffine(AFFINE);
    affine.name = "to_output";
    await toOmeZarr(store, await buildMultiscales([affine]), {
      version: "0.6",
    });

    const document = JSON.parse(
      await Deno.readTextFile(
        `${store}/coordinateTransformations/to_output/zarr.json`,
      ),
    );
    assertEquals(document.data_type, "float64");
    const [readAffine] = (await fromOmeZarr(store, { validate: true }))
      .metadata.coordinateTransformations!;
    if (readAffine.type !== "affine") {
      throw new Error("expected an affine");
    }
    assertEquals(bits(readAffine.affine), bits(AFFINE));
  } finally {
    await Deno.remove(directory, { recursive: true });
  }
});

Deno.test("an .ozx archive holds the matrix arrays", async () => {
  const affine = createAffine(AFFINE);
  affine.name = "to_output";
  const zipData = await toOmeZarrOzxData(await buildMultiscales([affine]), {
    version: "0.6",
  });

  const imported = await fromOmeZarr(
    ZipFileStore.fromBlob(new Blob([zipData as BlobPart])),
  );

  const [readAffine] = imported.metadata.coordinateTransformations!;
  if (readAffine.type !== "affine") {
    throw new Error("expected an affine");
  }
  assertEquals(readAffine.path, "coordinateTransformations/to_output");
  assertEquals(bits(readAffine.affine), bits(AFFINE));
});

Deno.test("the browser writer and reader carry the matrix arrays", async () => {
  const rotation = createRotation(ROTATION);
  rotation.name = "to_output";
  const store: MemoryStore = new Map();

  await toOmeZarrBrowser(store, await buildMultiscales([rotation]), {
    version: "0.6",
  });

  assertEquals(
    writtenTransforms(store)[0].path,
    "coordinateTransformations/to_output",
  );
  assertEquals("rotation" in writtenTransforms(store)[0], false);
  const [readRotation] = (await fromOmeZarrBrowser(store))
    .metadata.coordinateTransformations!;
  if (readRotation.type !== "rotation") {
    throw new Error("expected a rotation");
  }
  assertEquals(bits(readRotation.rotation), bits(ROTATION));
});

Deno.test("generated paths", () => {
  const transforms: Array<Record<string, unknown>> = [
    { type: "affine", affine: AFFINE, name: "to_output" },
    {
      type: "affine",
      affine: AFFINE,
      path: "coordinateTransformations/affine",
    },
    { type: "affine", affine: AFFINE },
    { type: "affine", affine: AFFINE, name: "not a node name" },
    { type: "affine", affine: AFFINE, name: "__reserved" },
    {
      type: "byDimension",
      transformations: [{
        transformation: { type: "rotation", rotation: [[1.0]] },
        inputAxes: [0],
        outputAxes: [0],
      }],
    },
    { type: "affine", affine: AFFINE, name: "to_output" },
  ];

  const arrays = externalizeMatrixTransforms(transforms);

  const byDimension = transforms[5].transformations as Array<
    { transformation: Record<string, unknown> }
  >;
  const paths = [
    transforms[0].path,
    transforms[1].path,
    transforms[2].path,
    transforms[3].path,
    transforms[4].path,
    byDimension[0].transformation.path,
    transforms[6].path,
  ];
  assertEquals(paths, [
    "coordinateTransformations/to_output",
    // A path the transform names is kept, and no generated path takes it.
    "coordinateTransformations/affine",
    "coordinateTransformations/affine_1",
    "coordinateTransformations/affine_2",
    "coordinateTransformations/affine_3",
    "coordinateTransformations/rotation",
    "coordinateTransformations/to_output_1",
  ]);
  assertEquals([...arrays.keys()].sort(), [...paths].sort());
  for (const transform of [...transforms, byDimension[0].transformation]) {
    assertEquals("affine" in transform, false);
    assertEquals("rotation" in transform, false);
  }
});

Deno.test("a path without a matrix is left to the store", () => {
  const transforms = [{ type: "rotation", path: "elsewhere/rotation" }];

  assertEquals(externalizeMatrixTransforms(transforms).size, 0);
  assertEquals(transforms, [{ type: "rotation", path: "elsewhere/rotation" }]);
});

for (
  const [transform, message] of [
    [{ type: "affine" }, "holds no matrix and names no path"],
    [{ type: "affine", affine: [[1.0, 0.0], [1.0]] }, "rectangular matrix"],
    [{ type: "rotation", rotation: [1.0, 0.0] }, "matrix of numbers"],
    [{ type: "affine", affine: [[]] }, "non-empty 2D matrix"],
    [{ type: "affine", affine: AFFINE, path: "../outside" }, "'..'"],
    [{ type: "affine", affine: AFFINE, path: "/absolute" }, "relative path"],
  ] as const
) {
  Deno.test(`an unwritable matrix is refused: ${message}`, () => {
    assertThrows(
      () => externalizeMatrixTransforms([structuredClone(transform)]),
      Error,
      message,
    );
  });
}

Deno.test("one path cannot hold two matrices", () => {
  assertThrows(
    () =>
      externalizeMatrixTransforms([
        { type: "affine", affine: AFFINE, path: "shared" },
        {
          type: "affine",
          affine: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
          path: "shared",
        },
      ]),
    Error,
    "different matrices",
  );
});

Deno.test("a refused matrix leaves the store untouched", async () => {
  const affine = createAffine(AFFINE);
  affine.path = "../outside";
  const multiscales = await buildMultiscales([affine]);
  const store: MemoryStore = new Map();

  await assertRejects(
    () => toOmeZarr(store, multiscales, { version: "0.6" }),
    Error,
    "'..'",
  );
  assertEquals(store.size, 0);
});

Deno.test("a path-only matrix written elsewhere is read", async () => {
  const affine = createAffine(AFFINE);
  affine.name = "t";
  const store: MemoryStore = new Map();
  await toOmeZarr(store, await buildMultiscales([affine]), { version: "0.6" });
  deleteNode(store, "coordinateTransformations");
  await writeMatrixArrays(
    zarr.root(store),
    new Map([["params/t", {
      shape: [3, 4] as [number, number],
      data: Float64Array.from(AFFINE.flat()),
    }]]),
  );
  const root = readJson(store, "/zarr.json");
  delete root.consolidated_metadata;
  const entry = rootEntry(root);
  (entry.coordinateTransformations as Array<Record<string, unknown>>)[0].path =
    "params/t";
  writeJson(store, "/zarr.json", root);

  const [readAffine] = (await fromOmeZarr(store, { validate: true }))
    .metadata.coordinateTransformations!;

  if (readAffine.type !== "affine") {
    throw new Error("expected an affine");
  }
  assertEquals(readAffine.path, "params/t");
  assertEquals(bits(readAffine.affine), bits(AFFINE));
});

Deno.test("a big-endian integer matrix array is read", async () => {
  const store: MemoryStore = new Map();
  const location = zarr.root(store);
  await zarr.create(location, { attributes: {} });
  const array = await zarr.create(location.resolve("params/r"), {
    shape: [2, 2],
    dtype: "int32",
    chunkShape: [2, 2],
    codecs: [{ name: "bytes", configuration: { endian: "big" } }],
  });
  await zarr.set(array, null, {
    data: Int32Array.from([0, -1, 1, 0]),
    shape: [2, 2],
    stride: [2, 1],
  });
  const raw = [{ type: "rotation", path: "params/r" }];

  const resolved = await resolveMatrixTransforms(raw, location);

  assertEquals(resolved[0].rotation, [[0, -1], [1, 0]]);
  // Twice, as a read that swapped the stored bytes in place would differ.
  assertEquals(
    (await resolveMatrixTransforms(raw, location))[0].rotation,
    [[0, -1], [1, 0]],
  );
  // The raw attributes are not edited.
  assertEquals(raw, [{ type: "rotation", path: "params/r" }]);
});

Deno.test("a matrix array that is not 2D is refused on read", async () => {
  const store: MemoryStore = new Map();
  const location = zarr.root(store);
  await zarr.create(location, { attributes: {} });
  const array = await zarr.create(location.resolve("params/v"), {
    shape: [3],
    dtype: "float64",
    chunkShape: [3],
  });
  await zarr.set(array, null, {
    data: Float64Array.from([1, 0, 0]),
    shape: [3],
    stride: [1],
  });
  const malformed = { type: "bijection", forward: null, inverse: "x" };

  // Malformed entries are left for the transform parser to report.
  assertEquals(
    await resolveMatrixTransforms([malformed], location),
    [malformed],
  );
  await assertRejects(
    () =>
      resolveMatrixTransforms(
        [{ type: "rotation", path: "params/v" }],
        location,
      ),
    Error,
    "non-empty 2D matrix",
  );
});

for (const path of ["../outside/r", "/outside/r", "params/./r"]) {
  Deno.test(`a matrix path outside the group is refused on read: ${path}`, async () => {
    // A crafted path cannot read an array beside the multiscales group.
    const store: MemoryStore = new Map();
    await writeMatrixArrays(
      zarr.root(store),
      new Map([["outside/r", {
        shape: [3, 3] as [number, number],
        data: Float64Array.from(ROTATION.flat()),
      }]]),
    );

    await assertRejects(
      () =>
        resolveMatrixTransforms(
          [{ type: "rotation", path }],
          zarr.root(store).resolve("image"),
        ),
      Error,
      "relative path inside the multiscales",
    );
  });
}

Deno.test("a missing matrix array is reported", async () => {
  const affine = createAffine(AFFINE);
  affine.name = "t";
  const store: MemoryStore = new Map();
  await toOmeZarr(store, await buildMultiscales([affine]), {
    version: "0.6",
    consolidateMetadata: false,
  });
  deleteNode(store, "coordinateTransformations");

  await assertRejects(
    () => fromOmeZarr(store),
    Error,
    "coordinateTransformations/t",
  );
});

Deno.test("an in-place upgrade moves inline matrices into arrays", async () => {
  const inline = createAffine(AFFINE);
  inline.name = "inline";
  const stored = createRotation(ROTATION);
  stored.name = "stored";
  const store: MemoryStore = new Map();
  await toOmeZarr(store, await buildMultiscales([inline, stored]), {
    version: "0.6",
  });
  // Rewrite the affine the way earlier releases did: inline, with no array.
  deleteNode(store, "coordinateTransformations/inline");
  const root = readJson(store, "/zarr.json");
  const entry = rootEntry(root);
  const written = entry.coordinateTransformations as Array<
    Record<string, unknown>
  >;
  delete written[0].path;
  written[0].affine = AFFINE;
  writeJson(store, "/zarr.json", root);
  const storedChunk = store.get("/coordinateTransformations/stored/c/0/0");
  assert(storedChunk !== undefined);

  await upgradeOmeZarr(store, { version: "0.9.dev1" });

  const [upgradedInline, upgradedStored] = writtenTransforms(store);
  assertEquals(upgradedInline.path, "coordinateTransformations/inline");
  assertEquals("affine" in upgradedInline, false);
  assertEquals(upgradedStored.path, "coordinateTransformations/stored");
  // The array the source already held is left as it was.
  assert(store.get("/coordinateTransformations/stored/c/0/0") === storedChunk);
  const consolidated = (readJson(store, "/zarr.json")
    .consolidated_metadata as { metadata: Record<string, unknown> }).metadata;
  assert("coordinateTransformations/inline" in consolidated);
  const upgraded = await fromOmeZarr(store, { validate: true });
  const [readAffine, readRotation] = upgraded.metadata
    .coordinateTransformations!;
  if (readAffine.type !== "affine" || readRotation.type !== "rotation") {
    throw new Error("expected an affine then a rotation");
  }
  // The inline source went through JSON, which spells -0 as 0; the array holds
  // what the source held.
  assertEquals(
    bits(readAffine.affine),
    bits(JSON.parse(JSON.stringify(AFFINE))),
  );
  assertEquals(bits(readRotation.rotation), bits(ROTATION));
});

Deno.test("an in-place upgrade refuses an occupied matrix path", async () => {
  // An unrelated array at the generated path is neither replaced nor lost.
  const inline = createAffine(AFFINE);
  inline.name = "inline";
  const store: MemoryStore = new Map();
  await toOmeZarr(store, await buildMultiscales([inline]), { version: "0.6" });
  deleteNode(store, "coordinateTransformations/inline");
  const root = readJson(store, "/zarr.json");
  const written = rootEntry(root).coordinateTransformations as Array<
    Record<string, unknown>
  >;
  delete written[0].path;
  written[0].affine = AFFINE;
  writeJson(store, "/zarr.json", root);
  await writeMatrixArrays(
    zarr.root(store),
    new Map([["coordinateTransformations/inline", {
      shape: [2, 2] as [number, number],
      data: Float64Array.from([1, 0, 0, 1]),
    }]]),
  );
  const rootBefore = store.get("/zarr.json");

  await assertRejects(
    () => upgradeOmeZarr(store, { version: "0.9.dev1" }),
    Error,
    "node at 'coordinateTransformations/inline'",
  );

  assert(store.get("/zarr.json") === rootBefore);
  assertEquals(
    await storedValues(store, "coordinateTransformations/inline"),
    [[1, 0], [0, 1]],
  );
});

Deno.test("a matrix path overlapping a dataset is refused", async () => {
  const dataset = (await buildMultiscales([])).metadata.datasets[0].path;
  const paths = [dataset, `${dataset}/m`];
  if (dataset.includes("/")) {
    paths.push(dataset.split("/")[0]);
  }
  for (const path of paths) {
    const affine = createAffine(AFFINE);
    affine.path = path;
    const multiscales = await buildMultiscales([affine]);
    const store: MemoryStore = new Map();

    await assertRejects(
      () => toOmeZarr(store, multiscales, { version: "0.6" }),
      Error,
      `overlaps the dataset array '${dataset}'`,
    );
    assertEquals(store.size, 0);
  }
});

Deno.test("byDimension items are externalized", async () => {
  const byDimension = createByDimension([
    {
      transformation: createRotation([[0.0, -1.0], [1.0, 0.0]]),
      inputAxes: [1, 2],
      outputAxes: [1, 2],
    },
    {
      transformation: createScale([2.0]),
      inputAxes: [0],
      outputAxes: [0],
    },
  ]);
  const store: MemoryStore = new Map();

  await toOmeZarr(store, await buildMultiscales([byDimension]), {
    version: "0.6",
  });

  const [written] = writtenTransforms(store);
  const item = (written.transformations as Array<
    { transformation: Record<string, unknown> }
  >)[0].transformation;
  assertEquals(item, {
    type: "rotation",
    path: "coordinateTransformations/rotation",
  });
  const [imported] = (await fromOmeZarr(store, { validate: true })).metadata
    .coordinateTransformations!;
  if (imported.type !== "byDimension") {
    throw new Error("expected a byDimension");
  }
  const rotation = imported.transformations[0].transformation;
  if (rotation.type !== "rotation") {
    throw new Error("expected a rotation item");
  }
  assertEquals(rotation.rotation, [[0.0, -1.0], [1.0, 0.0]]);
});
