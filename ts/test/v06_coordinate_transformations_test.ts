// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * OME-Zarr v0.6 (RFC 5) coordinate-systems and coordinate-transformations tests.
 *
 * Mirrors the Python `py/test/test_coordinate_transformations.py` and the v0.6
 * cases added to `test_convert_ome_zarr_version.py`, adapted to the TypeScript
 * API. Self-contained: builds a small multiscale image in memory rather than
 * relying on downloaded fixtures.
 */

import {
  assertEquals,
  assertExists,
  assertRejects,
  assertThrows,
} from "@std/assert";
import {
  AnatomicalOrientationValues,
  createAffine,
  createAxis,
  createBijection,
  createByDimension,
  createCoordinateSystem,
  createIdentity,
  createMapAxis,
  createRotation,
  createScale,
  createTransformSequence,
  createTranslation,
  fromOmeZarr,
  isSupportedVersion,
  isV06Version,
  type MemoryStore,
  NgffImage,
  type NgffMultiscales,
  toOmeZarr,
  type V06Transform,
} from "../src/mod.ts";
import {
  toMultiscales,
  toNgffImage,
} from "../src/process/to_multiscales-node.ts";
import { fromOmeZarr as fromOmeZarrBrowser } from "../src/io/from_ngff_zarr-browser.ts";
import { toOmeZarr as toOmeZarrBrowser } from "../src/io/to_ngff_zarr-browser.ts";
import { prepareRfc9Metadata } from "../src/io/to_ngff_zarr_ozx_common.ts";
import { CoordinateTransformationSchema } from "../src/schemas/coordinate_systems.ts";
import {
  parseV06Transforms,
  validateV06Transform,
} from "../src/utils/v06_metadata.ts";
import { fromFileUrl } from "@std/path";
import type { CoordinateSystem } from "../src/types/zarr_metadata.ts";

async function buildMultiscales(): Promise<NgffMultiscales> {
  const shape = [32, 32, 32];
  const data = new Uint8Array(shape[0] * shape[1] * shape[2]);
  for (let i = 0; i < data.length; i++) {
    data[i] = i % 256;
  }
  const image = await toNgffImage(data, { dims: ["z", "y", "x"], shape });
  return await toMultiscales(image, { scaleFactors: [2], chunks: 16 });
}

Deno.test("toMultiscales exposes the intrinsic coordinate system", async () => {
  const multiscales = await buildMultiscales();
  assertExists(multiscales.metadata.coordinateSystems);
  const names = multiscales.metadata.coordinateSystems!.map((cs) => cs.name);
  assertEquals(names.includes("intrinsic"), true);
});

// Locate the root group's zarr.json in a MemoryStore and return its parsed
// `ome` attributes, independent of how zarrita keys the root node.
function readOmeAttributes(store: MemoryStore): Record<string, unknown> {
  for (const [key, bytes] of store.entries()) {
    if (!key.endsWith("zarr.json")) continue;
    const node = JSON.parse(new TextDecoder().decode(bytes));
    if (
      node.node_type === "group" && node.attributes &&
      typeof node.attributes === "object" && "ome" in node.attributes
    ) {
      return node.attributes.ome as Record<string, unknown>;
    }
  }
  throw new Error("No root group with an 'ome' attribute found in the store");
}

// Rewrite the root group's `ome.multiscales[0]` entry in place, so reader
// behavior can be exercised against on-disk shapes the writer never emits
// (bare per-dataset transforms, unknown transform types, corrupt metadata).
// Only the root metadata is changed; the previously written arrays remain.
function mutateRootEntry(
  store: MemoryStore,
  mutate: (entry: Record<string, unknown>) => void,
): void {
  for (const [key, bytes] of store.entries()) {
    if (!key.endsWith("zarr.json")) continue;
    const node = JSON.parse(new TextDecoder().decode(bytes));
    if (
      node.node_type === "group" && node.attributes &&
      typeof node.attributes === "object" && "ome" in node.attributes
    ) {
      const ome = node.attributes.ome as {
        multiscales: Array<Record<string, unknown>>;
      };
      mutate(ome.multiscales[0]);
      store.set(key, new TextEncoder().encode(JSON.stringify(node)));
      return;
    }
  }
  throw new Error("No root group with an 'ome' attribute found in the store");
}

// Overwrite the root group's `ome.version` string in place, so the reader can
// be exercised against an on-disk version tag independent of what the writer
// currently emits (e.g. a plain `0.6` store, or a future draft tag).
function setOmeVersion(store: MemoryStore, version: string): void {
  for (const [key, bytes] of store.entries()) {
    if (!key.endsWith("zarr.json")) continue;
    const node = JSON.parse(new TextDecoder().decode(bytes));
    if (
      node.node_type === "group" && node.attributes &&
      typeof node.attributes === "object" && "ome" in node.attributes
    ) {
      (node.attributes.ome as { version: string }).version = version;
      store.set(key, new TextEncoder().encode(JSON.stringify(node)));
      return;
    }
  }
  throw new Error("No root group with an 'ome' attribute found in the store");
}

Deno.test("v0.6 write produces coordinate systems and sequence transforms", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });

  const ome = readOmeAttributes(store) as {
    version: string;
    multiscales: Array<Record<string, unknown>>;
  };
  // The on-disk tag is the pre-release the bundled schemas carry, not the
  // requested `"0.6"`.
  assertEquals(ome.version, "0.6rc0");

  const entry = ome.multiscales[0];
  // v0.6 carries axes inside coordinate systems, not at the entry level.
  assertEquals("axes" in entry, false);

  const coordinateSystems = entry.coordinateSystems as Array<{
    name: string;
    axes: Array<{ name: string }>;
  }>;
  assertEquals(coordinateSystems[0].name, "intrinsic");
  assertEquals(coordinateSystems[0].axes.map((a) => a.name), ["z", "y", "x"]);

  const datasets = entry.datasets as Array<{
    coordinateTransformations: Array<Record<string, unknown>>;
  }>;
  const sequence = datasets[0].coordinateTransformations[0] as {
    type: string;
    name: string;
    input: { path: string };
    output: { name: string };
    transformations: Array<{ type: string }>;
  };
  assertEquals(sequence.type, "sequence");
  assertEquals(sequence.name, "scale0_to_intrinsic");
  assertEquals(sequence.input.path, "scale0");
  assertEquals(sequence.output.name, "intrinsic");
  assertEquals(sequence.transformations[0].type, "scale");
  assertEquals(sequence.transformations[1].type, "translation");
});

Deno.test("v0.6 round-trip preserves multiscale structure", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });

  const imported = await fromOmeZarr(store, { validate: true, version: "0.6" });
  assertEquals(imported.metadata.version, "0.6");
  assertEquals(imported.images.length, multiscales.images.length);
  assertEquals(imported.images[0].dims, ["z", "y", "x"]);
  assertEquals(imported.images[0].scale, multiscales.images[0].scale);
  assertEquals(imported.images[1].scale, multiscales.images[1].scale);
  assertEquals(
    imported.images[0].translation,
    multiscales.images[0].translation,
  );

  assertExists(imported.metadata.coordinateSystems);
  assertEquals(imported.metadata.coordinateSystems![0].name, "intrinsic");
});

// Mirrors test_coordinate_transformations.py::test_transform_serialization:
// every transform type survives a v0.6 write/read round-trip, including its
// input/output coordinate-system references.
function transformsToRoundTrip(): V06Transform[] {
  return [
    createIdentity(),
    createScale([2.0, 2.0, 2.0]),
    createTranslation([10.0, 20.0, 30.0]),
    createRotation([
      [0.0, -1.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
    ]),
    createAffine([
      [1.0, 0.0, 0.0, 5.0],
      [0.0, 1.0, 0.0, 10.0],
      [0.0, 0.0, 1.0, 15.0],
    ]),
    createTransformSequence([
      createScale([2.0, 2.0, 2.0]),
      createTranslation([10.0, 20.0, 30.0]),
    ]),
    createMapAxis([2, 0, 1]),
    createByDimension([
      {
        transformation: createScale([2.0, 3.0]),
        inputAxes: [0, 1],
        outputAxes: [0, 1],
      },
      {
        transformation: createTranslation([5.0]),
        inputAxes: [2],
        outputAxes: [2],
      },
    ]),
    createBijection(
      { type: "displacements", path: "forward_field" },
      { type: "displacements", path: "inverse_field" },
    ),
  ];
}

for (const transform of transformsToRoundTrip()) {
  Deno.test(
    `v0.6 round-trips a top-level ${transform.type} transform`,
    async () => {
      const multiscales = await buildMultiscales();

      const outputCs = createCoordinateSystem("output", [
        createAxis("z", "space"),
        createAxis("y", "space"),
        createAxis("x", "space"),
      ]);
      const intrinsic = multiscales.metadata.coordinateSystems![0];
      transform.input = { name: intrinsic.name };
      transform.output = { name: outputCs.name };
      transform.name = `additional_${transform.type}`;

      multiscales.metadata.coordinateSystems!.push(outputCs);
      multiscales.metadata.coordinateTransformations = [transform];

      const store: MemoryStore = new Map();
      await toOmeZarr(store, multiscales, { version: "0.6" });

      const imported = await fromOmeZarr(store);
      const importedTransforms = imported.metadata.coordinateTransformations;
      assertExists(importedTransforms);
      assertEquals(importedTransforms!.length, 1);
      assertEquals(imported.metadata.coordinateSystems!.length, 2);

      const roundTripped = importedTransforms![0];
      assertEquals(roundTripped.type, transform.type);
      assertEquals(roundTripped.input, { name: intrinsic.name });
      assertEquals(roundTripped.output, { name: outputCs.name });
    },
  );
}

Deno.test("rotation and affine payloads survive the round-trip", async () => {
  const multiscales = await buildMultiscales();
  const rotation = createRotation([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
  ]);
  const affine = createAffine([
    [1.0, 0.0, 0.0, 5.0],
    [0.0, 1.0, 0.0, 10.0],
    [0.0, 0.0, 1.0, 15.0],
  ]);
  const intrinsic = multiscales.metadata.coordinateSystems![0];
  for (const t of [rotation, affine]) {
    t.input = { name: intrinsic.name };
    t.output = { name: intrinsic.name };
  }
  multiscales.metadata.coordinateTransformations = [rotation, affine];

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  const imported = await fromOmeZarr(store);

  const transforms = imported.metadata.coordinateTransformations!;
  assertEquals(transforms.length, 2);
  const importedRotation = transforms[0];
  const importedAffine = transforms[1];
  if (importedRotation.type !== "rotation") {
    throw new Error("expected a rotation transform");
  }
  if (importedAffine.type !== "affine") {
    throw new Error("expected an affine transform");
  }
  assertEquals(importedRotation.rotation, rotation.rotation);
  assertEquals(importedAffine.affine, affine.affine);
});

// Mirrors test_convert_ome_zarr_version.py::test_conversion, adapted to a
// self-contained in-memory multiscale image.
Deno.test("convert across v0.4, v0.5, and v0.6", async () => {
  const multiscales = await buildMultiscales();
  for (const version of ["0.4", "0.5", "0.6"] as const) {
    const store: MemoryStore = new Map();
    await toOmeZarr(store, multiscales, { version });
    const back = await fromOmeZarr(store, { validate: true, version });
    assertEquals(back.metadata.version, version);
    assertEquals(back.images.length, multiscales.images.length);
    assertEquals(back.images[0].dims, ["z", "y", "x"]);
  }
});

Deno.test("convert v0.6 -> v0.5 -> v0.4 chain preserves scale", async () => {
  const multiscales = await buildMultiscales();

  const s6: MemoryStore = new Map();
  await toOmeZarr(s6, multiscales, { version: "0.6" });
  const m6 = await fromOmeZarr(s6, { version: "0.6" });

  const s5: MemoryStore = new Map();
  await toOmeZarr(s5, m6, { version: "0.5" });
  const m5 = await fromOmeZarr(s5, { version: "0.5" });
  assertEquals(m5.metadata.version, "0.5");

  const s4: MemoryStore = new Map();
  await toOmeZarr(s4, m5, { version: "0.4" });
  const m4 = await fromOmeZarr(s4, { version: "0.4" });
  assertEquals(m4.metadata.version, "0.4");

  assertEquals(m4.images.length, multiscales.images.length);
  assertEquals(m4.images[0].scale, multiscales.images[0].scale);
  assertEquals(m4.images[1].scale, multiscales.images[1].scale);
});

Deno.test("v0.6 round-trip via browser modules", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarrBrowser(store, multiscales, { version: "0.6" });

  const imported = await fromOmeZarrBrowser(store, { version: "0.6" });
  assertEquals(imported.metadata.version, "0.6");
  assertEquals(imported.images.length, multiscales.images.length);
  assertEquals(imported.images[0].dims, ["z", "y", "x"]);
  assertEquals(imported.images[0].scale, multiscales.images[0].scale);
  assertExists(imported.metadata.coordinateSystems);
  assertEquals(imported.metadata.coordinateSystems![0].name, "intrinsic");
});

// Downgrading v0.6 -> v0.5 keeps the simple scale/translation subset of
// top-level transforms but drops richer ones (rotation/affine/sequence), which
// v0.4/v0.5 cannot represent. Exercises legacyTopLevelTransforms.
Deno.test("downgrade keeps a top-level scale but drops a rotation", async () => {
  const intrinsicName = "intrinsic";

  const withScale = await buildMultiscales();
  withScale.metadata.coordinateTransformations = [{
    type: "scale",
    scale: [2.0, 2.0, 2.0],
    input: { name: intrinsicName },
    output: { name: intrinsicName },
  }];
  const scaleStore: MemoryStore = new Map();
  await toOmeZarr(scaleStore, withScale, { version: "0.5" });
  const scaleBack = await fromOmeZarr(scaleStore, { version: "0.5" });
  const keptTransforms = scaleBack.metadata.coordinateTransformations!;
  assertEquals(keptTransforms.length, 1);
  assertEquals(keptTransforms[0].type, "scale");

  const withRotation = await buildMultiscales();
  withRotation.metadata.coordinateTransformations = [createRotation([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
  ])];
  withRotation.metadata.coordinateTransformations[0].input = {
    name: intrinsicName,
  };
  withRotation.metadata.coordinateTransformations[0].output = {
    name: intrinsicName,
  };
  const rotStore: MemoryStore = new Map();
  await toOmeZarr(rotStore, withRotation, { version: "0.5" });
  // The rotation is dropped, so the v0.5 entry carries no top-level transforms.
  const ome = readOmeAttributes(rotStore) as {
    multiscales: Array<Record<string, unknown>>;
  };
  assertEquals("coordinateTransformations" in ome.multiscales[0], false);
  const rotBack = await fromOmeZarr(rotStore, { version: "0.5" });
  assertEquals(rotBack.metadata.coordinateTransformations, undefined);
});

// Writing v0.6 from metadata that lacks coordinateSystems (e.g. built via
// createMetadata) synthesizes the implicit "intrinsic" system from the axes.
Deno.test("v0.6 write synthesizes intrinsic system when absent", async () => {
  const multiscales = await buildMultiscales();
  delete (multiscales.metadata as { coordinateSystems?: unknown })
    .coordinateSystems;

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });

  const imported = await fromOmeZarr(store, { version: "0.6" });
  assertExists(imported.metadata.coordinateSystems);
  assertEquals(imported.metadata.coordinateSystems![0].name, "intrinsic");
  assertEquals(
    imported.metadata.coordinateSystems![0].axes.map((a) => a.name),
    ["z", "y", "x"],
  );
});

// rotation/affine may carry an external `path` to binary parameters; it must
// survive the v0.6 round-trip.
Deno.test("v0.6 preserves the path field on rotation and affine", async () => {
  const multiscales = await buildMultiscales();
  const intrinsicName = multiscales.metadata.coordinateSystems![0].name;
  const rotation: V06Transform = {
    type: "rotation",
    rotation: [[0.0, -1.0], [1.0, 0.0]],
    path: "transforms/rotation",
    input: { name: intrinsicName },
    output: { name: intrinsicName },
  };
  const affine: V06Transform = {
    type: "affine",
    affine: [[1.0, 0.0], [0.0, 1.0]],
    path: "transforms/affine",
    input: { name: intrinsicName },
    output: { name: intrinsicName },
  };
  multiscales.metadata.coordinateTransformations = [rotation, affine];

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  const imported = await fromOmeZarr(store);

  const transforms = imported.metadata.coordinateTransformations!;
  if (transforms[0].type !== "rotation" || transforms[1].type !== "affine") {
    throw new Error("expected a rotation then an affine transform");
  }
  assertEquals(transforms[0].path, "transforms/rotation");
  assertEquals(transforms[1].path, "transforms/affine");
});

// An input/output `name` is retained only when it names a known coordinate
// system; an unknown name is dropped while a `path` reference is always kept.
// Mirrors the Python _parse_transforms identifier resolution.
Deno.test("v0.6 read drops unknown coordinate-system names, keeps paths", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  // Injected into the written document rather than passed to the writer: a
  // multiscale-level transform naming no output system is exactly what the
  // 0.6 writer refuses, and this test is about what the reader does with one.
  mutateRootEntry(store, (entry) => {
    entry.coordinateTransformations = [{
      type: "scale",
      scale: [2.0, 2.0, 2.0],
      input: { name: "does-not-exist" },
      output: { path: "scale0" },
    }];
  });
  const imported = await fromOmeZarr(store);

  const transform = imported.metadata.coordinateTransformations![0];
  // Unknown input name -> no identifier survives; path output -> kept.
  assertEquals(transform.input, undefined);
  assertEquals(transform.output, { path: "scale0" });
});

// From 0.6 a multiscale-level transform maps between two named coordinate
// systems and the schema requires both references, so the writer refuses the
// model rather than produce a store its own validated reader rejects.
Deno.test("v0.6 write refuses a top-level transform without references", async () => {
  const multiscales = await buildMultiscales();
  multiscales.metadata.coordinateTransformations = [{
    type: "scale",
    scale: [2.0, 2.0, 2.0],
  }];

  const store: MemoryStore = new Map();
  await assertRejects(
    () => toOmeZarr(store, multiscales, { version: "0.6" }),
    Error,
    "names no input coordinate system",
  );
});

// A dataset whose transforms are a bare [scale, translation] (not wrapped in a
// sequence, as another writer might emit) must still yield the per-axis scale
// and translation. Exercises the non-sequence branch of extractScaleTranslation.
Deno.test("v0.6 read handles bare per-dataset scale and translation", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });

  mutateRootEntry(store, (entry) => {
    const datasets = entry.datasets as Array<Record<string, unknown>>;
    datasets.forEach((ds, index) => {
      const factor = 2 ** index;
      ds.coordinateTransformations = [
        { type: "scale", scale: [factor, factor, factor] },
        { type: "translation", translation: [index, index, index] },
      ];
    });
  });

  const imported = await fromOmeZarr(store);
  assertEquals(imported.images[0].scale, { z: 1, y: 1, x: 1 });
  assertEquals(imported.images[0].translation, { z: 0, y: 0, x: 0 });
  assertEquals(imported.images[1].scale, { z: 2, y: 2, x: 2 });
  assertEquals(imported.images[1].translation, { z: 1, y: 1, x: 1 });
});

Deno.test("v0.6 read rejects metadata missing coordinateSystems", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  mutateRootEntry(store, (entry) => {
    delete entry.coordinateSystems;
  });
  await assertRejects(
    () => fromOmeZarr(store),
    Error,
    "coordinateSystems",
  );
});

Deno.test("v0.6 read rejects empty datasets", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  mutateRootEntry(store, (entry) => {
    entry.datasets = [];
  });
  await assertRejects(
    () => fromOmeZarr(store),
    Error,
    "datasets",
  );
});

Deno.test("v0.6 read rejects a malformed scale transform", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  mutateRootEntry(store, (entry) => {
    // `scale` must be an array of numbers; a string is rejected rather than
    // passed through an unchecked cast.
    entry.coordinateTransformations = [{
      type: "scale",
      scale: "not-an-array",
      input: { name: "intrinsic" },
      output: { name: "intrinsic" },
    }];
  });
  await assertRejects(
    () => fromOmeZarr(store),
    Error,
    "must be an array of numbers",
  );
});

Deno.test("v0.6 read rejects an unknown transform type", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  mutateRootEntry(store, (entry) => {
    entry.coordinateTransformations = [{
      type: "notARealTransform",
      input: { name: "intrinsic" },
      output: { name: "intrinsic" },
    }];
  });
  await assertRejects(
    () => fromOmeZarr(store),
    Error,
    "Unsupported transform type",
  );
});

// Anatomical orientation (RFC 4) and axis units carried on the intrinsic
// coordinate system survive a v0.6 round-trip.
Deno.test("v0.6 round-trips axis units and anatomical orientation", async () => {
  const shape = [16, 16, 16];
  const data = new Uint8Array(shape[0] * shape[1] * shape[2]);
  const base = await toNgffImage(data, { dims: ["z", "y", "x"], shape });
  const image = new NgffImage({
    data: base.data,
    dims: base.dims,
    scale: base.scale,
    translation: base.translation,
    name: base.name,
    axesUnits: { z: "micrometer", y: "micrometer", x: "micrometer" },
    axesOrientations: {
      z: {
        type: "anatomical",
        value: AnatomicalOrientationValues.InferiorToSuperior,
      },
      y: {
        type: "anatomical",
        value: AnatomicalOrientationValues.AnteriorToPosterior,
      },
      x: { type: "anatomical", value: AnatomicalOrientationValues.RightToLeft },
    },
    computedCallbacks: undefined,
  });
  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    chunks: 8,
  });

  const store: MemoryStore = new Map();
  // RFC-4 anatomical orientation is written automatically when present on the
  // image; there is no enabled-RFCs opt-in.
  await toOmeZarr(store, multiscales, { version: "0.6" });
  const imported = await fromOmeZarr(store, { version: "0.6", validate: true });

  assertEquals(imported.images[0].axesUnits, {
    z: "micrometer",
    y: "micrometer",
    x: "micrometer",
  });
  assertEquals(
    imported.images[0].axesOrientations?.x,
    { type: "anatomical", value: AnatomicalOrientationValues.RightToLeft },
  );
  const intrinsicAxes = imported.metadata.coordinateSystems![0].axes;
  assertEquals(intrinsicAxes[2].unit, "micrometer");
});

// The requested-version mismatch is a validation concern: it is enforced only
// under `validate`, consistent with the node reader and the v0.4/v0.5 path.
Deno.test("browser reader gates v0.6 version mismatch behind validate", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarrBrowser(store, multiscales, { version: "0.6" });

  // Without validation, a mismatched requested version does not throw.
  const lenient = await fromOmeZarrBrowser(store, { version: "0.4" });
  assertEquals(lenient.metadata.version, "0.6");

  // With validation, the mismatch is reported.
  await assertRejects(
    () => fromOmeZarrBrowser(store, { version: "0.4", validate: true }),
    Error,
    "Expected OME-Zarr version 0.4",
  );
});

// Under validate=true the v0.6 reader runs the same structural pass as v0.4/v0.5
// over the normalized metadata, so a dataset scale whose length disagrees with
// the axis count is rejected.
Deno.test("v0.6 read validates per-dataset scale length under validate", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  mutateRootEntry(store, (entry) => {
    const datasets = entry.datasets as Array<Record<string, unknown>>;
    const seq = (datasets[0].coordinateTransformations as Array<
      Record<string, unknown>
    >)[0];
    const transforms = seq.transformations as Array<Record<string, unknown>>;
    // Drop a component so the scale vector no longer matches the 3 axes.
    transforms[0].scale = [2.0, 2.0];
  });

  // Lenient read tolerates it; strict validation rejects it.
  const lenient = await fromOmeZarr(store);
  assertEquals(lenient.images.length, multiscales.images.length);
  await assertRejects(
    () => fromOmeZarr(store, { validate: true }),
    Error,
  );
});

// RFC-9 (.ozx) output is always v0.5, which cannot represent v0.6-only
// transforms; prepareRfc9Metadata must reduce them to the scale/translation
// subset so they never leak into the store.
Deno.test("RFC-9 metadata drops a v0.6-only rotation but keeps a scale", async () => {
  const withRotation = await buildMultiscales();
  withRotation.metadata.coordinateTransformations = [createRotation([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
  ])];
  const rotEntry = prepareRfc9Metadata(withRotation);
  // No representable top-level transform remains, so the field is omitted.
  assertEquals("coordinateTransformations" in rotEntry, false);

  const withScale = await buildMultiscales();
  withScale.metadata.coordinateTransformations = [createScale([2, 2, 2])];
  const scaleEntry = prepareRfc9Metadata(withScale);
  const kept = scaleEntry.coordinateTransformations as Array<
    Record<string, unknown>
  >;
  assertEquals(kept.length, 1);
  assertEquals(kept[0].type, "scale");
  // The v0.6-only input/output/name fields are stripped from the kept transform.
  assertEquals("input" in kept[0], false);
  assertEquals("output" in kept[0], false);
});

// --- OME-Zarr v0.6 pre-release version tag handling ---

// Writing version "0.6" tags the store with the pre-release "0.6rc0" on disk
// while keeping the in-memory metadata version at "0.6".
Deno.test("v0.6 write tags the store 0.6rc0 but reads back as 0.6", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });

  const ome = readOmeAttributes(store) as { version: string };
  assertEquals(ome.version, "0.6rc0");

  const imported = await fromOmeZarr(store);
  assertEquals(imported.metadata.version, "0.6");
});

// Round-trip with an explicit requested version "0.6" and validation: the
// pre-release "0.6rc0" on disk must satisfy the requested "0.6" (family match).
Deno.test("v0.6 round-trip with validate accepts the 0.6rc0 on-disk tag", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });

  const imported = await fromOmeZarr(store, { version: "0.6", validate: true });
  assertEquals(imported.metadata.version, "0.6");
  assertEquals(imported.images.length, multiscales.images.length);
});

// A store tagged with a plain "0.6" (e.g. a stricter writer) still reads as
// v0.6 -- isV06Version covers the whole family, not just the draft tag.
Deno.test("reader accepts a plain 0.6 on-disk version tag", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  setOmeVersion(store, "0.6");

  const imported = await fromOmeZarr(store, { version: "0.6", validate: true });
  assertEquals(imported.metadata.version, "0.6");
  assertEquals(imported.metadata.coordinateSystems![0].name, "intrinsic");
});

// A future draft tag in the 0.6 family (e.g. "0.6.dev5") is also read as v0.6.
Deno.test("reader accepts a future 0.6 draft tag", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  setOmeVersion(store, "0.6.dev5");

  const imported = await fromOmeZarr(store);
  assertEquals(imported.metadata.version, "0.6");
});

// The browser reader follows the same family rule for the draft tag.
Deno.test("browser reader accepts the 0.6.dev4 on-disk tag", async () => {
  const multiscales = await buildMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarrBrowser(store, multiscales, { version: "0.6" });

  const ome = readOmeAttributes(store) as { version: string };
  assertEquals(ome.version, "0.6rc0");

  const imported = await fromOmeZarrBrowser(store, { version: "0.6" });
  assertEquals(imported.metadata.version, "0.6");
  assertEquals(imported.images.length, multiscales.images.length);
});

// Both 0.6 pre-release tags are recognized, supported version strings: the
// current one is written, the earlier one is still read.
Deno.test("the 0.6 pre-release tags are supported versions", () => {
  assertEquals(isSupportedVersion("0.6rc0"), true);
  assertEquals(isV06Version("0.6rc0"), true);
  assertEquals(isSupportedVersion("0.6.dev4"), true);
  assertEquals(isV06Version("0.6.dev4"), true);
  assertEquals(isV06Version("0.6"), true);
  assertEquals(isV06Version("0.5"), false);
});

// ngff-zarr 0.29.0 wrote the byDimension axis keys in snake_case; the spec
// spells them inputAxes and outputAxes, which is what is written now. A store
// from that release must still read.
Deno.test("v0.6 read accepts the snake_case byDimension axis keys", async () => {
  const multiscales = await buildMultiscales();
  const intrinsic = multiscales.metadata.coordinateSystems![0];
  const byDimension = createByDimension([
    {
      transformation: createScale([2.0, 3.0]),
      inputAxes: [0, 1],
      outputAxes: [0, 1],
    },
    {
      transformation: createTranslation([5.0]),
      inputAxes: [2],
      outputAxes: [2],
    },
  ]);
  byDimension.input = { name: intrinsic.name };
  byDimension.output = { name: intrinsic.name };
  multiscales.metadata.coordinateTransformations = [byDimension];

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  mutateRootEntry(store, (entry) => {
    const [written] = entry.coordinateTransformations as Array<
      Record<string, unknown>
    >;
    for (
      const item of written.transformations as Array<Record<string, unknown>>
    ) {
      item.input_axes = item.inputAxes;
      item.output_axes = item.outputAxes;
      delete item.inputAxes;
      delete item.outputAxes;
    }
  });

  const imported = await fromOmeZarr(store);
  const transform = imported.metadata.coordinateTransformations![0];
  if (transform.type !== "byDimension") throw new Error("expected byDimension");
  assertEquals(transform.transformations.map((item) => item.inputAxes), [
    [0, 1],
    [2],
  ]);
  assertEquals(transform.transformations.map((item) => item.outputAxes), [[
    0,
    1,
  ], [2]]);
});

// Mirrors test_coordinate_transformations.py: the mapAxis, byDimension and
// bijection payloads survive the round-trip by value, not just by type.
Deno.test("mapAxis, byDimension and bijection payloads survive the round-trip", async () => {
  const multiscales = await buildMultiscales();
  const intrinsic = multiscales.metadata.coordinateSystems![0];

  const mapAxis = createMapAxis([2, 0, 1]);
  const byDimension = createByDimension([
    {
      transformation: createScale([2.0, 3.0]),
      inputAxes: [0, 1],
      outputAxes: [0, 1],
    },
    {
      transformation: createTranslation([5.0]),
      inputAxes: [2],
      outputAxes: [2],
    },
  ]);
  const bijection = createBijection(
    { type: "displacements", path: "forward_field" },
    { type: "displacements", path: "inverse_field" },
  );
  for (const t of [mapAxis, byDimension, bijection]) {
    t.input = { name: intrinsic.name };
    t.output = { name: intrinsic.name };
  }
  multiscales.metadata.coordinateTransformations = [
    mapAxis,
    byDimension,
    bijection,
  ];

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  const imported = await fromOmeZarr(store);

  const transforms = imported.metadata.coordinateTransformations!;
  assertEquals(transforms.length, 3);
  const [importedMapAxis, importedByDimension, importedBijection] = transforms;
  if (importedMapAxis.type !== "mapAxis") {
    throw new Error("expected a mapAxis transform");
  }
  if (importedByDimension.type !== "byDimension") {
    throw new Error("expected a byDimension transform");
  }
  if (importedBijection.type !== "bijection") {
    throw new Error("expected a bijection transform");
  }
  assertEquals(importedMapAxis.mapAxis, [2, 0, 1]);
  assertEquals(importedByDimension.transformations.length, 2);
  const [first, second] = importedByDimension.transformations;
  assertEquals(first.inputAxes, [0, 1]);
  assertEquals(first.outputAxes, [0, 1]);
  if (first.transformation.type !== "scale") {
    throw new Error("expected a scale item transformation");
  }
  assertEquals(first.transformation.scale, [2.0, 3.0]);
  assertEquals(second.inputAxes, [2]);
  assertEquals(second.outputAxes, [2]);
  if (importedBijection.forward.type !== "displacements") {
    throw new Error("expected a displacements forward transformation");
  }
  assertEquals(importedBijection.forward.path, "forward_field");
  if (importedBijection.inverse.type !== "displacements") {
    throw new Error("expected a displacements inverse transformation");
  }
  assertEquals(importedBijection.inverse.path, "inverse_field");
});

// The RFC 5 constraints the schema cannot express are enforced at read time.
Deno.test("invalid mapAxis, byDimension and bijection metadata are rejected", async () => {
  const cases: Array<{ name: string; transform: V06Transform }> = [
    // Not a permutation: index 1 missing, index 2 duplicated.
    { name: "mapAxis gap", transform: createMapAxis([2, 0, 2]) },
    // Output axis 1 produced by two items.
    {
      name: "byDimension duplicate output",
      transform: createByDimension([
        {
          transformation: createScale([2.0, 3.0]),
          inputAxes: [0, 1],
          outputAxes: [0, 1],
        },
        {
          transformation: createTranslation([5.0]),
          inputAxes: [2],
          outputAxes: [1],
        },
      ]),
    },
    // A 2D scale mapping a single axis.
    {
      name: "byDimension dimension mismatch",
      transform: createByDimension([
        {
          transformation: createScale([2.0, 3.0]),
          inputAxes: [0],
          outputAxes: [0],
        },
      ]),
    },
  ];

  for (const { name, transform } of cases) {
    const multiscales = await buildMultiscales();
    const intrinsic = multiscales.metadata.coordinateSystems![0];
    transform.input = { name: intrinsic.name };
    transform.output = { name: intrinsic.name };
    multiscales.metadata.coordinateTransformations = [transform];

    const store: MemoryStore = new Map();
    await toOmeZarr(store, multiscales, { version: "0.6" });
    await assertRejects(
      () => fromOmeZarr(store),
      Error,
      undefined,
      `case '${name}' should be rejected`,
    );
  }
});

// byDimension items must cover every output axis of a resolved coordinate
// system exactly once; two items over three output axes leave one uncovered.
Deno.test("byDimension incomplete output coverage is rejected", async () => {
  const multiscales = await buildMultiscales();
  const intrinsic = multiscales.metadata.coordinateSystems![0];
  const byDimension = createByDimension([
    {
      transformation: createScale([2.0, 3.0]),
      inputAxes: [0, 1],
      outputAxes: [0, 1],
    },
  ]);
  byDimension.input = { name: intrinsic.name };
  byDimension.output = { name: intrinsic.name };
  multiscales.metadata.coordinateTransformations = [byDimension];

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  await assertRejects(() => fromOmeZarr(store), Error, "exactly once");
});

// A byDimension item may wrap any transformation, including a sequence; the
// round-trip and the zod schema both accept the nesting.
Deno.test("byDimension items nest wrapper transformations", async () => {
  const multiscales = await buildMultiscales();
  const intrinsic = multiscales.metadata.coordinateSystems![0];
  const byDimension = createByDimension([
    {
      transformation: createTransformSequence([
        createScale([2.0, 3.0]),
        createTranslation([1.0, 1.0]),
      ]),
      inputAxes: [0, 1],
      outputAxes: [0, 1],
    },
    {
      transformation: createTranslation([5.0]),
      inputAxes: [2],
      outputAxes: [2],
    },
  ]);
  byDimension.input = { name: intrinsic.name };
  byDimension.output = { name: intrinsic.name };
  multiscales.metadata.coordinateTransformations = [byDimension];

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.6" });
  const imported = await fromOmeZarr(store);

  const transform = imported.metadata.coordinateTransformations![0];
  if (transform.type !== "byDimension") {
    throw new Error("expected a byDimension transform");
  }
  const nested = transform.transformations[0].transformation;
  if (nested.type !== "sequence") {
    throw new Error("expected a nested sequence transformation");
  }
  assertEquals(nested.transformations.length, 2);

  const parsed = CoordinateTransformationSchema.parse({
    type: "byDimension",
    transformations: [
      {
        transformation: {
          type: "sequence",
          transformations: [{ type: "scale", scale: [2.0, 3.0] }],
        },
        inputAxes: [0, 1],
        outputAxes: [0, 1],
      },
    ],
  });
  assertEquals(parsed.type, "byDimension");
});

// Programmatically built transforms are validated with the same rules the
// reader applies to on-disk metadata.
Deno.test("validateV06Transform rejects fractional and out-of-range axes", () => {
  assertThrows(
    () => validateV06Transform(createMapAxis([0.5, 1, 2]), []),
    Error,
    "integers",
  );

  const threeAxisSystem = createCoordinateSystem("system", [
    createAxis("z", "space"),
    createAxis("y", "space"),
    createAxis("x", "space"),
  ]);
  const outOfRange = createByDimension([
    {
      transformation: createScale([2.0, 3.0, 4.0]),
      inputAxes: [0, 1, 3],
      outputAxes: [0, 1, 2],
    },
  ]);
  outOfRange.input = { name: "system" };
  outOfRange.output = { name: "system" };
  assertThrows(
    () => validateV06Transform(outOfRange, [threeAxisSystem]),
    Error,
    "exceed",
  );
});

// OME-Zarr coordinate systems hold 2 to 5 axes; a mapAxis of another length
// is rejected on read as well as by the zod schema.
Deno.test("validateV06Transform bounds the mapAxis arity", () => {
  for (const indices of [[0], [], [5, 4, 3, 2, 1, 0]]) {
    assertThrows(
      () => validateV06Transform(createMapAxis(indices), []),
      Error,
      "between 2 and 5",
    );
  }
});

// rfc5_transform_cases.json is shared with the Python suite: both readers must
// give the same verdict on every case, so a rule enforced in one port and not
// the other fails here.
Deno.test("shared RFC-5 cases match the expected verdict", async () => {
  const path = fromFileUrl(
    new URL("../../py/test/rfc5_transform_cases.json", import.meta.url),
  );
  const spec = JSON.parse(await Deno.readTextFile(path));
  const systems = spec.coordinateSystems as CoordinateSystem[];
  const names = systems.map((system) => system.name);
  for (const testCase of spec.cases) {
    const parse = () =>
      parseV06Transforms([testCase.transformation], names, systems);
    if (testCase.ok) {
      parse();
    } else {
      assertThrows(parse, Error, undefined, testCase.name);
    }
  }
});
