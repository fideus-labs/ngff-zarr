// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
//
// Tests for upgradeOmeZarr, mirroring the Python upgrade_ome_zarr matrix:
// in-place same-format rewrites (chunks byte-identical, issue #219),
// write-to-new-store for every representative transition (source untouched),
// the cross-format in-place guidance error, and the same-version no-op.
import { assertEquals, assertExists, assertRejects } from "@std/assert";
import * as zarr from "zarrita";
import {
  fromOmeZarr,
  type MemoryStore,
  toOmeZarr,
  upgradeOmeZarr,
} from "../src/mod.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createNgffMultiscales,
} from "../src/utils/factory.ts";
import { NgffImage } from "../src/types/ngff_image.ts";

const SHAPE = [1, 2, 8, 8]; // c, z, y, x
const CHUNKS = [1, 1, 8, 8];
const STRIDE = [2 * 8 * 8, 8 * 8, 8, 1];

/** Deterministic uint16 payload so "array data intact" can be asserted exactly. */
function fixtureData(): Uint16Array {
  const n = SHAPE.reduce((a, b) => a * b, 1);
  const data = new Uint16Array(n);
  for (let i = 0; i < n; i++) data[i] = (i * 7 + 3) % 60000;
  return data;
}

/** Write a fresh synthetic single-scale image to a MemoryStore at `version`. */
async function makeSourceStore(
  version: "0.4" | "0.5" | "0.6",
): Promise<MemoryStore> {
  const seed: MemoryStore = new Map();
  const seedArray = await zarr.create(zarr.root(seed).resolve("seed"), {
    shape: SHAPE,
    chunk_shape: CHUNKS,
    data_type: "uint16" as zarr.DataType,
    fill_value: 0,
  });
  await zarr.set(seedArray, null, {
    data: fixtureData(),
    shape: SHAPE,
    stride: STRIDE,
  });

  const image = new NgffImage({
    data: seedArray,
    dims: ["c", "z", "y", "x"],
    scale: { c: 1, z: 1, y: 1, x: 1 },
    translation: { c: 0, z: 0, y: 0, x: 0 },
    name: "upgrade-fixture",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
  const axes = [
    createAxis("c", "channel"),
    createAxis("z", "space", "micrometer"),
    createAxis("y", "space", "micrometer"),
    createAxis("x", "space", "micrometer"),
  ];
  const datasets = [createDataset("0", [1, 1, 1, 1], [0, 0, 0, 0])];
  const metadata = createMetadata(axes, datasets, "upgrade-fixture", version);
  const multiscales = createNgffMultiscales([image], metadata);

  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version });
  return store;
}

const METADATA_SIDECARS = [
  "zarr.json",
  ".zarray",
  ".zattrs",
  ".zgroup",
  ".zmetadata",
];

/** Keys holding array chunk *data* (everything but the metadata sidecars). */
function chunkKeys(store: MemoryStore): string[] {
  return [...store.keys()]
    .filter((key) => !METADATA_SIDECARS.some((name) => key.endsWith(name)))
    .sort();
}

/** Snapshot every key -> bytes so store immutability can be asserted. */
function snapshot(store: MemoryStore): Map<string, string> {
  const snap = new Map<string, string>();
  for (const [key, value] of store) snap.set(key, [...value].join(","));
  return snap;
}

function assertSnapshotsEqual(
  before: Map<string, string>,
  after: Map<string, string>,
): void {
  assertEquals([...after.keys()].sort(), [...before.keys()].sort());
  for (const [key, value] of before) {
    assertEquals(after.get(key), value, `store key changed: ${key}`);
  }
}

function assertChunksUnchanged(
  store: MemoryStore,
  before: Map<string, string>,
  chunksBefore: string[],
): void {
  assertEquals(chunkKeys(store), chunksBefore);
  for (const key of chunksBefore) {
    assertEquals(
      [...store.get(key)!].join(","),
      before.get(key),
      `array chunk changed: ${key}`,
    );
  }
}

/** Read the root-group `ome` namespace attributes from a store. */
async function readOme(store: MemoryStore): Promise<Record<string, unknown>> {
  const group = await zarr.open(zarr.root(store), { kind: "group" });
  return (group.attrs as Record<string, Record<string, unknown>>).ome;
}

/** Re-read (validated) and return the full first-scale payload as uint16. */
async function readImageData(
  store: MemoryStore,
  version: "0.4" | "0.5" | "0.6",
): Promise<Uint16Array> {
  const multiscales = await fromOmeZarr(store, { validate: true, version });
  assertEquals(multiscales.images.length, 1);
  const { data } = await zarr.get(multiscales.images[0].data, null);
  return data as Uint16Array;
}

function assertDataIntact(actual: Uint16Array): void {
  const expected = fixtureData();
  assertEquals(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(actual[i], expected[i], `pixel ${i} differs`);
  }
}

// The on-disk `ome.version` string a given API version is written as.
const DISK_VERSION: Record<"0.4" | "0.5" | "0.6", string> = {
  "0.4": "0.4",
  "0.5": "0.5",
  "0.6": "0.6.dev4",
};

Deno.test("in-place same-format upgrade 0.5 -> 0.6 preserves chunks", async () => {
  const store = await makeSourceStore("0.5");
  assertEquals((await readOme(store)).version, "0.5");
  const chunksBefore = chunkKeys(store);
  assertEquals(chunksBefore.length > 0, true, "expected chunk data before");
  const before = snapshot(store);

  await upgradeOmeZarr(store, { version: "0.6" });

  const ome = await readOme(store);
  assertEquals(ome.version, "0.6.dev4");
  const ms0 = (ome.multiscales as Record<string, unknown>[])[0];
  assertExists(ms0.coordinateSystems);

  assertChunksUnchanged(store, before, chunksBefore);
  assertDataIntact(await readImageData(store, "0.6"));
});

Deno.test("in-place same-format downgrade 0.6 -> 0.5 preserves chunks", async () => {
  const store = await makeSourceStore("0.6");
  assertEquals((await readOme(store)).version, "0.6.dev4");
  const chunksBefore = chunkKeys(store);
  const before = snapshot(store);

  await upgradeOmeZarr(store, { version: "0.5" });

  const ome = await readOme(store);
  assertEquals(ome.version, "0.5");
  const ms0 = (ome.multiscales as Record<string, unknown>[])[0];
  assertExists(ms0.axes);
  assertEquals("coordinateSystems" in ms0, false);

  assertChunksUnchanged(store, before, chunksBefore);
  assertDataIntact(await readImageData(store, "0.5"));
});

for (
  const [source, target] of [
    ["0.4", "0.5"],
    ["0.4", "0.6"],
    ["0.5", "0.6"],
  ] as const
) {
  Deno.test(
    `write-to-new-store ${source} -> ${target} leaves the source unchanged`,
    async () => {
      const store = await makeSourceStore(source);
      const before = snapshot(store);
      const target_store: MemoryStore = new Map();

      await upgradeOmeZarr(store, { output: target_store, version: target });

      // The source store is left completely unchanged.
      assertSnapshotsEqual(before, snapshot(store));

      // The new store reads back validated at the requested version.
      assertEquals((await readOme(target_store)).version, DISK_VERSION[target]);
      assertDataIntact(await readImageData(target_store, target));
    },
  );
}

Deno.test("in-place cross-format upgrade 0.4 -> 0.5 throws guidance", async () => {
  const store = await makeSourceStore("0.4");
  const before = snapshot(store);

  await assertRejects(
    () => upgradeOmeZarr(store, { version: "0.5" }),
    Error,
    "output",
  );

  // The rejected cross-format upgrade left the store untouched.
  assertSnapshotsEqual(before, snapshot(store));
});

Deno.test("in-place no-op (same version) leaves the store unchanged", async () => {
  const store = await makeSourceStore("0.5");
  const before = snapshot(store);

  await upgradeOmeZarr(store, { version: "0.5" });

  assertSnapshotsEqual(before, snapshot(store));
});

Deno.test("in-place no-op with default target version on a 0.6 store", async () => {
  const store = await makeSourceStore("0.6");
  const before = snapshot(store);

  // version defaults to "0.6"; the 0.6 store is already at the target.
  await upgradeOmeZarr(store);

  assertSnapshotsEqual(before, snapshot(store));
});
