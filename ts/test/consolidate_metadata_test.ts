// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * `consolidateMetadata` can skip consolidation.
 *
 * Consolidated metadata is incompatible with backends that run their own
 * consolidation, such as Icechunk, so `toOmeZarr` must be able to skip it.
 * Mirrors `py/test/test_consolidate_metadata.py`.
 */
import { assert, assertEquals, assertFalse } from "@std/assert";
import { join } from "@std/path";
import * as zarr from "zarrita";

import { ZipFileStore } from "@zarrita/storage";

import {
  fromOmeZarr,
  type MemoryStore,
  toOmeZarr,
  toOmeZarrOzxData,
  upgradeOmeZarr,
} from "../src/mod.ts";
import { readOzxJsonFirst } from "../src/io/rfc9_zip.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createNgffMultiscales,
} from "../src/utils/factory.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import type { NgffMultiscales } from "../src/types/multiscales.ts";

const SHAPE = [8, 8];
const OUTPUT_DIR = join(Deno.cwd(), "test", "output");

await Deno.mkdir(OUTPUT_DIR, { recursive: true });

/** A two-level pyramid whose dataset paths are `scale0` and `scale1`. */
async function fixture(
  version: "0.4" | "0.5" | "0.6" = "0.5",
): Promise<NgffMultiscales> {
  const seed: MemoryStore = new Map();
  const images: NgffImage[] = [];
  for (let level = 0; level < 2; level++) {
    const shape = SHAPE.map((size) => size >> level);
    const array = await zarr.create(zarr.root(seed).resolve(`seed${level}`), {
      shape,
      chunk_shape: shape,
      data_type: "uint16" as zarr.DataType,
      fill_value: 0,
    });
    const scale = 1 << level;
    images.push(
      new NgffImage({
        data: array,
        dims: ["y", "x"],
        scale: { y: scale, x: scale },
        translation: { y: 0, x: 0 },
        name: "consolidate-fixture",
        axesUnits: undefined,
        computedCallbacks: undefined,
      }),
    );
  }
  const axes = [
    createAxis("y", "space", "micrometer"),
    createAxis("x", "space", "micrometer"),
  ];
  const datasets = images.map((_, level) =>
    createDataset(`scale${level}`, [1 << level, 1 << level], [0, 0])
  );
  const metadata = createMetadata(
    axes,
    datasets,
    "consolidate-fixture",
    version,
  );
  return createNgffMultiscales(images, metadata);
}

/** The root group's `consolidated_metadata` block, or `undefined`. */
function consolidatedBlock(
  store: MemoryStore,
): Record<string, unknown> | undefined {
  const raw = store.get("/zarr.json");
  assert(raw !== undefined, "store has no root zarr.json");
  const root = JSON.parse(new TextDecoder().decode(raw));
  return root.consolidated_metadata;
}

Deno.test("toOmeZarr consolidates metadata by default", async () => {
  const multiscales = await fixture();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.5" });

  const block = consolidatedBlock(store);
  assert(block !== undefined, "default write left the store unconsolidated");
  assertEquals(block.kind, "inline");
  assertEquals(block.must_understand, false);

  // Every array the writer wrote is inlined, keyed by its dataset path.
  const inlined = block.metadata as Record<string, Record<string, unknown>>;
  assertEquals(Object.keys(inlined).sort(), ["scale0", "scale1"]);
  for (const [path, entry] of Object.entries(inlined)) {
    const raw = store.get(`/${path}/zarr.json`);
    assert(raw !== undefined, `missing array document for ${path}`);
    const document = JSON.parse(new TextDecoder().decode(raw));
    assertEquals(entry.node_type, "array");
    assertEquals(entry.shape, document.shape);
    assertEquals(entry.data_type, document.data_type);
    assertEquals(entry.codecs, document.codecs);
    // Optional fields zarr-python always materializes on a consolidated entry.
    assertEquals(entry.attributes, {});
    assertEquals(entry.storage_transformers, []);
  }
});

Deno.test("toOmeZarr skips consolidation on request", async () => {
  const multiscales = await fixture();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, {
    version: "0.5",
    consolidateMetadata: false,
  });

  assertEquals(consolidatedBlock(store), undefined);
});

Deno.test("skipping consolidation leaves the arrays untouched", async () => {
  const multiscales = await fixture();
  const consolidated: MemoryStore = new Map();
  const plain: MemoryStore = new Map();
  await toOmeZarr(consolidated, multiscales, { version: "0.5" });
  await toOmeZarr(plain, multiscales, {
    version: "0.5",
    consolidateMetadata: false,
  });

  // The flag governs the root document alone: every other key is identical.
  assertEquals([...plain.keys()].sort(), [...consolidated.keys()].sort());
  for (const key of plain.keys()) {
    if (key === "/zarr.json") continue;
    assertEquals(
      [...plain.get(key)!].join(","),
      [...consolidated.get(key)!].join(","),
      `store key changed: ${key}`,
    );
  }
});

Deno.test("an unconsolidated store still reads back", async () => {
  const multiscales = await fixture();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, {
    version: "0.5",
    consolidateMetadata: false,
  });

  const read = await fromOmeZarr(store);
  assertEquals(read.images.length, 2);
  assertEquals(read.images[0].data.shape, SHAPE);
});

// A Zarr v3 write replaces the root document wholesale, so re-writing a
// consolidated store without consolidation drops the block instead of leaving
// it stale. This is the invariant Python's zarr v2 guard exists to protect and
// that its v3 path relies on; the TypeScript writer is v3-only, so it holds
// unconditionally here.
Deno.test("re-writing without consolidation self-cleans", async () => {
  const multiscales = await fixture();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.5" });
  assert(consolidatedBlock(store) !== undefined);

  await toOmeZarr(store, multiscales, {
    version: "0.5",
    consolidateMetadata: false,
  });
  assertEquals(consolidatedBlock(store), undefined);
});

Deno.test("in-place upgradeOmeZarr refreshes consolidated metadata", async () => {
  const multiscales = await fixture("0.5");
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, { version: "0.5" });

  await upgradeOmeZarr(store, { version: "0.6" });

  const block = consolidatedBlock(store);
  assert(
    block !== undefined,
    "the re-tag silently de-consolidated the store",
  );
  assertEquals(Object.keys(block.metadata as object).sort(), [
    "scale0",
    "scale1",
  ]);
});

Deno.test("in-place upgradeOmeZarr leaves an unconsolidated store alone", async () => {
  const multiscales = await fixture("0.5");
  const store: MemoryStore = new Map();
  await toOmeZarr(store, multiscales, {
    version: "0.5",
    consolidateMetadata: false,
  });

  await upgradeOmeZarr(store, { version: "0.6" });

  assertEquals(consolidatedBlock(store), undefined);
});

/** The root group document inside a .ozx archive. */
async function ozxRoot(zipData: Uint8Array): Promise<Record<string, unknown>> {
  const store = ZipFileStore.fromBlob(new Blob([zipData as BlobPart]));
  const raw = await store.get("/zarr.json");
  assert(raw !== undefined, ".ozx archive has no root zarr.json");
  return JSON.parse(new TextDecoder().decode(raw));
}

Deno.test(".ozx consolidates by default and can skip", async () => {
  const multiscales = await fixture("0.5");

  const consolidatedZip = await toOmeZarrOzxData(multiscales);
  const consolidatedRoot = await ozxRoot(consolidatedZip);
  const block = consolidatedRoot.consolidated_metadata as
    | Record<string, unknown>
    | undefined;
  assert(
    block !== undefined,
    ".ozx default write left the store unconsolidated",
  );
  assertEquals(Object.keys(block.metadata as object).sort(), [
    "scale0",
    "scale1",
  ]);

  const plainZip = await toOmeZarrOzxData(multiscales, {
    consolidateMetadata: false,
  });
  const plainRoot = await ozxRoot(plainZip);
  assertFalse("consolidated_metadata" in plainRoot);

  // RFC-9 requires the root document to be the archive's first entry; the
  // extra root rewrite consolidation performs must not disturb that.
  assert(readOzxJsonFirst(consolidatedZip), "root zarr.json is not first");
  assert(readOzxJsonFirst(plainZip), "root zarr.json is not first");
});

Deno.test("toOmeZarr consolidates a filesystem store", async () => {
  const multiscales = await fixture("0.5");
  const path = join(OUTPUT_DIR, "consolidated.ome.zarr");
  await Deno.remove(path, { recursive: true }).catch(() => {});
  await toOmeZarr(path, multiscales, { version: "0.5" });

  const root = JSON.parse(await Deno.readTextFile(join(path, "zarr.json")));
  assertEquals(
    Object.keys(root.consolidated_metadata.metadata).sort(),
    ["scale0", "scale1"],
  );

  const plainPath = join(OUTPUT_DIR, "unconsolidated.ome.zarr");
  await Deno.remove(plainPath, { recursive: true }).catch(() => {});
  await toOmeZarr(plainPath, multiscales, {
    version: "0.5",
    consolidateMetadata: false,
  });
  const plainRoot = JSON.parse(
    await Deno.readTextFile(join(plainPath, "zarr.json")),
  );
  assertFalse("consolidated_metadata" in plainRoot);
});
