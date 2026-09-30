// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Zarr v3 sharding: the writers' `chunksPerShard`, and reading the shards
 * back -- through the codec workers (fizarrita) and through zarrita's own
 * sharding reader, which decode independently of each other.
 */

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { join } from "@std/path";
import * as zarr from "zarrita";
import { FileSystemStore, ZipFileStore } from "@zarrita/storage";

import { fromOmeZarr } from "../src/io/from_ngff_zarr.ts";
import { fromOmeZarr as fromOmeZarrBrowser } from "../src/io/from_ngff_zarr-browser.ts";
import { toOmeZarr } from "../src/io/to_ngff_zarr.ts";
import {
  toOmeZarr as toOmeZarrBrowser,
  toOmeZarrOzx as toOmeZarrOzxBrowser,
} from "../src/io/to_ngff_zarr-browser.ts";
import type { NgffMultiscales } from "../src/types/multiscales.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";
import {
  arrayLayout,
  ensureRangeReads,
  gateShardingVersion,
  outerChunkShape,
} from "../src/utils/sharding.ts";
import { zarrGet, zarrSet } from "../src/utils/worker_pool.ts";

// Neither axis a multiple of the chunk, nor of the shard: the last shard on
// each axis is partial, and its last inner chunk is too.
const SHAPE = [37, 45];
const CHUNKS = [8, 8];

function sourceData(): Uint16Array {
  return new Uint16Array(SHAPE[0] * SHAPE[1]).map((_, i) => i);
}

async function multiscales(): Promise<NgffMultiscales> {
  const store = new Map<string, Uint8Array>();
  const array = await zarr.create(zarr.root(store).resolve("source"), {
    shape: SHAPE,
    chunkShape: CHUNKS,
    dtype: "uint16",
    fillValue: 0,
  });
  await zarr.set(array, null, {
    data: sourceData(),
    shape: SHAPE,
    stride: [SHAPE[1], 1],
  });
  const image = new NgffImage({
    data: array,
    dims: ["y", "x"],
    scale: { y: 1.0, x: 1.0 },
    translation: { y: 0.0, x: 0.0 },
    name: "sharded",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
  const metadata = createMetadata(
    [createAxis("y", "space"), createAxis("x", "space")],
    [createDataset("0", [1.0, 1.0], [0.0, 0.0])],
    "sharded",
    "0.5",
  );
  return createMultiscales([image], metadata);
}

function assertSourceData(data: unknown): void {
  assertEquals(Array.from(data as Uint16Array), Array.from(sourceData()));
}

/** Read `array` whole through the codec workers and through zarrita alone. */
async function assertReadsBack(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
): Promise<void> {
  assertEquals(array.chunks, CHUNKS);
  assertSourceData((await zarrGet(array)).data);
  assertSourceData((await zarr.get(array)).data);
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

Deno.test("outerChunkShape - no chunksPerShard keeps the chunk grid", () => {
  assertEquals(outerChunkShape([100, 100], ["y", "x"], [8, 8], undefined), [
    8,
    8,
  ]);
});

Deno.test("outerChunkShape - one count, per-axis list, per-axis name", () => {
  const dims = ["z", "y", "x"];
  const shape = [100, 100, 100];
  assertEquals(outerChunkShape(shape, dims, [8, 8, 8], 2), [16, 16, 16]);
  assertEquals(outerChunkShape(shape, dims, [8, 8, 8], [1, 2, 4]), [
    8,
    16,
    32,
  ]);
  // An axis left out of the mapping spans one chunk.
  assertEquals(outerChunkShape(shape, dims, [8, 8, 8], { y: 2, x: 4 }), [
    8,
    16,
    32,
  ]);
});

Deno.test("outerChunkShape - a shard never outgrows the chunks its axis needs", () => {
  // As the Python writer does: the fewest whole chunks covering the axis, so
  // the chunk still divides the shard. 37 needs 5 chunks of 8, 3 needs 1.
  assertEquals(outerChunkShape([37, 3], ["y", "x"], [8, 8], 16), [40, 8]);
});

Deno.test("outerChunkShape - rejects malformed chunksPerShard", () => {
  assertThrows(
    () => outerChunkShape([8, 8], ["y", "x"], [4, 4], [2]),
    Error,
    "chunksPerShard must have length 2",
  );
  assertThrows(
    () => outerChunkShape([8, 8], ["y", "x"], [4, 4], 0),
    Error,
    "positive integers",
  );
  assertThrows(
    () => outerChunkShape([8, 8], ["y", "x"], [4, 4], { y: 1.5 }),
    Error,
    "positive integers",
  );
  // A misspelled axis would otherwise leave its intended axis unsharded.
  assertThrows(
    () => outerChunkShape([8, 8], ["y", "x"], [4, 4], { y: 2, xx: 2 }),
    Error,
    "chunksPerShard names axes the image does not have: xx (axes: y, x)",
  );
});

Deno.test("arrayLayout - wraps the chunk codecs in sharding_indexed", () => {
  const codecs = [{ name: "bytes", configuration: { endian: "little" } }];
  assertEquals(arrayLayout([37, 45], ["y", "x"], [8, 8], codecs, undefined), {
    chunkShape: [8, 8],
    codecs,
  });
  assertEquals(arrayLayout([37, 45], ["y", "x"], [8, 8], codecs, 2), {
    chunkShape: [16, 16],
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [8, 8],
          codecs,
          index_codecs: [
            { name: "bytes", configuration: { endian: "little" } },
            { name: "crc32c" },
          ],
          index_location: "end",
        },
      },
    ],
  });
});

Deno.test("gateShardingVersion - refuses sharding at 0.4 only", () => {
  assertThrows(
    () => gateShardingVersion("0.4", 2),
    Error,
    "Sharding is only supported for OME-Zarr version 0.5 and later",
  );
  gateShardingVersion("0.4", undefined);
  gateShardingVersion("0.5", 2);
  gateShardingVersion("0.6", { x: 2 });
});

// ---------------------------------------------------------------------------
// ensureRangeReads
// ---------------------------------------------------------------------------

Deno.test("ensureRangeReads - slices ranges out of a Map, which it stays", async () => {
  const map = new Map<string, Uint8Array>([
    ["/a", new Uint8Array([0, 1, 2, 3, 4, 5])],
  ]);
  const store = ensureRangeReads(map) as typeof map & Required<zarr.Readable>;

  assertEquals(store instanceof Map, true);
  assertEquals(
    await store.getRange("/a", { offset: 1, length: 3 }),
    new Uint8Array([1, 2, 3]),
  );
  assertEquals(
    await store.getRange("/a", { suffixLength: 2 }),
    new Uint8Array([4, 5]),
  );
  assertEquals(
    await store.getRange("/missing", { suffixLength: 2 }),
    undefined,
  );

  // Writes go through to the caller's Map.
  store.set("/b", new Uint8Array([7]));
  assertEquals(map.get("/b"), new Uint8Array([7]));
});

Deno.test("ensureRangeReads - fills in a getRange left undefined", async () => {
  const bytes = new Uint8Array([0, 1, 2, 3]);
  const store = {
    get: (_key: zarr.AbsolutePath) => Promise.resolve(bytes),
    getRange: undefined,
  };
  const wrapped = ensureRangeReads(store) as unknown as Required<
    zarr.AsyncReadable
  >;

  assertEquals(
    await wrapped.getRange("/a", { suffixLength: 1 }),
    new Uint8Array([3]),
  );
});

Deno.test("ensureRangeReads - leaves a store with range reads alone", () => {
  const store = new FileSystemStore(Deno.cwd());
  assertEquals(ensureRangeReads(store) === store, true);
});

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

Deno.test("toOmeZarr - chunksPerShard writes one object per shard", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const path = join(dir, "sharded.ome.zarr");
    await toOmeZarr(path, await multiscales(), {
      chunksPerShard: { y: 2, x: 2 },
    });

    const meta = JSON.parse(
      await Deno.readTextFile(join(path, "0", "zarr.json")),
    );
    assertEquals(meta.chunk_grid.configuration.chunk_shape, [16, 16]);
    assertEquals(meta.codecs.length, 1);
    assertEquals(meta.codecs[0].name, "sharding_indexed");
    assertEquals(meta.codecs[0].configuration.chunk_shape, CHUNKS);

    // A 3x3 grid of shards, not a 5x6 grid of chunks.
    let shards = 0;
    for (const row of Deno.readDirSync(join(path, "0", "c"))) {
      shards += [...Deno.readDirSync(join(path, "0", "c", row.name))].length;
    }
    assertEquals(shards, 9);

    const read = await fromOmeZarr(path);
    await assertReadsBack(read.images[0].data);

    // A selection that cuts across shards and inner chunks.
    const region = await zarrGet(read.images[0].data, [
      zarr.slice(5, 30),
      zarr.slice(11, 44),
    ]);
    const expected: number[] = [];
    for (let y = 5; y < 30; y++) {
      for (let x = 11; x < 44; x++) expected.push(y * SHAPE[1] + x);
    }
    assertEquals(Array.from(region.data as Uint16Array), expected);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("toOmeZarr - a sharded Map store reads back", async () => {
  const store = new Map<string, Uint8Array>();
  await toOmeZarr(store, await multiscales(), { chunksPerShard: 2 });

  const read = await fromOmeZarr(store);
  await assertReadsBack(read.images[0].data);
});

Deno.test("toOmeZarr - refuses chunksPerShard at 0.4", async () => {
  await assertRejects(
    () =>
      toOmeZarr(new Map(), createMultiscalesAt04(), {
        version: "0.4",
        chunksPerShard: 2,
      }),
    Error,
    "Sharding is only supported for OME-Zarr version 0.5 and later",
  );
});

Deno.test("browser toOmeZarr - chunksPerShard round-trips", async () => {
  const store = new Map<string, Uint8Array>();
  await toOmeZarrBrowser(store, await multiscales(), {
    version: "0.5",
    chunksPerShard: [2, 2],
  });

  const read = await fromOmeZarrBrowser(store);
  await assertReadsBack(read.images[0].data);
});

Deno.test("browser toOmeZarrOzx - shards by default and round-trips", async () => {
  const zipData = await toOmeZarrOzxBrowser(await multiscales());
  const store = ZipFileStore.fromBlob(new Blob([zipData as BlobPart]));

  const meta = JSON.parse(
    new TextDecoder().decode(await store.get("/0/zarr.json")),
  );
  assertEquals(meta.chunk_grid.configuration.chunk_shape, [16, 16]);
  assertEquals(meta.codecs[0].name, "sharding_indexed");

  const read = await fromOmeZarrBrowser(store);
  await assertReadsBack(read.images[0].data);
});

// ---------------------------------------------------------------------------
// Main-thread fallback
// ---------------------------------------------------------------------------

/**
 * An array whose codec is registered in this module graph's registry and
 * never in a codec worker's, so only the main-thread fallback can code it.
 */
async function withMainThreadOnlyCodec(
  body: (array: zarr.Array<"uint8", Map<string, Uint8Array>>) => Promise<void>,
): Promise<void> {
  zarr.registry.set("test-main-thread-only", () =>
    Promise.resolve({
      fromConfig: () => ({
        kind: "bytes_to_bytes" as const,
        encode: (bytes: Uint8Array) => bytes,
        decode: (bytes: Uint8Array) => bytes,
      }),
    }));
  try {
    const store = new Map<string, Uint8Array>();
    const array = await zarr.create(zarr.root(store).resolve("a"), {
      shape: [4, 4],
      chunkShape: [2, 2],
      dtype: "uint8",
      fillValue: 0,
      codecs: [
        { name: "bytes", configuration: { endian: "little" } },
        { name: "test-main-thread-only", configuration: {} },
      ],
    });
    await body(array);
  } finally {
    zarr.registry.delete("test-main-thread-only");
  }
}

Deno.test("zarrGet falls back to zarr.get for a codec only this thread knows", async () => {
  await withMainThreadOnlyCodec(async (array) => {
    const data = new Uint8Array(16).map((_, i) => i);
    await zarr.set(array, null, { data, shape: [4, 4], stride: [4, 1] });

    assertEquals((await zarrGet(array)).data, data);
  });
});

Deno.test("zarrSet falls back to zarr.set for a dtype the workers lack", async () => {
  // fizarrita's worker codecs refuse bool; zarrita's own pipeline writes it.
  const store = new Map<string, Uint8Array>();
  const array = await zarr.create(zarr.root(store).resolve("b"), {
    shape: [4],
    chunkShape: [2],
    dtype: "bool",
    fillValue: false,
  });
  const data = new zarr.BoolArray([true, false, true, true]);
  await zarrSet(array, null, { data, shape: [4], stride: [1] });

  const read = (await zarr.get(array)).data as zarr.BoolArray;
  assertEquals([0, 1, 2, 3].map((i) => read.get(i)), [true, false, true, true]);
});

Deno.test("zarrSet reports both refusals when neither path can write", async () => {
  // A sharded bool array: the workers lack bool, zarr.set() lacks sharding.
  const layout = arrayLayout(
    [4],
    ["x"],
    [2],
    [{ name: "bytes", configuration: {} }],
    2,
  );
  const array = await zarr.create(
    zarr.root(ensureRangeReads(new Map<string, Uint8Array>())).resolve("b"),
    {
      shape: [4],
      chunkShape: layout.chunkShape,
      dtype: "bool",
      fillValue: false,
      codecs: layout.codecs,
    },
  );

  const error = await assertRejects(
    () =>
      zarrSet(array, null, {
        data: new zarr.BoolArray(4),
        shape: [4],
        stride: [1],
      }),
    AggregateError,
  );
  assertEquals(
    error.errors.map((cause: Error) => cause.message),
    [
      "Unsupported: data_type bool in worker codecs",
      "Unsupported: set on sharded arrays",
    ],
  );
});

Deno.test("zarrSet falls back to zarr.set for a codec only this thread knows", async () => {
  await withMainThreadOnlyCodec(async (array) => {
    const data = new Uint8Array(16).map((_, i) => 100 + i);
    await zarrSet(array, null, { data, shape: [4, 4], stride: [4, 1] });

    assertEquals((await zarr.get(array)).data, data);
  });
});

function createMultiscalesAt04(): NgffMultiscales {
  const metadata = createMetadata(
    [createAxis("y", "space"), createAxis("x", "space")],
    [createDataset("0", [1.0, 1.0], [0.0, 0.0])],
    "sharded",
    "0.4",
  );
  return createMultiscales([], metadata);
}
