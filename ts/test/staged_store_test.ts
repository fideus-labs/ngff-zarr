// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Staging a store: writing and reading an image below a path of a store
 * (`toOmeZarr` and `fromOmeZarr` with `path`), and packing a staged store
 * into an `.ozx` archive with `storeToZip`, in the Node and browser modules.
 * Scenes and transformations with a displacement field, the reason to stage
 * a store, are covered in scene_test.ts and transformation_test.ts.
 */
import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import * as zarr from "zarrita";
import { ZipFileStore } from "@zarrita/storage";

import {
  fromOmeZarr,
  getZipFileList,
  type NgffMultiscales,
  readOzxVersion,
  storeToZip,
  toMultiscales,
  toNgffImage,
  toOmeZarr,
} from "../src/mod.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { fromOmeZarr as fromOmeZarrBrowser } from "../src/io/from_ngff_zarr-browser.ts";
import {
  storeToZip as storeToZipBrowser,
  toOmeZarr as toOmeZarrBrowser,
} from "../src/io/to_ngff_zarr-browser.ts";

const PIXELS = new Uint8Array(12 * 10).map((_, i) => (i * 7) % 256);

async function image(): Promise<NgffMultiscales> {
  return await toMultiscales(
    await toNgffImage(PIXELS, { dims: ["y", "x"], shape: [12, 10] }),
    { scaleFactors: [] },
  );
}

async function pixelsOf(multiscales: NgffMultiscales): Promise<Uint8Array> {
  return (await zarr.get(multiscales.images[0].data)).data as Uint8Array;
}

function documentAt(store: MemoryStore, key: string): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(store.get(key)));
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

const WRITERS = [
  ["Node", toOmeZarr, fromOmeZarr, storeToZip],
  ["browser", toOmeZarrBrowser, fromOmeZarrBrowser, storeToZipBrowser],
] as const;

for (const [module, write, read] of WRITERS) {
  Deno.test(`${module}: an image is written and read below a path of a MemoryStore`, async () => {
    const store: MemoryStore = new Map();
    await write(store, await image(), { version: "0.6", path: "a/b" });

    // The group above is created, empty; there is no root document, since
    // nothing was written at the root.
    assertEquals(documentAt(store, "/a/zarr.json").attributes, {});
    assertEquals(store.has("/zarr.json"), false);
    // The image is consolidated in its own root document.
    const imageRoot = documentAt(store, "/a/b/zarr.json");
    const datasetPath = (await image()).metadata.datasets[0].path;
    assertEquals(
      Object.keys(
        (imageRoot.consolidated_metadata as { metadata: object }).metadata,
      ),
      [datasetPath],
    );

    const readBack = await read(store, { path: "a/b", validate: true });
    assertEquals(await pixelsOf(readBack), PIXELS);
  });

  Deno.test(`${module}: a path that leaves the store is refused`, async () => {
    const multiscales = await image();
    for (const path of ["../b", "/b", "a//b", "a/./b", "%2e%2e/b"]) {
      await assertRejects(
        () => write(new Map(), multiscales, { path }),
        Error,
        "relative path below the store's root",
      );
      await assertRejects(
        () => read(new Map(), { path }),
        Error,
        "relative path below the store's root",
      );
    }
  });
}

Deno.test("Node: an image is written and read below a path of a directory", async () => {
  await withTempDir(async (dir) => {
    const store = `${dir}/staged.ome.zarr`;
    await toOmeZarr(store, await image(), { version: "0.6", path: "a/b" });
    assertEquals(await pixelsOf(await fromOmeZarr(`${store}/a/b`)), PIXELS);
    assertEquals(
      await pixelsOf(await fromOmeZarr(store, { path: "a/b" })),
      PIXELS,
    );
    await assertRejects(
      async () =>
        await toOmeZarr(`${dir}/staged.ozx`, await image(), { path: "a/b" }),
      Error,
      "storeToZip()",
    );
  });
});

for (const [module, write, read, pack] of WRITERS) {
  Deno.test(`${module}: storeToZip packs a MemoryStore at the version its root declares`, async () => {
    for (const version of ["0.5", "0.6"] as const) {
      const store: MemoryStore = new Map();
      await write(store, await image(), { version });
      const zip = pack(store);
      assertEquals(readOzxVersion(zip), version);
      assertEquals(getZipFileList(zip)[0], "zarr.json");
      const archive = ZipFileStore.fromBlob(new Blob([zip as BlobPart]));
      assertEquals(await pixelsOf(await read(archive)), PIXELS);
    }
    // A version given is recorded as given.
    const store: MemoryStore = new Map();
    await write(store, await image(), { version: "0.6" });
    assertEquals(
      readOzxVersion(pack(store, { version: "0.9.dev1" })),
      "0.9.dev1",
    );
  });

  Deno.test(`${module}: storeToZip refuses a store RFC-9 cannot hold`, async () => {
    assertThrows(() => pack(new Map()), Error, "no root zarr.json");
    // RFC-9 holds OME-Zarr 0.5 and later; this writer stores a 0.4 image in
    // a Zarr v3 container, but its metadata still declares 0.4.
    const v04: MemoryStore = new Map();
    await write(v04, await image(), { version: "0.4" });
    assertThrows(() => pack(v04), Error, "requires OME-Zarr version 0.5");
    const v2: MemoryStore = new Map([[
      "/zarr.json",
      new TextEncoder().encode(JSON.stringify({ zarr_format: 2 })),
    ]]);
    assertThrows(() => pack(v2), Error, "zarr_format 2");
  });
}

Deno.test("Node: storeToZip writes a directory or a MemoryStore to an archive", async () => {
  await withTempDir(async (dir) => {
    const store = `${dir}/image.ome.zarr`;
    await toOmeZarr(store, await image(), { version: "0.6" });
    const memory: MemoryStore = new Map();
    await toOmeZarr(memory, await image(), { version: "0.6" });

    for (
      const [source, zipPath] of [[store, `${dir}/a.ozx`], [
        memory,
        `${dir}/b.ozx`,
      ]] as const
    ) {
      await storeToZip(source, zipPath);
      const zip = await Deno.readFile(zipPath);
      assertEquals(readOzxVersion(zip), "0.6");
      const archive = ZipFileStore.fromBlob(new Blob([zip as BlobPart]));
      assertEquals(await pixelsOf(await fromOmeZarr(archive)), PIXELS);
    }
    assertThrows(
      () => storeToZip(store as unknown as MemoryStore),
      Error,
      "storeToZip(directory, zipPath)",
    );
  });
});
