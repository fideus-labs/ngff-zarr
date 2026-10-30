// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/** toOmeZarr honors overwrite: false and the codecs it is given. */
import { test } from "vite-plus/test";
import { makeTempDir, remove } from "./_fs.ts";
import { assertEquals, assertRejects } from "./_assert.ts";
import { unzipSync } from "fflate";
import * as zarr from "zarrita";

import { fromOmeZarr } from "../src/io/from_ngff_zarr.ts";
import { toOmeZarr, toOmeZarrOzxData } from "../src/io/to_ngff_zarr.ts";
import type { NgffMultiscales } from "../src/types/multiscales.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { bytesOnlyCodecs } from "../src/utils/codecs.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";

async function multiscales() {
  const source = new Map<string, Uint8Array>();
  const data = await zarr.create(zarr.root(source).resolve("d"), {
    shape: [8, 8],
    chunkShape: [4, 4],
    dtype: "uint8",
    fillValue: 0,
  });
  const image = new NgffImage({
    data,
    dims: ["y", "x"],
    scale: { y: 1, x: 1 },
    translation: { y: 0, x: 0 },
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
  const metadata = createMetadata(
    [createAxis("y", "space"), createAxis("x", "space")],
    [createDataset("s0", [1, 1], [0, 0])],
    "image",
  );
  return createMultiscales([image], metadata);
}

test("overwrite: false refuses a store that holds the image", async () => {
  const store = new Map<string, Uint8Array>();
  await toOmeZarr(store, await multiscales());
  await assertRejects(
    async () => toOmeZarr(store, await multiscales(), { overwrite: false }),
    Error,
    "already holds",
  );
});

test(".ozx archives use the codecs they are given", async () => {
  const archive = await toOmeZarrOzxData(await multiscales(), {
    codecs: bytesOnlyCodecs(),
  });
  const array = JSON.parse(
    new TextDecoder().decode(unzipSync(archive)["s0/zarr.json"]),
  );
  const names = JSON.stringify(array.codecs);
  assertEquals(names.includes("blosc"), false, names);
});

test("overwrite: false refuses an existing .ozx file", async () => {
  const dir = await makeTempDir();
  const path = `${dir}/image.ozx`;
  await toOmeZarr(path, await multiscales());
  await assertRejects(
    async () => toOmeZarr(path, await multiscales(), { overwrite: false }),
    Error,
    "already exists",
  );
  await remove(dir, { recursive: true });
});

test("boolean images are written and read back", async () => {
  const source = new Map<string, Uint8Array>();
  const data = await zarr.create(zarr.root(source).resolve("mask"), {
    shape: [2, 2],
    chunkShape: [2, 2],
    dtype: "bool",
    fillValue: false,
  });
  await zarr.set(data, [0, 1], true);
  const image = new NgffImage({
    data,
    dims: ["y", "x"],
    scale: { y: 1, x: 1 },
    translation: { y: 0, x: 0 },
    name: "mask",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
  const metadata = createMetadata(
    [createAxis("y", "space"), createAxis("x", "space")],
    [createDataset("s0", [1, 1], [0, 0])],
    "mask",
  );
  const store = new Map<string, Uint8Array>();
  await toOmeZarr(store, createMultiscales([image], metadata));

  const written = await zarr.open(zarr.root(store).resolve("s0"), {
    kind: "array",
  });
  const read = await zarr.get(written);
  assertEquals(Array.from(read.data as Iterable<boolean>), [
    false,
    true,
    false,
    false,
  ]);
});

test("an .ozx path written by toOmeZarr reads back by path", async () => {
  const dir = await makeTempDir();
  const path = `${dir}/image.ozx`;
  await toOmeZarr(path, await multiscales());
  const read = (await fromOmeZarr(path)) as NgffMultiscales;
  assertEquals(read.images[0].data.shape, [8, 8]);
  await remove(dir, { recursive: true });
});
