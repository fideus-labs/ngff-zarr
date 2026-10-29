// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The browser reader returns what the Node reader returns, including omero
 * channels and RFC-4 orientations. (This package writes 0.5 and later; both
 * readers parse 0.4 with the same fromZarrAttrsV04.)
 */
import { test } from "vite-plus/test";
import { assertEquals, assertExists } from "./_assert.ts";
import * as zarr from "zarrita";

import { fromOmeZarr as fromBrowser } from "../src/io/from_ngff_zarr-browser.ts";
import { fromOmeZarr as fromNode } from "../src/io/from_ngff_zarr.ts";
import { toOmeZarr } from "../src/io/to_ngff_zarr.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../src/utils/factory.ts";

async function writeStore(version: "0.5") {
  const source = new Map<string, Uint8Array>();
  const data = await zarr.create(zarr.root(source).resolve("d"), {
    shape: [4, 4],
    chunkShape: [2, 2],
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
  const axes = [
    createAxis("y", "space", "micrometer", {
      type: "anatomical",
      value: "anterior-to-posterior",
    } as never),
    createAxis("x", "space", "micrometer", {
      type: "anatomical",
      value: "left-to-right",
    } as never),
  ];
  const metadata = createMetadata(
    axes,
    [createDataset("s0", [1, 1], [0, 0])],
    "image",
    version,
  );
  metadata.omero = {
    channels: [
      {
        color: "FF0000",
        window: { min: 0, max: 255, start: 0, end: 255 },
      },
    ],
  } as never;
  const store = new Map<string, Uint8Array>();
  await toOmeZarr(store, createMultiscales([image], metadata), { version });
  return store;
}

for (const version of ["0.5"] as const) {
  test(`browser reader matches the Node reader at ${version}`, async () => {
    const store = await writeStore(version);
    const node = await fromNode(store);
    const browser = await fromBrowser(store);

    assertExists(node.metadata.omero);
    assertExists(node.images[0].axesOrientations);
    assertEquals(browser.metadata.omero, node.metadata.omero);
    assertEquals(
      browser.images[0].axesOrientations,
      node.images[0].axesOrientations,
    );
  });
}
