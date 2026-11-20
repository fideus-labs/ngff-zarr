// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { test } from "vite-plus/test";
import { assert, assertEquals } from "./_assert.ts";

import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { toOmeZarr } from "../src/io/to_ngff_zarr.ts";
import { toOmeZarr as toOmeZarrBrowser } from "../src/io/to_ngff_zarr-browser.ts";
import { toNgffImage } from "../src/io/to_ngff_image.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";

const DIMS = ["c", "y", "x"];

/** The metadata document of every array in `store`. */
function arrayDocuments(store: MemoryStore): Record<string, unknown>[] {
  return [...store.entries()]
    .filter(([key]) => key.endsWith("zarr.json"))
    .map(([, bytes]) => JSON.parse(new TextDecoder().decode(bytes)))
    .filter((document) => document.node_type === "array");
}

for (const [name, write] of [
  ["Node", toOmeZarr],
  ["browser", toOmeZarrBrowser],
] as const) {
  for (const version of ["0.5", "0.6", "0.9.dev1"] as const) {
    test(`${name}: a ${version} array names its dimensions`, async () => {
      // OME-Zarr 0.5 requires `dimension_names` on every multiscale level,
      // matching the axes; the Python port writes it on every Zarr v3 array.
      const image = await toNgffImage(new Uint8Array(2 * 8 * 8), {
        dims: DIMS,
        shape: [2, 8, 8],
      });
      const multiscales = await toMultiscales(image, { scaleFactors: [2] });
      const store: MemoryStore = new Map();
      await write(store, multiscales, { version });

      const documents = arrayDocuments(store);
      assert(documents.length >= 2, "two levels were written");
      for (const document of documents) {
        assertEquals(document.dimension_names, DIMS);
      }
    });
  }
}
