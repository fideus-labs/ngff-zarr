// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/** fromZarrAttrsV04 refuses metadata it cannot model, and only maps real orientations. */
import { test } from "vite-plus/test";
import { assertEquals, assertRejects } from "./_assert.ts";
import * as zarr from "zarrita";

import { fromZarrAttrsV04 } from "../src/utils/from_zarr_attrs.ts";

async function storeWithArray() {
  const store = new Map<string, Uint8Array>();
  await zarr.create(zarr.root(store), { attributes: {} });
  await zarr.create(zarr.root(store).resolve("0"), {
    shape: [2, 2],
    chunkShape: [2, 2],
    dtype: "uint8",
  });
  return store;
}

function attrs(axes: unknown[]) {
  return {
    multiscales: [
      {
        version: "0.4",
        axes,
        datasets: [
          {
            path: "0",
            coordinateTransformations: [{ type: "scale", scale: [1, 1] }],
          },
        ],
      },
    ],
  };
}

test("metadata with no axes is refused", async () => {
  await assertRejects(
    async () => fromZarrAttrsV04(attrs([]), await storeWithArray()),
    Error,
    "no axes",
  );
});

test("an orientation value naming an Object property is ignored", async () => {
  const { images } = await fromZarrAttrsV04(
    attrs([
      {
        name: "y",
        type: "space",
        orientation: { type: "anatomical", value: "constructor" },
      },
      { name: "x", type: "space" },
    ]),
    await storeWithArray(),
  );
  assertEquals(images[0].axesOrientations, undefined);
});
