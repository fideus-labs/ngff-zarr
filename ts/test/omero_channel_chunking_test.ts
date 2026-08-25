// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Regression tests for per-channel statistics when the channel axis is
 * chunked.
 *
 * Each chunk covers only `chunkShape[cIndex]` channels, but the statistics
 * loop used to iterate the *full* channel count and read past the decoded
 * data. The out-of-range reads yielded `undefined`, which the accumulator
 * counted (`Number.isNaN(undefined)` is `false`), and every chunk's channels
 * were merged into positions starting at zero — so channel 0 absorbed the
 * whole array's range and the rest came back null.
 */

import { assertEquals } from "@std/assert";
import * as zarr from "zarrita";

import {
  computeOmeroFromNgffImage,
  terminateOmeroWorkerPool,
} from "../src/utils/compute_omero.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { defaultCodecs } from "../src/utils/codecs.ts";
import { zarrSet } from "../src/utils/worker_pool.ts";

const N_CHANNELS = 3;
const HEIGHT = 8;
const WIDTH = 8;
const PLANE = HEIGHT * WIDTH;

/**
 * Build a (c, y, x) image whose channel `c` holds values in
 * `[(c + 1) * 100, (c + 1) * 100 + 6]` — disjoint per channel, so a merge
 * into the wrong channel is unmistakable.
 */
async function makeChannelImage(cChunk: number): Promise<NgffImage> {
  const shape = [N_CHANNELS, HEIGHT, WIDTH];
  const data = new Uint16Array(N_CHANNELS * PLANE);
  for (let c = 0; c < N_CHANNELS; c++) {
    for (let i = 0; i < PLANE; i++) {
      data[c * PLANE + i] = (c + 1) * 100 + (i % 7);
    }
  }

  const store: MemoryStore = new Map<string, Uint8Array>();
  const arr = await zarr.create(zarr.root(store).resolve("/data"), {
    shape,
    chunk_shape: [cChunk, HEIGHT, WIDTH],
    data_type: "uint16",
    fill_value: 0,
    codecs: defaultCodecs("uint16"),
  });
  await zarrSet(arr, null, {
    data,
    shape,
    stride: [PLANE, WIDTH, 1],
  } as never);

  return new NgffImage({
    data: arr,
    dims: ["c", "y", "x"],
    scale: { y: 1, x: 1 },
    translation: { y: 0, x: 0 },
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

/** The per-channel [min, max] the fixture data implies. */
function expectedWindows(): Array<[number, number]> {
  return Array.from(
    { length: N_CHANNELS },
    (_, c) => [(c + 1) * 100, (c + 1) * 100 + 6] as [number, number],
  );
}

// A chunk per channel, two channels per chunk (so the last chunk is a partial
// edge chunk), and the unchunked baseline.
for (const cChunk of [1, 2, N_CHANNELS]) {
  Deno.test(
    `omero channel statistics with channel chunk size ${cChunk}`,
    async () => {
      const image = await makeChannelImage(cChunk);
      try {
        const omero = await computeOmeroFromNgffImage(image);

        assertEquals(omero.channels.length, N_CHANNELS);
        for (const [c, [min, max]] of expectedWindows().entries()) {
          assertEquals(
            [omero.channels[c].window!.min, omero.channels[c].window!.max],
            [min, max],
            `channel ${c} statistics are wrong for chunk size ${cChunk}`,
          );
        }
      } finally {
        terminateOmeroWorkerPool();
      }
    },
  );
}
