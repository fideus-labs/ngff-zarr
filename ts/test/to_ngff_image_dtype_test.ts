// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The element type `toNgffImage` stores, and the values it stores.
 *
 * A zarr chunk is copied as raw bytes sized by the array's declared
 * `data_type`, so a buffer written under a type it does not hold is
 * reinterpreted rather than converted, and the shapes stay right either way.
 * Every test here reads the values back.
 */

import { assert, assertEquals } from "@std/assert";
import * as zarr from "zarrita";

import { fromOmeZarr } from "../src/io/from_ngff_zarr.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { toNgffImage } from "../src/io/to_ngff_image.ts";
import { toOmeZarr } from "../src/io/to_ngff_zarr.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";

/** The values stored in `image`, in row-major order. */
async function readBack(
  image: { data: zarr.Array<zarr.DataType, zarr.Readable> },
): Promise<{ values: number[]; constructor: string }> {
  const chunk = await zarr.get(image.data, null);
  return {
    values: Array.from(chunk.data as ArrayLike<number>),
    constructor: (chunk.data as object).constructor.name,
  };
}

Deno.test("toNgffImage preserves each typed array's element type", async () => {
  const cases: [ArrayLike<number>, string][] = [
    [new Uint8Array(24), "uint8"],
    [new Uint8ClampedArray(24), "uint8"],
    [new Int8Array(24), "int8"],
    [new Uint16Array(24), "uint16"],
    [new Int16Array(24), "int16"],
    [new Uint32Array(24), "uint32"],
    [new Int32Array(24), "int32"],
    [new Float32Array(24), "float32"],
    [new Float64Array(24), "float64"],
  ];

  for (const [data, dataType] of cases) {
    const image = await toNgffImage(data, {
      dims: ["y", "x"],
      shape: [4, 6],
    });
    assertEquals(image.data.dtype, dataType, data.constructor.name);
  }
});

Deno.test("toNgffImage round-trips uint8 values", async () => {
  // 255 catches a narrower type, and a byte-wise reinterpretation would
  // populate only the first six of the twenty-four elements.
  const data = new Uint8Array(24);
  for (let i = 0; i < data.length; i++) {
    data[i] = i + 1;
  }
  data[23] = 255;

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const { values, constructor } = await readBack(image);

  assertEquals(image.data.dtype, "uint8");
  assertEquals(constructor, "Uint8Array");
  assertEquals(values, Array.from(data));
});

Deno.test("toNgffImage round-trips uint16 values", async () => {
  // Above 255, so a truncation to uint8 shows up here too.
  const data = new Uint16Array(24);
  for (let i = 0; i < data.length; i++) {
    data[i] = (i + 1) * 1000;
  }

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const { values, constructor } = await readBack(image);

  assertEquals(image.data.dtype, "uint16");
  assertEquals(constructor, "Uint16Array");
  assertEquals(values, Array.from(data));
});

Deno.test("toNgffImage round-trips signed values", async () => {
  const int8 = new Int8Array(24);
  const int16 = new Int16Array(24);
  const int32 = new Int32Array(24);
  for (let i = 0; i < 24; i++) {
    int8[i] = i % 2 === 0 ? i + 1 : -(i + 1);
    int16[i] = -(i + 1) * 1000;
    int32[i] = -(i + 1) * 100000;
  }

  for (const data of [int8, int16, int32]) {
    const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
    const { values, constructor } = await readBack(image);

    assertEquals(constructor, data.constructor.name);
    assertEquals(values, Array.from(data));
  }
});

Deno.test("toNgffImage round-trips 32-bit unsigned values", async () => {
  // Beyond float32's exact integer range, where a widening to float32 is
  // lossy.
  const data = new Uint32Array(24);
  for (let i = 0; i < data.length; i++) {
    data[i] = 16777216 + i + 1;
  }

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const { values, constructor } = await readBack(image);

  assertEquals(image.data.dtype, "uint32");
  assertEquals(constructor, "Uint32Array");
  assertEquals(values, Array.from(data));
});

Deno.test("toNgffImage round-trips float32 values", async () => {
  // Multiples of 0.25 are exact in float32, so the comparison is exact.
  const data = new Float32Array(24);
  for (let i = 0; i < data.length; i++) {
    data[i] = (i + 1) * 0.25;
  }

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const { values, constructor } = await readBack(image);

  assertEquals(image.data.dtype, "float32");
  assertEquals(constructor, "Float32Array");
  assertEquals(values, Array.from(data));
});

Deno.test("toNgffImage keeps float64 precision", async () => {
  // 1/3 is not representable in float32, so a widening is visible in the
  // read-back.
  const data = new Float64Array(24);
  for (let i = 0; i < data.length; i++) {
    data[i] = (i + 1) / 3;
  }

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const { values, constructor } = await readBack(image);

  assertEquals(image.data.dtype, "float64");
  assertEquals(constructor, "Float64Array");
  assertEquals(values, Array.from(data));
});

Deno.test("toNgffImage stores a Uint8ClampedArray as uint8", async () => {
  // What canvas ImageData hands over. It is not a Uint8Array, so it needs
  // its own branch in the element-type chain.
  const data = new Uint8ClampedArray(24);
  for (let i = 0; i < data.length; i++) {
    data[i] = i + 1;
  }

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const { values, constructor } = await readBack(image);

  assertEquals(image.data.dtype, "uint8");
  assertEquals(constructor, "Uint8Array");
  assertEquals(values, Array.from(data));
});

Deno.test("toNgffImage converts plain arrays to float32", async () => {
  const twoDimensional = await toNgffImage([[1, 2, 3], [4, 5, 6]], {
    dims: ["y", "x"],
  });
  assertEquals(twoDimensional.data.dtype, "float32");
  assertEquals((await readBack(twoDimensional)).values, [1, 2, 3, 4, 5, 6]);

  const oneDimensional = await toNgffImage([1, 2, 3, 4], { dims: ["x"] });
  assertEquals(oneDimensional.data.dtype, "float32");
  assertEquals((await readBack(oneDimensional)).values, [1, 2, 3, 4]);

  const threeDimensional = await toNgffImage([[[1, 2], [3, 4]]], {
    dims: ["z", "y", "x"],
  });
  assertEquals(threeDimensional.data.dtype, "float32");
  assertEquals((await readBack(threeDimensional)).values, [1, 2, 3, 4]);
});

Deno.test("toNgffImage compresses for the element type it wrote", async () => {
  // defaultCodecs derives the blosc typesize and shuffle mode from the data
  // type; a 1-byte type takes noshuffle.
  const blosc = async (data: ArrayLike<number>) => {
    const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
    const store = image.data.store as Map<string, Uint8Array>;
    const key = [...store.keys()].find((k) => k.endsWith("zarr.json"));
    assert(key, "no array metadata in store");
    const metadata = JSON.parse(new TextDecoder().decode(store.get(key)!));
    const codec = (metadata.codecs as {
      name: string;
      configuration: Record<string, unknown>;
    }[]).find((entry) => entry.name === "blosc");
    assert(codec, "no blosc codec declared in array metadata");
    return codec.configuration;
  };

  assertEquals(await blosc(new Uint8Array(24)), {
    cname: "zstd",
    clevel: 5,
    shuffle: "noshuffle",
    typesize: 1,
    blocksize: 0,
  });
  const uint16 = await blosc(new Uint16Array(24));
  assertEquals(uint16.shuffle, "shuffle");
  assertEquals(uint16.typesize, 2);
  assertEquals((await blosc(new Float64Array(24))).typesize, 8);
});

Deno.test("uint8 values survive the write to a store", async () => {
  // What the image holds is what lands in the store, so a data type the
  // pipeline reports but does not write is caught here.
  const data = new Uint8Array(4 * 6);
  for (let i = 0; i < data.length; i++) {
    data[i] = i * 10 + 1;
  }

  const image = await toNgffImage(data, { dims: ["y", "x"], shape: [4, 6] });
  const multiscales = await toMultiscales(image, { scaleFactors: [] });
  const store: MemoryStore = new Map<string, Uint8Array>();
  await toOmeZarr(store, multiscales, { version: "0.5" });

  const read = await fromOmeZarr(store);
  const scale0 = read.images[0].data;
  assertEquals(scale0.dtype, "uint8");

  const chunk = await zarr.get(scale0, null);
  assertEquals(Array.from(chunk.data as ArrayLike<number>), Array.from(data));
});
