// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Regression tests for the Zarr v3 blosc `shuffle` mode.
 *
 * Zarr v3 declares the shuffle mode as a string, but the *numcodecs*
 * binding zarrita resolves `blosc` to expects the Zarr v2 integer. The
 * string used to slip through numcodecs' range check unnoticed, so
 * ngff-zarr advertised shuffle in `zarr.json` while writing unshuffled
 * chunks. These tests read the blosc frame header back out and assert
 * that what was written matches what was declared.
 */

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import * as zarr from "zarrita";

import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import { fromNgffZarr } from "../src/io/from_ngff_zarr.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import { defaultCodecs, typeSizeForDtype } from "../src/utils/codecs.ts";
import type { ZarrCodec } from "../src/utils/codecs.ts";
import { zarrGet, zarrSet } from "../src/utils/worker_pool.ts";

// ---------------------------------------------------------------------------
// Blosc frame header
// ---------------------------------------------------------------------------

/** Shuffle mode recorded in a blosc frame's header flags. */
type FrameShuffle = "noshuffle" | "shuffle" | "bitshuffle";

/**
 * Read the shuffle mode out of a blosc frame.
 *
 * The 16-byte header carries the flags in byte 2: bit 0 is byte shuffle,
 * bit 2 is bit shuffle.
 */
function frameShuffle(chunk: Uint8Array): FrameShuffle {
  const flags = chunk[2];
  if (flags & 0x4) return "bitshuffle";
  if (flags & 0x1) return "shuffle";
  return "noshuffle";
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SHAPE = [16, 48, 48];

/**
 * Deterministic image-like data: a smooth ramp plus structured detail, so
 * it compresses well but not so trivially that the shuffle mode is
 * irrelevant.
 */
function sampleValues(count: number): number[] {
  const values = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    values[i] = Math.round(
      900 + 400 * Math.sin(i / 419) + 60 * Math.sin(i / 11) + (i % 13),
    );
  }
  return values;
}

function typedArrayFor(dtype: string, values: number[]) {
  switch (dtype) {
    case "uint8":
      return Uint8Array.from(values, (v) => v & 0xff);
    case "uint16":
      return Uint16Array.from(values);
    case "int16":
      return Int16Array.from(values);
    case "float32":
      return Float32Array.from(values, (v) => v * 1.5);
    default:
      throw new Error(`Unhandled dtype in test fixture: ${dtype}`);
  }
}

/** Build a single-scale NgffImage backed by an in-memory zarr array. */
async function makeImage(dtype: string): Promise<NgffImage> {
  const count = SHAPE[0] * SHAPE[1] * SHAPE[2];
  const data = typedArrayFor(dtype, sampleValues(count));

  const store: MemoryStore = new Map<string, Uint8Array>();
  const zarrArray = await zarr.create(zarr.root(store).resolve("/data"), {
    shape: SHAPE,
    chunk_shape: SHAPE,
    data_type: dtype as zarr.DataType,
    fill_value: 0,
    codecs: defaultCodecs(dtype),
  });

  await zarrSet(zarrArray, null, {
    data,
    shape: SHAPE,
    stride: [SHAPE[1] * SHAPE[2], SHAPE[2], 1],
  } as never);

  return new NgffImage({
    data: zarrArray,
    dims: ["z", "y", "x"],
    scale: { z: 1, y: 1, x: 1 },
    translation: { z: 0, y: 0, x: 0 },
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

/** Write a single-scale image and return the resulting store. */
async function writeStore(
  dtype: string,
  codecs?: ZarrCodec[],
): Promise<MemoryStore> {
  const image = await makeImage(dtype);
  const multiscales = await toMultiscales(image, { scaleFactors: [] });
  const store: MemoryStore = new Map<string, Uint8Array>();
  await toNgffZarr(store, multiscales, {
    version: "0.5",
    ...(codecs ? { codecs } : {}),
  });
  return store;
}

/** The blosc configuration declared in the store's array metadata. */
function declaredBloscConfig(store: MemoryStore): Record<string, unknown> {
  const key = [...store.keys()].find(
    (k) => k.includes("scale0") && k.endsWith("zarr.json"),
  );
  assert(key, "no scale0 array metadata in store");
  const meta = JSON.parse(new TextDecoder().decode(store.get(key)!));
  const blosc = (meta.codecs as { name: string; configuration: never }[])
    .find((c) => c.name === "blosc");
  assert(blosc, "no blosc codec declared in array metadata");
  return blosc.configuration;
}

/** Every encoded chunk in the store, in insertion order. */
function chunks(store: MemoryStore): Uint8Array[] {
  return [...store.entries()]
    .filter(([k]) => !k.endsWith("zarr.json"))
    .map(([, v]) => v);
}

function totalChunkBytes(store: MemoryStore): number {
  return chunks(store).reduce((sum, c) => sum + c.byteLength, 0);
}

// ---------------------------------------------------------------------------
// Written chunks honour the declared shuffle mode
// ---------------------------------------------------------------------------

for (const dtype of ["uint8", "uint16", "int16", "float32"]) {
  Deno.test(
    `toNgffZarr writes ${dtype} chunks matching the declared shuffle mode`,
    async () => {
      const store = await writeStore(dtype);
      const declared = declaredBloscConfig(store).shuffle;

      // Spec compliance: Zarr v3 requires the string enum, not the
      // Zarr v2 integer.
      assertEquals(
        typeof declared,
        "string",
        "shuffle must be declared as a Zarr v3 string",
      );

      const written = chunks(store);
      assert(written.length > 0, "no chunks written");
      for (const chunk of written) {
        assertEquals(
          frameShuffle(chunk),
          declared,
          `chunk frame disagrees with declared shuffle for ${dtype}`,
        );
      }
    },
  );
}

Deno.test("default shuffle mode follows the data type width", async () => {
  // Byte shuffle only has meaning for multi-byte elements.
  assertEquals(
    declaredBloscConfig(await writeStore("uint8")).shuffle,
    "noshuffle",
  );
  assertEquals(
    declaredBloscConfig(await writeStore("uint16")).shuffle,
    "shuffle",
  );
});

Deno.test("declared typesize matches the data type", async () => {
  for (const dtype of ["uint8", "uint16", "float32"]) {
    assertEquals(
      declaredBloscConfig(await writeStore(dtype)).typesize,
      typeSizeForDtype(dtype),
      dtype,
    );
  }
});

// ---------------------------------------------------------------------------
// The declared mode actually changes the bytes written
// ---------------------------------------------------------------------------

function bloscCodecs(dtype: string, shuffle: string): ZarrCodec[] {
  return [
    { name: "bytes", configuration: { endian: "little" } },
    {
      name: "blosc",
      configuration: {
        cname: "zstd",
        clevel: 5,
        shuffle,
        typesize: typeSizeForDtype(dtype),
        blocksize: 0,
      },
    },
  ];
}

Deno.test(
  "each string shuffle mode produces distinct chunk bytes",
  async () => {
    // Before the string→integer normalization every mode silently encoded
    // as noshuffle, so all three stores came out byte-identical.
    const sizes = new Map<string, number>();
    for (const shuffle of ["noshuffle", "shuffle", "bitshuffle"]) {
      const store = await writeStore("uint16", bloscCodecs("uint16", shuffle));
      assertEquals(
        frameShuffle(chunks(store)[0]),
        shuffle,
        `frame does not record ${shuffle}`,
      );
      sizes.set(shuffle, totalChunkBytes(store));
    }

    assertNotEquals(sizes.get("shuffle"), sizes.get("noshuffle"));
    assertNotEquals(sizes.get("bitshuffle"), sizes.get("noshuffle"));
  },
);

Deno.test(
  "an explicit integer shuffle still encodes correctly",
  async () => {
    // Zarr v2 style integers reach numcodecs untouched; accepting them
    // keeps user-supplied codec pipelines working.
    const codecs = bloscCodecs("uint16", "shuffle");
    codecs[1].configuration.shuffle = 1;
    const store = await writeStore("uint16", codecs);
    assertEquals(frameShuffle(chunks(store)[0]), "shuffle");
  },
);

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

for (const dtype of ["uint8", "uint16", "float32"]) {
  Deno.test(`shuffled ${dtype} chunks round-trip unchanged`, async () => {
    const image = await makeImage(dtype);
    const multiscales = await toMultiscales(image, { scaleFactors: [] });
    const store: MemoryStore = new Map<string, Uint8Array>();
    await toNgffZarr(store, multiscales, { version: "0.5" });

    const expected = await zarrGet(image.data, null);
    const actual = await zarrGet(
      (await fromNgffZarr(store)).images[0].data,
      null,
    );

    assertEquals(
      Array.from(actual.data as unknown as ArrayLike<number>),
      Array.from(expected.data as unknown as ArrayLike<number>),
    );
  });
}
