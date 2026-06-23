// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
//
// Big-endian pixel data reads.
//
// OME-Zarr produced by other tools (raw2ometiff, NGFF-Converter, ITK's
// OME-Zarr IO) can store pixel data big-endian. OME-Zarr 0.4 (Zarr v2) encodes
// byte order in the `dtype` string (`>u2`); OME-Zarr 0.5 (Zarr v3) encodes it
// in the `bytes` codec (`endian: "big"`). These tests build stores with
// genuinely big-endian chunk bytes (laid out by hand, independent of the
// writer) and assert that `fromNgffZarr` decodes correct pixel values.
//
// See https://github.com/fideus-labs/ngff-zarr/issues/97
import { assertEquals } from "@std/assert";
import * as zarr from "zarrita";
import { fromNgffZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { zarrGet } from "../src/utils/worker_pool.ts";

const enc = new TextEncoder();

interface DTypeInfo {
  /** Zarr v3 `data_type` name. */
  dataType: string;
  /** Zarr v2 big-endian `dtype` string. */
  v2: string;
  /** Bytes per element. */
  size: number;
  /** Write one big-endian element into a DataView. */
  write: (dv: DataView, offset: number, value: number) => void;
}

const DTYPES: Record<string, DTypeInfo> = {
  uint16: {
    dataType: "uint16",
    v2: ">u2",
    size: 2,
    write: (dv, o, v) => dv.setUint16(o, v, false),
  },
  int16: {
    dataType: "int16",
    v2: ">i2",
    size: 2,
    write: (dv, o, v) => dv.setInt16(o, v, false),
  },
  int32: {
    dataType: "int32",
    v2: ">i4",
    size: 4,
    write: (dv, o, v) => dv.setInt32(o, v, false),
  },
  float32: {
    dataType: "float32",
    v2: ">f4",
    size: 4,
    write: (dv, o, v) => dv.setFloat32(o, v, false),
  },
};

/** Encode `values` as big-endian bytes for the given dtype. */
function bigEndianBytes(values: number[], info: DTypeInfo): Uint8Array {
  const buf = new ArrayBuffer(values.length * info.size);
  const dv = new DataView(buf);
  values.forEach((v, i) => info.write(dv, i * info.size, v));
  return new Uint8Array(buf);
}

const SHAPE = [2, 3];
const CHUNK = [2, 3];
const AXES = [{ name: "y", type: "space" }, { name: "x", type: "space" }];

/** A 2x3 OME-Zarr 0.4 (Zarr v2) store with a big-endian array. */
function buildV04(values: number[], info: DTypeInfo): MemoryStore {
  const s = new Map<string, Uint8Array>();
  s.set("/.zgroup", enc.encode(JSON.stringify({ zarr_format: 2 })));
  s.set(
    "/.zattrs",
    enc.encode(JSON.stringify({
      multiscales: [{
        version: "0.4",
        name: "image",
        axes: AXES,
        datasets: [{
          path: "0",
          coordinateTransformations: [{ type: "scale", scale: [1, 1] }],
        }],
      }],
    })),
  );
  s.set(
    "/0/.zarray",
    enc.encode(JSON.stringify({
      zarr_format: 2,
      shape: SHAPE,
      chunks: CHUNK,
      dtype: info.v2,
      compressor: null,
      fill_value: 0,
      order: "C",
      filters: null,
      dimension_separator: ".",
    })),
  );
  s.set(
    "/0/.zattrs",
    enc.encode(JSON.stringify({ _ARRAY_DIMENSIONS: ["y", "x"] })),
  );
  s.set("/0/0.0", bigEndianBytes(values, info));
  return s;
}

/** A 2x3 OME-Zarr 0.5 (Zarr v3) store with an `endian: "big"` array. */
function buildV05(values: number[], info: DTypeInfo): MemoryStore {
  const s = new Map<string, Uint8Array>();
  s.set(
    "/zarr.json",
    enc.encode(JSON.stringify({
      zarr_format: 3,
      node_type: "group",
      attributes: {
        ome: {
          version: "0.5",
          multiscales: [{
            version: "0.5",
            name: "image",
            axes: AXES,
            datasets: [{
              path: "scale0",
              coordinateTransformations: [{ type: "scale", scale: [1, 1] }],
            }],
          }],
        },
      },
    })),
  );
  s.set(
    "/scale0/zarr.json",
    enc.encode(JSON.stringify({
      zarr_format: 3,
      node_type: "array",
      shape: SHAPE,
      data_type: info.dataType,
      chunk_grid: { name: "regular", configuration: { chunk_shape: CHUNK } },
      chunk_key_encoding: {
        name: "default",
        configuration: { separator: "/" },
      },
      fill_value: 0,
      codecs: [{ name: "bytes", configuration: { endian: "big" } }],
      attributes: {},
      dimension_names: ["y", "x"],
      storage_transformers: [],
    })),
  );
  s.set("/scale0/c/0/0", bigEndianBytes(values, info));
  return s;
}

async function readValues(store: MemoryStore): Promise<number[]> {
  const ms = await fromNgffZarr(store);
  const chunk = await zarr.get(ms.images[0].data);
  return Array.from(chunk.data as ArrayLike<number>);
}

const VALUES = [10, 20, 30, 40, 50, 60];

for (const [name, info] of Object.entries(DTYPES)) {
  Deno.test(`fromNgffZarr reads big-endian OME-Zarr 0.4 (${info.v2})`, async () => {
    assertEquals(await readValues(buildV04(VALUES, info)), VALUES);
  });

  Deno.test(`fromNgffZarr reads big-endian OME-Zarr 0.5 (${name}, endian:big)`, async () => {
    assertEquals(await readValues(buildV05(VALUES, info)), VALUES);
  });
}

Deno.test("big-endian values differ from a naive little-endian misread", async () => {
  // Guards against a regression that silently skips the byte swap: reading the
  // big-endian bytes as little-endian would yield byte-swapped values, not the
  // originals. 10 (0x000A) misread little-endian is 0x0A00 = 2560.
  const info = DTYPES.uint16;
  const misread = [2560, 5120, 7680, 10240, 12800, 15360];
  const v04 = await readValues(buildV04(VALUES, info));
  const v05 = await readValues(buildV05(VALUES, info));
  assertEquals(v04, VALUES);
  assertEquals(v05, VALUES);
  assertNotMisread(v04, misread);
  assertNotMisread(v05, misread);
});

function assertNotMisread(actual: number[], misread: number[]): void {
  assertEquals(
    JSON.stringify(actual) === JSON.stringify(misread),
    false,
    "values look byte-swapped — the big-endian byte order was not honored",
  );
}

Deno.test("repeated reads of a big-endian array via zarrGet stay correct", async () => {
  // The production read path (worker-accelerated `zarrGet`) must not corrupt
  // the backing store across repeated reads. A naive in-place byte swap on the
  // stored buffer would make the second read of an in-memory store wrong.
  const info = DTYPES.uint16;
  const store = buildV05(VALUES, info);
  const ms = await fromNgffZarr(store);
  const arr = ms.images[0].data;

  const r1 = Array.from((await zarrGet(arr)).data as ArrayLike<number>);
  const r2 = Array.from((await zarrGet(arr)).data as ArrayLike<number>);
  const r3 = Array.from((await zarrGet(arr)).data as ArrayLike<number>);
  assertEquals(r1, VALUES);
  assertEquals(r2, VALUES);
  assertEquals(r3, VALUES);

  // The stored chunk bytes must be unchanged after reading.
  assertEquals(
    Array.from(store.get("/scale0/c/0/0")!),
    Array.from(bigEndianBytes(VALUES, info)),
  );
});
