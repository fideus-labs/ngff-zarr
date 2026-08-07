// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertThrows } from "@std/assert";
import {
  AVAILABLE_CODECS,
  bloscShuffleToInt,
  bytesOnlyCodecs,
  codecFromName,
  defaultBloscShuffle,
  defaultCodecs,
  typeSizeForDtype,
} from "../src/utils/codecs.ts";
import type { ZarrCodec } from "../src/utils/codecs.ts";

// ---------------------------------------------------------------------------
// AVAILABLE_CODECS
// ---------------------------------------------------------------------------

Deno.test("AVAILABLE_CODECS includes base codecs", () => {
  const names: string[] = [...AVAILABLE_CODECS];
  for (const expected of ["none", "gzip", "lz4", "zstd"]) {
    assertEquals(
      names.includes(expected),
      true,
      `Missing ${expected} in AVAILABLE_CODECS`,
    );
  }
});

Deno.test("AVAILABLE_CODECS includes blosc variants", () => {
  const names: string[] = [...AVAILABLE_CODECS];
  for (
    const expected of [
      "blosc",
      "blosc:lz4",
      "blosc:zstd",
      "blosc:zlib",
      "blosc:lz4hc",
    ]
  ) {
    assertEquals(
      names.includes(expected),
      true,
      `Missing ${expected} in AVAILABLE_CODECS`,
    );
  }
});

// ---------------------------------------------------------------------------
// codecFromName — "none"
// ---------------------------------------------------------------------------

Deno.test("codecFromName('none') returns bytes-only pipeline", () => {
  const result = codecFromName("none", "uint8");
  assertEquals(result.length, 1);
  assertEquals(result[0].name, "bytes");
});

Deno.test("codecFromName('none') matches bytesOnlyCodecs()", () => {
  const fromName = codecFromName("none", "float32");
  const expected = bytesOnlyCodecs();
  assertEquals(fromName, expected);
});

// ---------------------------------------------------------------------------
// codecFromName — gzip
// ---------------------------------------------------------------------------

Deno.test("codecFromName('gzip') - default level", () => {
  const result = codecFromName("gzip", "float32");
  assertEquals(result.length, 2);
  assertEquals(result[0].name, "bytes");
  assertEquals(result[1].name, "gzip");
  assertEquals(result[1].configuration.level, 6);
});

Deno.test("codecFromName('gzip') - custom level", () => {
  const result = codecFromName("gzip", "uint16", 3);
  assertEquals(result[1].configuration.level, 3);
});

// ---------------------------------------------------------------------------
// codecFromName — zstd
// ---------------------------------------------------------------------------

Deno.test("codecFromName('zstd') - default level", () => {
  const result = codecFromName("zstd", "uint8");
  assertEquals(result.length, 2);
  assertEquals(result[1].name, "zstd");
  assertEquals(result[1].configuration.level, 3);
});

Deno.test("codecFromName('zstd') - custom level", () => {
  const result = codecFromName("zstd", "float64", 10);
  assertEquals(result[1].configuration.level, 10);
});

// ---------------------------------------------------------------------------
// codecFromName — lz4
// ---------------------------------------------------------------------------

Deno.test("codecFromName('lz4') returns bytes + lz4", () => {
  const result = codecFromName("lz4", "int32");
  assertEquals(result.length, 2);
  assertEquals(result[1].name, "lz4");
});

// ---------------------------------------------------------------------------
// codecFromName — blosc
// ---------------------------------------------------------------------------

Deno.test("codecFromName('blosc') defaults to lz4 inner algorithm", () => {
  const result = codecFromName("blosc", "float32");
  assertEquals(result.length, 2);
  assertEquals(result[1].name, "blosc");
  assertEquals(result[1].configuration.cname, "lz4");
  assertEquals(result[1].configuration.clevel, 5);
  assertEquals(
    result[1].configuration.typesize,
    typeSizeForDtype("float32"),
  );
});

Deno.test("codecFromName('blosc:zstd') uses zstd inner algorithm", () => {
  const result = codecFromName("blosc:zstd", "uint16");
  assertEquals(result[1].configuration.cname, "zstd");
  assertEquals(result[1].configuration.clevel, 5);
  assertEquals(result[1].configuration.typesize, 2);
});

Deno.test("codecFromName('blosc:lz4hc') with custom level", () => {
  const result = codecFromName("blosc:lz4hc", "int8", 9);
  assertEquals(result[1].configuration.cname, "lz4hc");
  assertEquals(result[1].configuration.clevel, 9);
});

// ---------------------------------------------------------------------------
// codecFromName — errors
// ---------------------------------------------------------------------------

Deno.test("codecFromName throws on unknown codec", () => {
  assertThrows(
    () => codecFromName("bogus", "uint8"),
    Error,
    'Unknown codec: "bogus"',
  );
});

Deno.test("codecFromName throws on unknown blosc variant", () => {
  assertThrows(
    () => codecFromName("blosc:bogus", "uint8"),
    Error,
    'Unknown codec: "blosc:bogus"',
  );
});

Deno.test("codecFromName throws on empty blosc variant", () => {
  assertThrows(
    () => codecFromName("blosc:", "uint8"),
    Error,
    'Unknown codec: "blosc:"',
  );
});

// ---------------------------------------------------------------------------
// blosc shuffle: v3 string ↔ numcodecs integer
// ---------------------------------------------------------------------------

Deno.test("bloscShuffleToInt maps the v3 string enum to numcodecs ints", () => {
  assertEquals(bloscShuffleToInt("noshuffle"), 0);
  assertEquals(bloscShuffleToInt("shuffle"), 1);
  assertEquals(bloscShuffleToInt("bitshuffle"), 2);
});

Deno.test("bloscShuffleToInt passes integers through unchanged", () => {
  assertEquals(bloscShuffleToInt(0), 0);
  assertEquals(bloscShuffleToInt(1), 1);
  assertEquals(bloscShuffleToInt(2), 2);
  assertEquals(bloscShuffleToInt(-1), -1); // AUTOSHUFFLE
});

Deno.test("bloscShuffleToInt leaves an absent shuffle undefined", () => {
  assertEquals(bloscShuffleToInt(undefined), undefined);
  assertEquals(bloscShuffleToInt(null), undefined);
});

Deno.test(
  "bloscShuffleToInt falls back to noshuffle for out-of-spec values",
  () => {
    // Reading an out-of-spec archive must keep working; blosc frames are
    // self-describing so the declared mode is ignored when decoding.
    assertEquals(bloscShuffleToInt("bogus"), 0);
    assertEquals(bloscShuffleToInt({}), 0);
  },
);

Deno.test("defaultBloscShuffle enables shuffle for multi-byte types", () => {
  for (const dtype of ["int16", "uint16", "float32", "int64", "float64"]) {
    assertEquals(defaultBloscShuffle(dtype), "shuffle", dtype);
  }
});

Deno.test("defaultBloscShuffle disables shuffle for single-byte types", () => {
  // numcodecs hardcodes a blosc typesize of 4, so shuffling a 1-byte type
  // interleaves four unrelated elements and inflates the output.
  for (const dtype of ["int8", "uint8", "bool"]) {
    assertEquals(defaultBloscShuffle(dtype), "noshuffle", dtype);
  }
});

Deno.test("defaultCodecs declares the shuffle mode as a v3 string", () => {
  const shuffle = defaultCodecs("uint16")[1].configuration.shuffle;
  assertEquals(typeof shuffle, "string");
  assertEquals(shuffle, "shuffle");
  assertEquals(defaultCodecs("uint8")[1].configuration.shuffle, "noshuffle");
});

Deno.test("codecFromName declares the shuffle mode as a v3 string", () => {
  const blosc = codecFromName("blosc:zstd", "float32")[1];
  assertEquals(blosc.configuration.shuffle, "shuffle");
  assertEquals(
    codecFromName("blosc:zstd", "uint8")[1].configuration.shuffle,
    "noshuffle",
  );
});

// ---------------------------------------------------------------------------
// consistency: defaultCodecs matches codecFromName('blosc:zstd')
// ---------------------------------------------------------------------------

Deno.test(
  "defaultCodecs and codecFromName('blosc:zstd') produce equivalent blosc config",
  () => {
    const dflt: ZarrCodec[] = defaultCodecs("float32");
    const fromName: ZarrCodec[] = codecFromName("blosc:zstd", "float32");

    assertEquals(dflt[0].name, fromName[0].name); // both "bytes"
    assertEquals(dflt[1].name, fromName[1].name); // both "blosc"
    assertEquals(dflt[1].configuration.cname, fromName[1].configuration.cname);
    assertEquals(
      dflt[1].configuration.clevel,
      fromName[1].configuration.clevel,
    );
    assertEquals(
      dflt[1].configuration.typesize,
      fromName[1].configuration.typesize,
    );
  },
);
