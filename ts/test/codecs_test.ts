// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertThrows } from "@std/assert";
import {
  AVAILABLE_CODECS,
  bytesOnlyCodecs,
  codecFromName,
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
