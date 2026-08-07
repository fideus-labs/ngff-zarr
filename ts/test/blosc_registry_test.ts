// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Unit tests for the blosc codec registry patch.
 *
 * `test/blosc_shuffle_test.ts` covers the end-to-end effect on written
 * chunks; these exercise the installer directly against stub registries,
 * including the paths a real write never reaches (missing blosc entry,
 * repeat installation, synchronous codec loaders).
 */

import { assert, assertEquals } from "@std/assert";

import { installBloscShuffleNormalization } from "../src/utils/blosc_registry.ts";
import type { CodecRegistry } from "../src/utils/blosc_registry.ts";

// ---------------------------------------------------------------------------
// Stub registry
// ---------------------------------------------------------------------------

/** Config objects seen by the underlying codec, in call order. */
type Recorder = { configs: Record<string, unknown>[] };

/**
 * Build a registry whose `blosc` entry records the configuration it is
 * handed instead of constructing a real codec.
 *
 * @param sync - Return the codec entry directly rather than as a promise,
 *   the way zarrita registers its bundled (non-npm) codecs.
 */
function stubRegistry(sync = false): {
  registry: CodecRegistry;
  recorder: Recorder;
} {
  const recorder: Recorder = { configs: [] };
  const entry = {
    fromConfig(config: Record<string, unknown>, _meta: unknown) {
      recorder.configs.push(config);
      return { config };
    },
  };

  const registry: CodecRegistry = new Map();
  registry.set("blosc", sync ? () => entry : () => Promise.resolve(entry));
  return { registry, recorder };
}

/** Resolve the registry's blosc entry and run its `fromConfig`. */
async function callFromConfig(
  registry: CodecRegistry,
  config: Record<string, unknown>,
) {
  const entry = await registry.get("blosc")!();
  return entry.fromConfig(config, { data_type: "uint16" });
}

// ---------------------------------------------------------------------------
// Shuffle normalization
// ---------------------------------------------------------------------------

Deno.test("installer converts string shuffle to the numcodecs int", async () => {
  const { registry, recorder } = stubRegistry();
  installBloscShuffleNormalization(registry);

  for (
    const [name, int] of [
      ["noshuffle", 0],
      ["shuffle", 1],
      ["bitshuffle", 2],
    ] as const
  ) {
    await callFromConfig(registry, { cname: "zstd", shuffle: name });
    assertEquals(recorder.configs.at(-1)?.shuffle, int, name);
  }
});

Deno.test("installer leaves an integer shuffle untouched", async () => {
  const { registry, recorder } = stubRegistry();
  installBloscShuffleNormalization(registry);

  await callFromConfig(registry, { cname: "zstd", shuffle: 2 });
  assertEquals(recorder.configs[0].shuffle, 2);
});

Deno.test("installer leaves an absent shuffle absent", async () => {
  // Rewriting `undefined` to 0 would override the numcodecs default of
  // SHUFFLE for configs that never mentioned shuffle at all.
  const { registry, recorder } = stubRegistry();
  installBloscShuffleNormalization(registry);

  await callFromConfig(registry, { cname: "zstd" });
  assertEquals("shuffle" in recorder.configs[0], false);
});

Deno.test("installer preserves the rest of the configuration", async () => {
  const { registry, recorder } = stubRegistry();
  installBloscShuffleNormalization(registry);

  await callFromConfig(registry, {
    cname: "zstd",
    clevel: 7,
    shuffle: "bitshuffle",
    typesize: 2,
    blocksize: 0,
  });

  assertEquals(recorder.configs[0], {
    cname: "zstd",
    clevel: 7,
    shuffle: 2,
    typesize: 2,
    blocksize: 0,
  });
});

Deno.test("installer does not mutate the caller's config object", async () => {
  // The config comes straight from parsed zarr.json metadata, which the
  // caller may reuse for other purposes.
  const { registry } = stubRegistry();
  installBloscShuffleNormalization(registry);

  const config = { cname: "zstd", shuffle: "shuffle" };
  await callFromConfig(registry, config);
  assertEquals(config.shuffle, "shuffle");
});

// ---------------------------------------------------------------------------
// Installation
// ---------------------------------------------------------------------------

Deno.test("installer is idempotent", async () => {
  // worker_pool and the codec worker may both install onto one registry
  // when the module graphs happen to share a zarrita instance.
  const { registry, recorder } = stubRegistry();
  installBloscShuffleNormalization(registry);
  const afterFirst = registry.get("blosc");

  installBloscShuffleNormalization(registry);
  assertEquals(registry.get("blosc"), afterFirst, "entry was re-wrapped");

  await callFromConfig(registry, { shuffle: "shuffle" });
  assertEquals(recorder.configs[0].shuffle, 1);
});

Deno.test("installer is a no-op on a registry without blosc", () => {
  const registry: CodecRegistry = new Map();
  installBloscShuffleNormalization(registry);
  assertEquals(registry.has("blosc"), false);
});

Deno.test("installer preserves the entry's other properties", async () => {
  // The real entry is the numcodecs Blosc class, which carries codecId and
  // the shuffle constants; zarrita's CodecEntry type also allows a `kind`.
  // Narrowing the entry to just fromConfig would drop them.
  const { registry } = stubRegistry();
  const entry = await registry.get("blosc")!();
  Object.assign(entry, {
    codecId: "blosc",
    kind: "bytes_to_bytes",
    SHUFFLE: 1,
  });

  installBloscShuffleNormalization(registry);

  const wrapped = await registry.get("blosc")!();
  assertEquals(wrapped.codecId, "blosc");
  assertEquals(wrapped.kind, "bytes_to_bytes");
  assertEquals(wrapped.SHUFFLE, 1);
});

Deno.test("installer leaves other codecs alone", async () => {
  const { registry } = stubRegistry();
  const gzip = () => ({ fromConfig: () => ({ id: "gzip" }) });
  registry.set("gzip", gzip);

  installBloscShuffleNormalization(registry);

  assertEquals(registry.get("gzip"), gzip);
  assert(await registry.get("blosc")!(), "blosc entry still resolves");
});

Deno.test("installer supports a synchronous codec loader", async () => {
  // zarrita registers its bundled codecs as `() => Codec` rather than a
  // promise-returning dynamic import.
  const { registry, recorder } = stubRegistry(true);
  installBloscShuffleNormalization(registry);

  await callFromConfig(registry, { shuffle: "bitshuffle" });
  assertEquals(recorder.configs[0].shuffle, 2);
});

Deno.test("installer defers loading the codec until first use", () => {
  // The registry entry is a lazy dynamic import; wrapping it must not
  // pull the blosc WASM module into every context that installs it.
  let loaded = 0;
  const registry: CodecRegistry = new Map();
  registry.set("blosc", () => {
    loaded++;
    return Promise.resolve({ fromConfig: () => ({}) });
  });

  installBloscShuffleNormalization(registry);
  assertEquals(loaded, 0);
});
