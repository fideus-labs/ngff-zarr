// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Blosc shuffle normalization for zarrita's codec registry.
 *
 * Zarr v3 metadata declares the blosc `shuffle` mode as a string
 * (`"noshuffle"` / `"shuffle"` / `"bitshuffle"`), but zarrita resolves
 * the `blosc` codec to the *numcodecs* binding, whose `fromConfig()`
 * expects the Zarr v2 integer constant. The string slips through
 * numcodecs' `shuffle < -1 || shuffle > 2` range check — comparing a
 * string to a number is always `false` — and reaches the WASM encoder,
 * which coerces it to 0. The result is metadata that advertises shuffle
 * over chunks that were written unshuffled.
 *
 * {@link installBloscShuffleNormalization} wraps the registry's `blosc`
 * entry so the integer form is handed to numcodecs while the store keeps
 * the spec-compliant string. Decoding is unaffected either way — blosc
 * frames are self-describing and numcodecs ignores the configured mode
 * when decompressing.
 *
 * The patch must be installed in **every context that encodes chunks**.
 * Codec work is offloaded to Web Workers, and each worker has its own
 * module graph — so `../workers/omero_codec_worker.ts` installs it
 * worker-side, and `./worker_pool.ts` installs it on the main thread for
 * the paths that fall back to zarrita's in-process pipeline.
 */

import { bloscShuffleToInt } from "./codecs.ts";

/**
 * Minimal shape of a zarrita codec registry entry.
 *
 * Mirrors zarrita's internal `CodecEntry`. Only `fromConfig` is called
 * here, but entries carry other properties — zarrita's type declares an
 * optional `kind`, and the numcodecs blosc class exposes `codecId` and
 * the shuffle constants — so the wrapper forwards whatever else is
 * there rather than narrowing the entry to one method.
 */
interface CodecEntryLike {
  // deno-lint-ignore no-explicit-any
  fromConfig(config: any, meta: any): any;
  // deno-lint-ignore no-explicit-any
  [key: string]: any;
}

/** A zarrita codec registry: codec name → lazy codec loader. */
export type CodecRegistry = Map<
  string,
  () => Promise<CodecEntryLike> | CodecEntryLike
>;

/** Registries already patched, so repeated installs do not re-wrap. */
const patched = new WeakSet<CodecRegistry>();

/**
 * Wrap a registry's `blosc` entry so string shuffle modes are converted
 * to the integer form numcodecs understands.
 *
 * Idempotent: installing twice on the same registry is a no-op. Safe to
 * call on a registry with no `blosc` entry (does nothing).
 *
 * @param registry - A zarrita codec registry (its exported `registry`).
 */
export function installBloscShuffleNormalization(
  registry: CodecRegistry,
): void {
  if (patched.has(registry)) return;

  const loadBlosc = registry.get("blosc");
  if (!loadBlosc) return;

  patched.add(registry);

  registry.set("blosc", async () => {
    const Codec = await loadBlosc();
    return {
      // Carry over the entry's own enumerable properties (codecId, the
      // shuffle constants, a `kind` if present). Spreading first means the
      // override below always wins, whether or not the entry exposes its
      // own enumerable `fromConfig`.
      ...Codec,
      // deno-lint-ignore no-explicit-any
      fromConfig(config: any, meta: any) {
        if (config && typeof config.shuffle === "string") {
          config = { ...config, shuffle: bloscShuffleToInt(config.shuffle) };
        }
        return Codec.fromConfig(config, meta);
      },
    };
  });
}
