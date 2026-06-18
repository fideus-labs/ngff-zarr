// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Reader-integration tests for strict structural validation on the image read
 * path (`fromOmeZarr({ validate: true })` in `../src/io/from_ngff_zarr.ts`),
 * exercising the wiring added this phase rather than the rule functions in
 * isolation:
 *
 * - a valid in-memory v0.4 `(y, x)` two-level image passes both the schema and
 *   structural layers under `{ validate: true }`;
 * - an image whose `scale` vector length disagrees with the axis count -- a v0.4
 *   MUST the JSON Schema cannot express -- is rejected under `{ validate: true }`
 *   but reads silently under the default `{ validate: false }`.
 *
 * `fromOmeZarr` wraps any read failure in a generic `Error` whose message it
 * preserves, so the structural {@link ValidationError} surfaces here as an
 * `Error` carrying the `Spec rule [<rule>] violated: ...` text rather than as a
 * `ValidationError` instance (the rule's type/`location` surface is covered by
 * the direct unit tests in `structural_validation_test.ts`). This suite mirrors
 * `py/test/test_structural_validation_reader.py`.
 */

import { assertExists, assertRejects, assertStringIncludes } from "@std/assert";
import * as zarr from "zarrita";
import { fromOmeZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { fromZarrAttrsV04 } from "../src/utils/from_zarr_attrs.ts";
import { SpecRule } from "../src/utils/structural_validation.ts";

/**
 * Build an in-memory v0.4 store from the given multiscales metadata, creating a
 * small 2D zarr array for each declared dataset path so the reader can open
 * every level.
 */
async function createImageStore(
  multiscalesMetadata: Record<string, unknown>,
): Promise<MemoryStore> {
  const store: MemoryStore = new Map();
  const root = zarr.root(store);
  await zarr.create(root, {
    attributes: { multiscales: [multiscalesMetadata] },
  });
  const datasets = multiscalesMetadata.datasets as Array<{ path: string }>;
  for (const dataset of datasets) {
    await zarr.create(root.resolve(dataset.path), {
      shape: [16, 16],
      data_type: "uint8",
      chunk_shape: [16, 16],
      fill_value: 0,
    });
  }
  return store;
}

/**
 * A valid v0.4 `(y, x)` two-level multiscales: finest level first, each level's
 * `scale` length matching the two axes. Passes every structural rule.
 */
function buildValidImageMetadata(): Record<string, unknown> {
  return {
    version: "0.4",
    name: "image",
    axes: [
      { name: "y", type: "space", unit: "micrometer" },
      { name: "x", type: "space", unit: "micrometer" },
    ],
    datasets: [
      {
        path: "0",
        coordinateTransformations: [{ type: "scale", scale: [1.0, 1.0] }],
      },
      {
        path: "1",
        coordinateTransformations: [{ type: "scale", scale: [2.0, 2.0] }],
      },
    ],
  };
}

Deno.test("fromOmeZarr - valid v0.4 image passes strict validation", async () => {
  const store = await createImageStore(buildValidImageMetadata());
  const multiscales = await fromOmeZarr(store, { validate: true });
  assertExists(multiscales);
});

Deno.test(
  "fromOmeZarr - scale-length violation throws only when validating",
  async () => {
    // Corrupt the first level's scale to length 3 against the 2-axis (y, x)
    // image. This stays schema-valid (the scale still has >= 2 entries) but
    // violates the structural scale-length MUST.
    const metadata = buildValidImageMetadata();
    (metadata.datasets as Array<Record<string, unknown>>)[0]
      .coordinateTransformations = [{ type: "scale", scale: [1.0, 1.0, 1.0] }];
    const store = await createImageStore(metadata);

    // validate: true surfaces the structural violation. fromOmeZarr re-wraps the
    // ValidationError in a generic Error, preserving its message text.
    const error = await assertRejects(
      () => fromOmeZarr(store, { validate: true }),
      Error,
      SpecRule.ScaleLengthMismatch,
    );
    assertStringIncludes(error.message, "Spec rule [");

    // The same corrupted store reads cleanly when validation is disabled.
    const multiscales = await fromOmeZarr(store, { validate: false });
    assertExists(multiscales);
  },
);

Deno.test(
  "fromZarrAttrsV04 - structural validation gates a multi-digit minor version",
  async () => {
    // A future "0.10" minor is newer than "0.4" and must still be validated.
    // The gate compares version components as integers; a naive parseFloat
    // would read "0.10" as 0.1 (< 0.4) and silently skip the structural pass.
    // Exercised directly on fromZarrAttrsV04 because fromOmeZarr's version
    // detection does not (yet) admit a "0.10" store.
    const metadata = buildValidImageMetadata();
    metadata.version = "0.10";
    (metadata.datasets as Array<Record<string, unknown>>)[0]
      .coordinateTransformations = [{ type: "scale", scale: [1.0, 1.0, 1.0] }];
    const store = await createImageStore(metadata);

    const error = await assertRejects(
      () => fromZarrAttrsV04({ multiscales: [metadata] }, store, true),
      Error,
      SpecRule.ScaleLengthMismatch,
    );
    assertStringIncludes(error.message, "Spec rule [");

    // A pre-0.4 minor is still skipped: structural rules predate that model.
    metadata.version = "0.3";
    const v03Store = await createImageStore(metadata);
    const result = await fromZarrAttrsV04(
      { multiscales: [metadata] },
      v03Store,
      true,
    );
    assertExists(result.metadata);
  },
);
