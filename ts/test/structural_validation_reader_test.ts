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

import {
  assertEquals,
  assertExists,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import * as zarr from "zarrita";
import { fromOmeZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import {
  fromZarrAttrsV04,
  fromZarrAttrsV05,
} from "../src/utils/from_zarr_attrs.ts";
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

/**
 * Build an in-memory v0.5 store: the multiscales metadata wrapped under the
 * `ome` namespace (with `version` hoisted to `ome.version`), and a small Zarr
 * v3 array for each declared dataset path.
 */
async function createImageStoreV05(
  multiscalesMetadata: Record<string, unknown>,
): Promise<MemoryStore> {
  const store: MemoryStore = new Map();
  const root = zarr.root(store);
  await zarr.create(root, {
    attributes: {
      ome: { version: "0.5", multiscales: [multiscalesMetadata] },
    },
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
 * A valid v0.5 `(y, x)` two-level multiscale entry: the v0.4 baseline with the
 * per-entry `version` dropped (it lives on `ome.version` at v0.5).
 */
function buildValidImageMetadataV05(): Record<string, unknown> {
  const metadata = buildValidImageMetadata();
  delete metadata.version;
  return metadata;
}

Deno.test(
  "fromOmeZarr - valid v0.5 image passes strict validation",
  async () => {
    // A clean v0.5 store passes strict validation: the reader hoists
    // `ome.version` so the structural pass sees v0.5, and the new namespacing
    // rules accept it (no leaked group-level key, no contradictory zarr_format).
    const store = await createImageStoreV05(buildValidImageMetadataV05());
    const multiscales = await fromOmeZarr(store, { validate: true });
    assertExists(multiscales);
  },
);

Deno.test(
  "fromOmeZarr - v0.5 ome-namespace violation throws only when validating",
  async () => {
    // Leak the group-level `ome` wrapper key into the multiscale entry, where
    // it must never appear -- a double-namespaced / malformed v0.5 document.
    const metadata = buildValidImageMetadataV05();
    metadata.ome = { version: "0.5" };
    const store = await createImageStoreV05(metadata);

    const error = await assertRejects(
      () => fromOmeZarr(store, { validate: true }),
      Error,
      SpecRule.OmeNamespace,
    );
    assertStringIncludes(error.message, "Spec rule [");

    // The same store reads cleanly when validation is disabled.
    const multiscales = await fromOmeZarr(store, { validate: false });
    assertExists(multiscales);
  },
);

Deno.test(
  "fromOmeZarr - v0.5 zarr_format violation throws only when validating",
  async () => {
    // Mis-embed a non-3 `zarr_format` byte inside the multiscale entry; it
    // belongs on the enclosing Zarr v3 group, never on the entry, and a value
    // other than 3 contradicts the v0.5 => Zarr v3 requirement.
    const metadata = buildValidImageMetadataV05();
    metadata.zarr_format = 2;
    const store = await createImageStoreV05(metadata);

    const error = await assertRejects(
      () => fromOmeZarr(store, { validate: true }),
      Error,
      SpecRule.ZarrFormat,
    );
    assertStringIncludes(error.message, "Spec rule [");

    const multiscales = await fromOmeZarr(store, { validate: false });
    assertExists(multiscales);
  },
);

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

Deno.test(
  "fromZarrAttrsV05 - v0.5 namespacing validated when ome.version missing",
  async () => {
    // A malformed v0.5 store missing `ome.version` must still be validated with
    // the v0.5 namespacing rules. fromZarrAttrsV05 definitionally handles v0.5,
    // so it floors the spec version to "0.5" before the structural pass -- a
    // leaked group-level wrapper key is therefore still caught, rather than
    // silently treated as v0.4 (where the rule is inert). Exercised directly on
    // fromZarrAttrsV05 because fromOmeZarr rejects a versionless `ome` store at
    // its version-detection step.
    const metadata = buildValidImageMetadataV05();
    metadata.ome = { version: "0.5" }; // leaked group-level wrapper key
    const store = await createImageStoreV05(metadata);

    // An `ome` wrapper WITHOUT a version, mirroring a malformed v0.5 document.
    const error = await assertRejects(
      () => fromZarrAttrsV05({ ome: { multiscales: [metadata] } }, store, true),
      Error,
      SpecRule.OmeNamespace,
    );
    assertStringIncludes(error.message, "Spec rule [");
  },
);

Deno.test(
  "fromZarrAttrsV05 - v0.5 namespacing validated when ome.version is null",
  async () => {
    // A malformed v0.5 store with an explicit `ome.version: null` must still be
    // validated with the v0.5 namespacing rules: the nullish-coalescing floor
    // treats null like a missing version and defaults it to "0.5", so a leaked
    // group-level wrapper key is still caught rather than silently read as v0.4.
    const metadata = buildValidImageMetadataV05();
    metadata.ome = { version: "0.5" }; // leaked group-level wrapper key
    const store = await createImageStoreV05(metadata);

    const error = await assertRejects(
      () =>
        fromZarrAttrsV05(
          { ome: { version: null, multiscales: [metadata] } },
          store,
          true,
        ),
      Error,
      SpecRule.OmeNamespace,
    );
    assertStringIncludes(error.message, "Spec rule [");
  },
);

Deno.test(
  "fromZarrAttrsV04 - serializer-written @type is not routed to extra",
  async () => {
    // `Metadata.extra` captures only leaked / mis-namespaced keys. The Python
    // writer stamps a JSON-LD `@type` ("ngff:Image") onto every entry, so it is
    // a recognized entry field, not a leak, and must not appear in `extra`.
    const metadata = buildValidImageMetadata();
    metadata["@type"] = "ngff:Image";
    const store = await createImageStore(metadata);

    const result = await fromZarrAttrsV04(
      { multiscales: [metadata] },
      store,
      true,
    );
    assertEquals(result.metadata.extra, {});
  },
);
