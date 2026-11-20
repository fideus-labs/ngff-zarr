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

import { test } from "vite-plus/test";
import {
  assertEquals,
  assertExists,
  assertInstanceOf,
  assertRejects,
  assertStringIncludes,
} from "./_assert.ts";
import * as zarr from "zarrita";
import { fromOmeZarr, type MemoryStore } from "../src/io/from_ngff_zarr.ts";
import { ValidationError } from "../src/utils/validate.ts";
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
      dtype: "uint8",
      chunkShape: [16, 16],
      fillValue: 0,
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
      dtype: "uint8",
      chunkShape: [16, 16],
      fillValue: 0,
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

test("fromOmeZarr - valid v0.5 image passes strict validation", async () => {
  // A clean v0.5 store passes strict validation: the reader hoists
  // `ome.version` so the structural pass sees v0.5, and the new namespacing
  // rules accept it (no leaked group-level key, no contradictory zarr_format).
  const store = await createImageStoreV05(buildValidImageMetadataV05());
  const multiscales = await fromOmeZarr(store, { validate: true });
  assertExists(multiscales);
});

test("fromOmeZarr - v0.5 ome-namespace violation throws only when validating", async () => {
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
});

test("fromOmeZarr - v0.5 zarr_format violation throws only when validating", async () => {
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
});

test("fromOmeZarr - valid v0.4 image passes strict validation", async () => {
  const store = await createImageStore(buildValidImageMetadata());
  const multiscales = await fromOmeZarr(store, { validate: true });
  assertExists(multiscales);
});

test("fromOmeZarr - scale-length violation throws only when validating", async () => {
  // Corrupt the first level's scale to length 3 against the 2-axis (y, x)
  // image. This stays schema-valid (the scale still has >= 2 entries) but
  // violates the structural scale-length MUST.
  const metadata = buildValidImageMetadata();
  (
    metadata.datasets as Array<Record<string, unknown>>
  )[0].coordinateTransformations = [{ type: "scale", scale: [1.0, 1.0, 1.0] }];
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
});

test("fromZarrAttrsV04 - structural validation gates a multi-digit minor version", async () => {
  // A future "0.10" minor is newer than "0.4" and must still be validated.
  // The gate compares version components as integers; a naive parseFloat
  // would read "0.10" as 0.1 (< 0.4) and silently skip the structural pass.
  // Exercised directly on fromZarrAttrsV04 because fromOmeZarr's version
  // detection does not (yet) admit a "0.10" store.
  const metadata = buildValidImageMetadata();
  metadata.version = "0.10";
  (
    metadata.datasets as Array<Record<string, unknown>>
  )[0].coordinateTransformations = [{ type: "scale", scale: [1.0, 1.0, 1.0] }];
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
});

/** The root document of an in-memory store, to edit and store again. */
function rootDocument(store: MemoryStore): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(store.get("/zarr.json")!));
}

function setRootDocument(
  store: MemoryStore,
  document: Record<string, unknown>,
): void {
  store.set("/zarr.json", new TextEncoder().encode(JSON.stringify(document)));
}

test("fromOmeZarr - a v0.5 store without a version is a validation error", async () => {
  // From 0.5 `ome.version` MUST be present. Mirrors the Python
  // `test_v05_store_without_a_version_is_refused`.
  const store = await createImageStoreV05(buildValidImageMetadataV05());
  const document = rootDocument(store);
  delete (
    (document.attributes as Record<string, unknown>).ome as Record<
      string,
      unknown
    >
  ).version;
  setRootDocument(store, document);

  const error = await assertRejects(
    () => fromOmeZarr(store, { validate: true }),
    Error,
    "NGFF-GN-001",
  );
  assertInstanceOf(error.cause, ValidationError);
  // Without validation the store reads as 0.5, as in the Python port.
  assertEquals((await fromOmeZarr(store)).images.length, 2);
});

test("fromOmeZarr - a label name that leaves its group is not followed", async () => {
  // The store resolver decodes percent-encoded dot segments, so the raw name
  // is not where it reads. Mirrors the Python
  // `test_a_label_name_that_leaves_its_group_is_not_followed`.
  const store = await createImageStoreV05(buildValidImageMetadataV05());
  const root = zarr.root(store);
  const labels = ["%2e%2e/%2e%2e/private", ".%2e/.%2e/private", "..%2fprivate"];
  await zarr.create(root.resolve("labels"), {
    attributes: { ome: { version: "0.5", labels } },
  });
  await zarr.create(root.resolve("private"), {
    attributes: { ome: { version: "0.5", multiscales: 42 } },
  });

  assertEquals((await fromOmeZarr(store, { validate: true })).images.length, 2);
});

test("fromZarrAttrsV05 - v0.5 namespacing validated when ome.version missing", async () => {
  // A malformed v0.5 store missing `ome.version` must still be validated with
  // the v0.5 namespacing rules. fromZarrAttrsV05 definitionally handles v0.5,
  // so it floors the spec version to "0.5" before the structural pass -- a
  // leaked group-level wrapper key is therefore still caught, rather than
  // silently treated as v0.4 (where the rule is inert).
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
});

test("fromZarrAttrsV05 - v0.5 namespacing validated when ome.version is null", async () => {
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
});

test("fromZarrAttrsV04 - serializer-written @type is not routed to extra", async () => {
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
});

// --- The declared version drives the axis rules on the read path ---

/**
 * Build an in-memory store whose group attributes are the given `ome` block,
 * with one zarr array per declared dataset path.
 *
 * This is the v0.6 / 0.9.dev1 layout: the multiscales sit under `ome`, and the
 * axes under a coordinate system rather than a flat `axes` list.
 */
async function createOmeNamespacedStore(
  ome: Record<string, unknown>,
  shape: number[],
): Promise<MemoryStore> {
  const store: MemoryStore = new Map();
  const root = zarr.root(store);
  await zarr.create(root, { attributes: { ome } });
  const entry = (ome.multiscales as Array<Record<string, unknown>>)[0];
  for (const dataset of entry.datasets as Array<{ path: string }>) {
    await zarr.create(root.resolve(dataset.path), {
      shape,
      dtype: "uint8",
      chunkShape: shape,
      fillValue: 0,
    });
  }
  return store;
}

/** A single-level `ome` block over `axes`, tagged with the given version. */
function omeBlock(
  version: string,
  axes: Array<Record<string, unknown>>,
): Record<string, unknown> {
  const rank = axes.length;
  return {
    version,
    multiscales: [
      {
        name: "image",
        coordinateSystems: [{ name: "intrinsic", axes }],
        datasets: [
          {
            path: "0",
            coordinateTransformations: [
              {
                input: { path: "0" },
                output: { name: "intrinsic" },
                name: "scale0_to_intrinsic",
                type: "sequence",
                transformations: [
                  { type: "scale", scale: Array(rank).fill(1.0) },
                  { type: "translation", translation: Array(rank).fill(0.0) },
                ],
              },
            ],
          },
        ],
      },
    ],
  };
}

const SIX_SPACE_AXES = ["a", "b", "c", "d", "e", "f"].map((name) => ({
  name,
  type: "space",
}));

test("reader - a 0.9.dev1 store with six axes passes strict validation", async () => {
  // The axis rules are inert at 0.9.dev1, so this store is legal. Reading it
  // under `validate: true` only works if the declared version reaches
  // validateStructural: without it every store is held to the v0.4 caps, and
  // a document this port's own writer emits is refused on the way back in.
  const store = await createOmeNamespacedStore(
    omeBlock("0.9.dev1", SIX_SPACE_AXES),
    [2, 2, 2, 2, 2, 2],
  );

  const result = await fromOmeZarr(store, { validate: true });
  assertEquals(result.images[0].dims, ["a", "b", "c", "d", "e", "f"]);
});

test("reader - the same six-axis store tagged 0.6 is refused", async () => {
  // The negative control for the test above: the version is what relaxes the
  // rules, not the read path going soft on every store.
  const store = await createOmeNamespacedStore(
    omeBlock("0.6.dev4", SIX_SPACE_AXES),
    [2, 2, 2, 2, 2, 2],
  );

  // The 0.6 axes schema allows at most five axes.
  const error = await assertRejects(
    () => fromOmeZarr(store, { validate: true }),
    Error,
  );
  assertStringIncludes(error.message, "Array has too many items (6 > 5)");
});

test("reader - a v0.6 array coordinate system needs no space axis", async () => {
  // The v0.6 axes schema is a `oneOf`: 2 or 3 `space` axes, *or* two or more
  // `array` axes. The reader must not hold the second arm to the space-axis
  // floor, or a store this port's own writer accepts at 0.6 cannot be read.
  const store = await createOmeNamespacedStore(
    omeBlock("0.6.dev4", [
      { name: "i", type: "array" },
      { name: "j", type: "array" },
    ]),
    [4, 4],
  );

  const result = await fromOmeZarr(store, { validate: true });
  assertEquals(result.images[0].dims, ["i", "j"]);
});

test("reader - a channel without a color is refused when validating", async () => {
  // The 0.6 image schema does not require `color`; the spec text does.
  const ome = omeBlock("0.6", [
    { name: "y", type: "space" },
    { name: "x", type: "space" },
  ]);
  ome.omero = { channels: [{ window: { min: 0, max: 1, start: 0, end: 1 } }] };
  const store = await createOmeNamespacedStore(ome, [4, 4]);

  await assertRejects(
    () => fromOmeZarr(store, { validate: true }),
    Error,
    SpecRule.OmeroChannelColorFormat,
  );
});

// --- The declared version drives the RFC 4 orientation checks on read ---

/** `(t, y, x)` axes with an orientation on the non-spatial time axis. */
const NON_SPACE_ORIENTED_AXES: Array<Record<string, unknown>> = [
  {
    name: "t",
    type: "time",
    orientation: { type: "anatomical", value: "inferior-to-superior" },
  },
  { name: "y", type: "space" },
  { name: "x", type: "space" },
];

test("reader - a 0.6 store with orientation on a non-space axis reads under validate", async () => {
  // RFC 4 orientation is normative from OME-Zarr 0.9.dev1 only; at 0.6 a
  // declared orientation is read back but not validated.
  const store = await createOmeNamespacedStore(
    omeBlock("0.6.dev4", NON_SPACE_ORIENTED_AXES),
    [2, 2, 2],
  );

  const result = await fromOmeZarr(store, { validate: true });
  assertEquals(result.images[0].dims, ["t", "y", "x"]);
});

test("reader - the same store tagged 0.9.dev1 is refused", async () => {
  // The negative control: at 0.9.dev1 the orientation rules are normative,
  // so the same document is refused with the on-non-space rule.
  const store = await createOmeNamespacedStore(
    omeBlock("0.9.dev1", NON_SPACE_ORIENTED_AXES),
    [2, 2, 2],
  );

  const error = await assertRejects(
    () => fromOmeZarr(store, { validate: true }),
    Error,
  );
  assertStringIncludes(error.message, SpecRule.AxisOrientationOnNonSpace);
});

test("fromZarrAttrsV04 - an out-of-vocabulary orientation reads under validate", async () => {
  // The v0.4 reader hook takes the same gate: an orientation value outside
  // the RFC 4 vocabulary reads cleanly at 0.4.
  const metadata = buildValidImageMetadata();
  (metadata.axes as Array<Record<string, unknown>>)[0].orientation = {
    type: "anatomical",
    value: "up-to-down",
  };
  const store = await createImageStore(metadata);

  const result = await fromZarrAttrsV04(
    { multiscales: [metadata] },
    store,
    true,
  );
  assertExists(result.metadata);
});
