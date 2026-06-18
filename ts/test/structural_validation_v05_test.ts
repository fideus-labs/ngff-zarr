// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for the OME-Zarr v0.5 structural-validation rules in
 * `../src/utils/structural_validation.ts`.
 *
 * The two v0.5 namespacing rules -- {@link validateZarrFormatForVersion}
 * (`zarr-format`) and {@link validateOmeNamespace} (`ome-namespace`) -- fire
 * only for v0.5 metadata and are inert for v0.4. They inspect the leaked-key
 * `extra` passthrough the reader fills when a malformed or double-namespaced
 * document puts a group-level key inside a multiscale entry. Each rule is
 * exercised directly (inert-on-v0.4, accepts-clean-v0.5, and
 * rejects-malformed-v0.5) and once end-to-end through the
 * {@link validateStructural} orchestrator.
 *
 * This suite mirrors `py/test/test_structural_validation_v05.py` so the two
 * stay in lockstep.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  SpecRule,
  validateOmeNamespace,
  validateStructural,
  validateZarrFormatForVersion,
  ValidationError,
} from "../src/utils/structural_validation.ts";
import { createScale, type Metadata } from "../src/types/zarr_metadata.ts";

/** A fully valid v0.4 `[c, y, x]` two-level multiscales metadata. */
function validV04Metadata(): Metadata {
  return {
    axes: [
      { name: "c", type: "channel", unit: undefined },
      { name: "y", type: "space", unit: undefined },
      { name: "x", type: "space", unit: undefined },
    ],
    datasets: [
      { path: "0", coordinateTransformations: [createScale([1.0, 1.0, 1.0])] },
      { path: "1", coordinateTransformations: [createScale([1.0, 2.0, 2.0])] },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

/**
 * The valid baseline relabeled to v0.5 with no leaked `extra` keys: the cleanly
 * unwrapped form a v0.5 read produces (`version` hoisted out of `ome.version`).
 */
function validV05Metadata(): Metadata {
  return { ...validV04Metadata(), version: "0.5" };
}

// --- zarr-format -----------------------------------------------------------

Deno.test("zarr-format rule is inert on v0.4 even with a stray key", () => {
  // Gated strictly on the v0.5 version: a v0.4 dataset is Zarr v2, so the rule
  // must not fire even if a zarr_format key is (anomalously) present in extra.
  const metadata = validV04Metadata();
  metadata.extra = { zarr_format: 2 };
  validateZarrFormatForVersion(metadata);
});

Deno.test("zarr-format rule accepts clean v0.5", () => {
  // A clean v0.5 entry carries no zarr_format in extra; nothing to contradict
  // Zarr v3, so the rule is inert.
  validateZarrFormatForVersion(validV05Metadata());
});

Deno.test(
  "zarr-format rule accepts v0.5 with an explicit zarr_format of 3",
  () => {
    const metadata = validV05Metadata();
    metadata.extra = { zarr_format: 3 };
    validateZarrFormatForVersion(metadata);
  },
);

Deno.test("zarr-format rule rejects v0.5 with a non-3 zarr_format", () => {
  // A v0.5 entry that declares a non-3 zarr_format is incompatible with the
  // v0.5 => Zarr v3 requirement and must be rejected.
  const metadata = validV05Metadata();
  metadata.extra = { zarr_format: 2 };
  const error = assertThrows(
    () => validateZarrFormatForVersion(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.ZarrFormat);
  assertEquals(error.location, "multiscales[0]");
});

// --- ome-namespace ---------------------------------------------------------

Deno.test(
  "ome-namespace rule is inert on v0.4 even with a residual wrapper",
  () => {
    // Gated strictly on the v0.5 version: a v0.4 store has no ome wrapper, so
    // the rule must not fire on v0.4 metadata regardless of extra.
    const metadata = validV04Metadata();
    metadata.extra = { ome: { version: "0.5" } };
    validateOmeNamespace(metadata);
  },
);

Deno.test("ome-namespace rule accepts clean v0.5", () => {
  // A cleanly-unwrapped v0.5 entry retains no group-level wrapper key.
  validateOmeNamespace(validV05Metadata());
});

Deno.test("ome-namespace rule rejects a residual ome wrapper", () => {
  // A v0.5 entry that still carries the group-level ome key is
  // double-namespaced / malformed.
  const metadata = validV05Metadata();
  metadata.extra = { ome: { version: "0.5", multiscales: [] } };
  const error = assertThrows(
    () => validateOmeNamespace(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.OmeNamespace);
  assertEquals(error.location, "multiscales[0]");
});

Deno.test("ome-namespace rule rejects a residual multiscales wrapper", () => {
  // The multiscales array is also a group-level key; it must not appear inside
  // a multiscale entry.
  const metadata = validV05Metadata();
  metadata.extra = { multiscales: [] };
  const error = assertThrows(
    () => validateOmeNamespace(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.OmeNamespace);
  assertEquals(error.location, "multiscales[0]");
});

// --- end-to-end through validateStructural ---------------------------------

Deno.test("validateStructural accepts clean v0.5", () => {
  // End-to-end through the orchestrator: a clean v0.5 metadata passes every
  // rule (the new v0.5 namespacing rules included).
  validateStructural(validV05Metadata());
});

Deno.test("validateStructural flags a malformed v0.5 ome namespace", () => {
  const metadata = validV05Metadata();
  metadata.extra = { ome: { version: "0.5" } };
  const error = assertThrows(
    () => validateStructural(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.OmeNamespace);
});

Deno.test("validateStructural flags a malformed v0.5 zarr_format", () => {
  const metadata = validV05Metadata();
  metadata.extra = { zarr_format: 2 };
  const error = assertThrows(
    () => validateStructural(metadata),
    ValidationError,
  );
  assertEquals(error.rule, SpecRule.ZarrFormat);
});
