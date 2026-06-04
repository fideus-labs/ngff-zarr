// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for the OME-Zarr v0.4 HCS plate/well structural rules in
 * `../src/utils/structural_validation.ts`.
 *
 * Paired pass/fail tests for the two plate/well rules added this phase:
 *
 * - {@link validatePlateWellIndexConsistency} -- a small 2x2 plate
 *   ({@link buildValidPlate}) is mutated in exactly one field so a well's
 *   recorded `rowIndex`/`columnIndex` (or its `path`) disagrees with the
 *   row/column named in its `path`.
 * - {@link validateWellAcquisition} -- a multi-acquisition plate whose well
 *   image omits its `acquisition` reference.
 *
 * Each rule function is exercised directly so the violation is isolated to that
 * rule, asserting the raised {@link ValidationError} carries the expected
 * {@link SpecRule} and a populated `location`. The companion
 * {@link validatePlate}/{@link validateWell} orchestrators are also checked for
 * their strict/`schema_only` behavior.
 *
 * The rule identifiers and location strings are kept byte-for-byte identical to
 * the package's Python implementation; this suite mirrors
 * `py/test/test_structural_validation_hcs.py` so the two stay in lockstep.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  SpecRule,
  validatePlate,
  validatePlateWellIndexConsistency,
  validateWell,
  validateWellAcquisition,
  ValidationError,
  ValidationLevel,
} from "../src/utils/structural_validation.ts";
import type { PlateMetadata, WellMetadata } from "../src/types/hcs.ts";

/**
 * Return a small, fully consistent 2x2 plate.
 *
 * Rows `A` (index 0) and `B` (index 1); columns `1` (index 0) and `2` (index 1);
 * two wells whose `path` and recorded indices agree. This baseline passes the
 * index-consistency rule, so each test mutates exactly one field of a fresh copy
 * to trip it.
 */
function buildValidPlate(): PlateMetadata {
  return {
    columns: [{ name: "1" }, { name: "2" }],
    rows: [{ name: "A" }, { name: "B" }],
    wells: [
      { path: "A/1", rowIndex: 0, columnIndex: 0 },
      { path: "B/2", rowIndex: 1, columnIndex: 1 },
    ],
    version: "0.4",
  };
}

/**
 * Return {@link buildValidPlate} extended with two acquisitions.
 *
 * With more than one acquisition declared, every well image must reference one.
 */
function buildMultiAcquisitionPlate(): PlateMetadata {
  const plate = buildValidPlate();
  plate.acquisitions = [{ id: 0 }, { id: 1 }];
  return plate;
}

/** Build well metadata carrying the given images. */
function buildWell(images: WellMetadata["images"]): WellMetadata {
  return { images, version: "0.4" };
}

/**
 * Assert that `fn` throws a {@link ValidationError} carrying `expectedRule` and
 * exactly `expectedLocation` (a non-empty, populated location).
 */
function assertRuleViolation(
  fn: () => void,
  expectedRule: SpecRule,
  expectedLocation: string,
): void {
  const error = assertThrows(fn, ValidationError);
  assertEquals(error.rule, expectedRule);
  assertEquals(error.location, expectedLocation);
}

// ---------------------------------------------------------------------------
// plate-row-index-consistency
// ---------------------------------------------------------------------------

Deno.test("validatePlateWellIndexConsistency - accepts a consistent plate", () => {
  validatePlateWellIndexConsistency(buildValidPlate());
});

Deno.test(
  "validatePlateWellIndexConsistency - rejects a mismatched or malformed well",
  () => {
    // Each mutation makes the first well's path disagree with its recorded
    // indices (or be malformed); all trip the same rule at plate.wells[0].
    const mutations: Array<(plate: PlateMetadata) => void> = [
      // row-index-mismatch: path "A/1" => row "A" is index 0, but rowIndex
      // claims 1.
      (plate) => {
        plate.wells[0].rowIndex = 1;
      },
      // column-index-mismatch: path "A/1" => column "1" is index 0, but
      // columnIndex claims 1.
      (plate) => {
        plate.wells[0].columnIndex = 1;
      },
      // unknown-row: row "Z" is not declared in plate.rows.
      (plate) => {
        plate.wells[0] = { path: "Z/1", rowIndex: 0, columnIndex: 0 };
      },
      // unknown-column: column "9" is not declared in plate.columns.
      (plate) => {
        plate.wells[0] = { path: "A/9", rowIndex: 0, columnIndex: 0 };
      },
      // malformed-path: a path that is not "<row>/<column>".
      (plate) => {
        plate.wells[0] = { path: "A-1", rowIndex: 0, columnIndex: 0 };
      },
    ];
    for (const mutate of mutations) {
      const plate = buildValidPlate();
      mutate(plate);
      assertRuleViolation(
        () => validatePlateWellIndexConsistency(plate),
        SpecRule.PlateRowIndexConsistency,
        "plate.wells[0]",
      );
    }
  },
);

// ---------------------------------------------------------------------------
// well-acquisition-missing
// ---------------------------------------------------------------------------

Deno.test("validateWellAcquisition - no-op for zero or one acquisition", () => {
  // Zero or one acquisition imposes no requirement, so a missing reference is OK.
  const well = buildWell([{ path: "0" }]);
  validateWellAcquisition(buildValidPlate(), well); // acquisitions undefined
  const single = buildValidPlate();
  single.acquisitions = [{ id: 0 }];
  validateWellAcquisition(single, well);
});

Deno.test(
  "validateWellAcquisition - accepts multi-acquisition with references",
  () => {
    const plate = buildMultiAcquisitionPlate();
    const well = buildWell([
      { path: "0", acquisition: 0 },
      { path: "1", acquisition: 1 },
    ]);
    validateWellAcquisition(plate, well);
  },
);

Deno.test(
  "validateWellAcquisition - rejects a missing acquisition reference",
  () => {
    const plate = buildMultiAcquisitionPlate();
    // The second image omits its acquisition reference; the first is fine.
    const well = buildWell([
      { path: "0", acquisition: 0 },
      { path: "1" },
    ]);
    assertRuleViolation(
      () => validateWellAcquisition(plate, well),
      SpecRule.WellAcquisitionMissing,
      "well.images[1].acquisition",
    );
  },
);

// ---------------------------------------------------------------------------
// Orchestrators: validatePlate / validateWell
// ---------------------------------------------------------------------------

Deno.test("validatePlate - strict rejects; schema_only skips", () => {
  const plate = buildValidPlate();
  plate.wells[0].rowIndex = 1; // trip plate-row-index-consistency
  assertRuleViolation(
    () => validatePlate(plate), // strict by default
    SpecRule.PlateRowIndexConsistency,
    "plate.wells[0]",
  );
  // schema_only runs no structural rule, so the same plate passes.
  validatePlate(plate, { level: ValidationLevel.SchemaOnly });
});

Deno.test("validateWell - strict rejects; schema_only skips", () => {
  const plate = buildMultiAcquisitionPlate();
  const well = buildWell([{ path: "0" }]); // missing acquisition reference
  assertRuleViolation(
    () => validateWell(plate, well), // strict by default
    SpecRule.WellAcquisitionMissing,
    "well.images[0].acquisition",
  );
  // schema_only runs no structural rule, so the same well passes.
  validateWell(plate, well, { level: ValidationLevel.SchemaOnly });
});
