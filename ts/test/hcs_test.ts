// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { test } from "vite-plus/test";
import { assertEquals, assertNotEquals, assertThrows } from "./_assert.ts";
import { fromHcsZarr, toHcsZarr } from "../src/io/hcs.ts";
import { HCSPlate as HCSPlateClass } from "../src/types/hcs.ts";
import type {
  PlateAcquisition,
  PlateColumn,
  PlateRow,
  PlateWell,
  WellImage,
} from "../src/schemas/index.ts";
import type { PlateMetadata, WellMetadata } from "../src/types/hcs.ts";
import {
  validatePlate,
  ValidationLevel,
} from "../src/utils/structural_validation.ts";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

// Test data path - using the Python test data
const TEST_DATA_DIR = resolve("../py/test/data/input");
const HCS_DATA_PATH = resolve(TEST_DATA_DIR, "hcs.ome.zarr");

// fromHcsZarr does not read plate metadata yet, so the plate tests build an
// HCSPlate over the fixture from this metadata; wells load from the store.
const PLATE_METADATA: PlateMetadata = {
  columns: [{ name: "1" }, { name: "2" }],
  rows: [{ name: "A" }, { name: "B" }],
  wells: [
    { path: "A/1", rowIndex: 0, columnIndex: 0 },
    { path: "A/2", rowIndex: 0, columnIndex: 1 },
    { path: "B/1", rowIndex: 1, columnIndex: 0 },
    { path: "B/2", rowIndex: 1, columnIndex: 1 },
  ],
  version: "0.4",
  acquisitions: [{ id: 0, name: "Test Acquisition" }],
  field_count: 2,
  name: "Test Plate",
};

function fixturePlate(validate = false): HCSPlateClass {
  if (validate) {
    validatePlate(PLATE_METADATA, { level: ValidationLevel.Strict });
  }
  return new HCSPlateClass({
    store: HCS_DATA_PATH,
    metadata: PLATE_METADATA,
    validate,
  });
}

test("test_hcs_data_exists", () => {
  assertEquals(existsSync(HCS_DATA_PATH), true, "HCS test data should exist");
  assertEquals(
    existsSync(resolve(HCS_DATA_PATH, ".zattrs")),
    true,
    ".zattrs file should exist",
  );
});

test("test_from_hcs_zarr_basic", () => {
  const plate = fixturePlate();

  assertEquals(
    plate instanceof HCSPlateClass,
    true,
    "Should return HCSPlate instance",
  );
  assertEquals(plate.name, "Test Plate", "Plate should have correct name");

  // Check plate structure
  assertEquals(plate.rows.length, 2, "Should have 2 rows (A, B)");
  assertEquals(plate.columns.length, 2, "Should have 2 columns (1, 2)");
  assertEquals(
    plate.wells.length,
    4,
    "Should have 4 wells (A/1, A/2, B/1, B/2)",
  );

  // Check row names
  const rowNames = plate.rows.map((row) => row.name);
  assertEquals(rowNames, ["A", "B"], "Row names should be A, B");

  // Check column names
  const columnNames = plate.columns.map((col) => col.name);
  assertEquals(columnNames, ["1", "2"], "Column names should be 1, 2");

  // Check wells
  const wellPaths = plate.wells.map((well) => well.path);
  assertEquals(wellPaths.includes("A/1"), true, "Should include well A/1");
  assertEquals(wellPaths.includes("A/2"), true, "Should include well A/2");
  assertEquals(wellPaths.includes("B/1"), true, "Should include well B/1");
  assertEquals(wellPaths.includes("B/2"), true, "Should include well B/2");
});

test("test_from_hcs_zarr_acquisitions", () => {
  const plate = fixturePlate();

  assertNotEquals(
    plate.acquisitions,
    undefined,
    "Acquisitions should not be undefined",
  );
  assertEquals(plate.acquisitions!.length, 1, "Should have 1 acquisition");

  const acq = plate.acquisitions![0];
  assertEquals(acq.id, 0, "Acquisition should have ID 0");
});

test("test_from_hcs_zarr_field_count", () => {
  const plate = fixturePlate();

  assertEquals(plate.fieldCount, 2, "Should have field count of 2");
});

test("test_get_well_by_name", () => {
  const plate = fixturePlate();

  const well = plate.getWell("A", "1");
  assertNotEquals(well, null, "Should find well A/1");
  assertEquals(well!.path, "A/1", "Well should have correct path");
  assertEquals(well!.rowIndex, 0, "Well should have correct row index");
  assertEquals(well!.columnIndex, 0, "Well should have correct column index");

  // Test non-existent well
  const noneWell = plate.getWell("Z", "99");
  assertEquals(noneWell, null, "Should return null for non-existent well");
});

test("test_get_well_by_indices", () => {
  const plate = fixturePlate();

  const well1 = plate.getWellByIndices(0, 0); // A/1
  assertNotEquals(well1, null, "Should find well at indices 0,0");
  assertEquals(well1!.path, "A/1", "Should be well A/1");

  const well2 = plate.getWellByIndices(1, 1); // B/2
  assertNotEquals(well2, null, "Should find well at indices 1,1");
  assertEquals(well2!.path, "B/2", "Should be well B/2");

  // Test out of bounds
  const noneWell = plate.getWellByIndices(99, 99);
  assertEquals(noneWell, null, "Should return null for out of bounds indices");
});

test("test_well_images", () => {
  const plate = fixturePlate();
  const well = plate.getWell("A", "1");

  assertNotEquals(well, null, "Well should not be null");
  assertEquals(well!.images.length, 2, "Should have 2 fields of view");

  // Check image paths
  const imagePaths = well!.images.map((img) => img.path);
  assertEquals(imagePaths.includes("0"), true, "Should include image path '0'");
  assertEquals(imagePaths.includes("1"), true, "Should include image path '1'");

  // Check acquisitions
  for (const img of well!.images) {
    assertEquals(img.acquisition, 0, "Image should have acquisition 0");
  }
});

test("test_get_image_from_well", async () => {
  const plate = fixturePlate();
  const well = plate.getWell("A", "1");

  assertNotEquals(well, null, "Well should not be null");

  // Get first field of view
  const _image = await well!.getImage(0);

  // Test invalid field index
  const noneImage = await well!.getImage(99);
  assertEquals(noneImage, null, "Should return null for invalid field index");
});

test("test_get_image_by_acquisition", async () => {
  const plate = fixturePlate();
  const well = plate.getWell("A", "1");

  assertNotEquals(well, null, "Well should not be null");

  // Get image from acquisition 0
  const _image = await well!.getImageByAcquisition(0, 0);

  // Test non-existent acquisition
  const noneImage = await well!.getImageByAcquisition(99, 0);
  assertEquals(
    noneImage,
    null,
    "Should return null for non-existent acquisition",
  );
});

test("test_plate_metadata_creation", () => {
  const columns: PlateColumn[] = [{ name: "1" }, { name: "2" }];
  const rows: PlateRow[] = [{ name: "A" }, { name: "B" }];
  const wells: PlateWell[] = [
    { path: "A/1", rowIndex: 0, columnIndex: 0 },
    { path: "A/2", rowIndex: 0, columnIndex: 1 },
    { path: "B/1", rowIndex: 1, columnIndex: 0 },
    { path: "B/2", rowIndex: 1, columnIndex: 1 },
  ];
  const acquisitions: PlateAcquisition[] = [
    {
      id: 0,
      name: "Test Acquisition",
    },
  ];

  const plateMetadata: PlateMetadata = {
    columns,
    rows,
    wells,
    acquisitions,
    field_count: 2,
    name: "Test Plate",
    version: "0.4",
  };

  assertEquals(plateMetadata.name, "Test Plate");
  assertEquals(plateMetadata.columns.length, 2);
  assertEquals(plateMetadata.rows.length, 2);
  assertEquals(plateMetadata.wells.length, 4);
  assertEquals(plateMetadata.acquisitions!.length, 1);
});

test("test_well_metadata_creation", () => {
  const images: WellImage[] = [
    { path: "0", acquisition: 0 },
    { path: "1", acquisition: 0 },
  ];

  const wellMetadata: WellMetadata = {
    images,
    version: "0.4",
  };

  assertEquals(wellMetadata.images.length, 2);
  assertEquals(wellMetadata.version, "0.4");
  assertEquals(wellMetadata.images[0].path, "0");
  assertEquals(wellMetadata.images[0].acquisition, 0);
});

test("test_validate_hcs_metadata", () => {
  // The strict plate structural rules (validatePlate) accept the fixture's
  // internally consistent plate metadata.
  const plate = fixturePlate(true);
  assertNotEquals(plate, null);
});

test("test_validate_hcs_well_metadata", () => {
  // Loading a well from a plate read with validate: true runs the strict
  // per-well structural rules (validateWell) where the well's own metadata is
  // first available. The fixture plate declares a single acquisition, so the
  // acquisition rule is a no-op and the well loads without throwing.
  const plate = fixturePlate(true);
  const well = plate.getWell("A", "1");
  assertNotEquals(well, null, "Well A/1 should load under validation");
  assertEquals(well!.images.length, 2, "Well should expose its two fields");
});

test("fromHcsZarr and toHcsZarr refuse instead of faking a plate", () => {
  assertThrows(() => fromHcsZarr(HCS_DATA_PATH), Error, "not supported");
  assertThrows(
    () => toHcsZarr(fixturePlate(), "/tmp/plate.ome.zarr"),
    Error,
    "not supported",
  );
});
