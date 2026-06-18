// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { HCSPlate, PlateMetadata } from "../types/hcs.ts";
import { HCSPlate as HCSPlateClass } from "../types/hcs.ts";
import {
  validatePlate,
  ValidationLevel,
} from "../utils/structural_validation.ts";

export interface FromHcsZarrOptions {
  validate?: boolean;
  wellCacheSize?: number;
  imageCacheSize?: number;
}

export interface ToHcsZarrOptions {
  // Future options for writing HCS data
  compressionLevel?: number;
}

/**
 * Read an HCS plate from an OME-Zarr NGFF store.
 */
export function fromHcsZarr(
  store: string | object,
  options: FromHcsZarrOptions = {},
): HCSPlate {
  const { validate = false, wellCacheSize, imageCacheSize } = options;

  if (typeof store !== "string") {
    throw new Error("Non-string store types not yet supported");
  }

  // For now, create a mock implementation
  // This will be properly implemented when the zarrita API integration is complete
  const plateMetadata: PlateMetadata = {
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

  if (validate) {
    // Strict structural validation of the parsed plate, layered after the
    // schema pass (schema first, then structural -- mirrors the image reader).
    // Per-well image rules run lazily in HCSPlate.getWell, where each well's
    // own metadata is loaded. These plate/well rules are version-agnostic and
    // the HCS reader handles only v0.4/v0.5 plates, so -- unlike the image
    // reader's structural pass -- no >=0.4 version gate is required.
    validatePlate(plateMetadata, { level: ValidationLevel.Strict });
  }

  return new HCSPlateClass({
    store,
    metadata: plateMetadata,
    wellCacheSize,
    imageCacheSize,
    validate,
  });
}

/**
 * Write an HCS plate to an OME-Zarr NGFF store.
 */
export function toHcsZarr(
  plate: HCSPlate,
  store: string | object,
  _options: ToHcsZarrOptions = {},
): void {
  if (typeof store !== "string") {
    throw new Error("Non-string store types not yet supported");
  }

  // Mock implementation for now
  console.log(`HCS plate structure would be created at ${store}`);
  console.log(`Plate: ${plate.name}`);
  console.log(`Wells: ${plate.wells.length}`);
  console.log(`Rows: ${plate.rows.length}, Columns: ${plate.columns.length}`);
  if (plate.acquisitions) {
    console.log(`Acquisitions: ${plate.acquisitions.length}`);
  }
}
