// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Example demonstrating OME-Zarr schema validation usage
 */

import {
  ImageSchemaV01,
  ImageSchemaV05,
  OmeZarrMetadataSchema,
  OmeZarrVersionSchema,
  PlateSchemaV05,
} from "../src/schemas/ome_zarr.ts";

// Example 1: Version validation
export function validateVersion(version: string): boolean {
  const result = OmeZarrVersionSchema.safeParse(version);
  return result.success;
}

// Example 2: Image v0.1 validation
export function validateImageV01(data: unknown): boolean {
  const result = ImageSchemaV01.safeParse(data);
  return result.success;
}

// Example 3: Image v0.5 validation
export function validateImageV05(data: unknown): boolean {
  const result = ImageSchemaV05.safeParse(data);
  return result.success;
}

// Example 4: Plate validation
export function validatePlate(data: unknown): boolean {
  const result = PlateSchemaV05.safeParse(data);
  return result.success;
}

// Example 5: Universal OME-Zarr validation
export function validateOmeZarrMetadata(data: unknown): boolean {
  const result = OmeZarrMetadataSchema.safeParse(data);
  return result.success;
}

// Example usage with sample data
export const sampleImageV05 = {
  ome: {
    multiscales: [
      {
        datasets: [
          {
            path: "0",
            coordinateTransformations: [
              {
                type: "scale",
                scale: [1.0, 1.0, 1.0],
              },
            ],
          },
        ],
        axes: [
          { name: "z", type: "space", unit: "micrometer" },
          { name: "y", type: "space", unit: "micrometer" },
          { name: "x", type: "space", unit: "micrometer" },
        ],
      },
    ],
    version: "0.5",
  },
};

export const samplePlate = {
  ome: {
    plate: {
      columns: [{ name: "1" }, { name: "2" }],
      rows: [{ name: "A" }, { name: "B" }],
      wells: [
        {
          path: "A/1",
          rowIndex: 0,
          columnIndex: 0,
        },
      ],
    },
    version: "0.5",
  },
};

// Validation functions that return detailed results
export function validateWithDetails(data: unknown) {
  const result = OmeZarrMetadataSchema.safeParse(data);

  if (result.success) {
    return {
      valid: true,
      data: result.data,
      errors: null,
    };
  } else {
    return {
      valid: false,
      data: null,
      errors: result.error.issues,
    };
  }
}
