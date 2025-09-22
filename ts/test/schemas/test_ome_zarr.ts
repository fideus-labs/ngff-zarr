// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertExists } from "@std/assert";
import {
  ImageSchemaV01,
  ImageSchemaV05,
  LabelSchemaV04,
  OmeMetadataSchema,
  OmeZarrMetadataSchema,
  OmeZarrVersionSchema,
  PlateSchemaV05,
  WellSchemaV05,
} from "../../src/schemas/ome_zarr.ts";

Deno.test("OME-Zarr Version Schema", () => {
  const validVersions = ["0.1", "0.2", "0.3", "0.4", "0.5"];

  for (const version of validVersions) {
    const result = OmeZarrVersionSchema.safeParse(version);
    assertEquals(result.success, true, `Version ${version} should be valid`);
  }

  const invalidVersions = ["0.6", "1.0", "invalid", ""];

  for (const version of invalidVersions) {
    const result = OmeZarrVersionSchema.safeParse(version);
    assertEquals(result.success, false, `Version ${version} should be invalid`);
  }
});

Deno.test("Image Schema V0.1 - basic validation", () => {
  const imageV01 = {
    multiscales: [
      {
        datasets: [{ path: "0" }, { path: "1" }],
        version: "0.1",
      },
    ],
  };

  const result = ImageSchemaV01.safeParse(imageV01);
  assertEquals(result.success, true);
});

Deno.test("Image Schema V0.1 - with omero metadata", () => {
  const imageV01WithOmero = {
    multiscales: [
      {
        name: "test-image",
        datasets: [{ path: "0" }, { path: "1" }],
        version: "0.1",
        metadata: {
          method: "gaussian",
          version: "1.0",
        },
      },
    ],
    omero: {
      channels: [
        {
          window: {
            start: 0,
            min: 0,
            end: 255,
            max: 255,
          },
          color: "FF0000",
          active: true,
        },
      ],
    },
  };

  const result = ImageSchemaV01.safeParse(imageV01WithOmero);
  assertEquals(result.success, true);
});

Deno.test("Image Schema V0.5 - basic validation", () => {
  const imageV05 = {
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
            {
              path: "1",
              coordinateTransformations: [
                {
                  type: "scale",
                  scale: [2.0, 2.0, 2.0],
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

  const result = ImageSchemaV05.safeParse(imageV05);
  assertEquals(result.success, true);
});

Deno.test("Image Schema V0.5 - requires proper number of space axes", () => {
  const imageWithOneSpaceAxis = {
    ome: {
      multiscales: [
        {
          datasets: [
            {
              path: "0",
              coordinateTransformations: [
                {
                  type: "scale",
                  scale: [1.0, 1.0],
                },
              ],
            },
          ],
          axes: [
            { name: "x", type: "space", unit: "micrometer" },
            { name: "t", type: "time" },
          ],
        },
      ],
      version: "0.5",
    },
  };

  const result = ImageSchemaV05.safeParse(imageWithOneSpaceAxis);
  assertEquals(result.success, false);
});

Deno.test("Plate Schema V0.5 - basic validation", () => {
  const plateV05 = {
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
          {
            path: "A/2",
            rowIndex: 0,
            columnIndex: 1,
          },
          {
            path: "B/1",
            rowIndex: 1,
            columnIndex: 0,
          },
          {
            path: "B/2",
            rowIndex: 1,
            columnIndex: 1,
          },
        ],
      },
      version: "0.5",
    },
  };

  const result = PlateSchemaV05.safeParse(plateV05);
  assertEquals(result.success, true);
});

Deno.test("Plate Schema V0.5 - with acquisitions", () => {
  const plateWithAcquisitions = {
    ome: {
      plate: {
        name: "Test Plate",
        acquisitions: [
          {
            id: 0,
            name: "Acquisition 1",
            maximumfieldcount: 4,
            starttime: 1234567890,
            endtime: 1234567900,
          },
        ],
        field_count: 2,
        columns: [{ name: "1" }],
        rows: [{ name: "A" }],
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

  const result = PlateSchemaV05.safeParse(plateWithAcquisitions);
  assertEquals(result.success, true);
});

Deno.test("Plate Schema V0.5 - rejects invalid well path format", () => {
  const plateWithInvalidWellPath = {
    ome: {
      plate: {
        columns: [{ name: "1" }],
        rows: [{ name: "A" }],
        wells: [
          {
            path: "invalid-path-format",
            rowIndex: 0,
            columnIndex: 0,
          },
        ],
      },
      version: "0.5",
    },
  };

  const result = PlateSchemaV05.safeParse(plateWithInvalidWellPath);
  assertEquals(result.success, false);
});

Deno.test("Well Schema V0.5 - basic validation", () => {
  const wellV05 = {
    ome: {
      well: {
        images: [
          {
            path: "0",
          },
          {
            path: "1",
            acquisition: 0,
          },
        ],
      },
      version: "0.5",
    },
  };

  const result = WellSchemaV05.safeParse(wellV05);
  assertEquals(result.success, true);
});

Deno.test("Well Schema V0.5 - rejects invalid image path format", () => {
  const wellWithInvalidPath = {
    ome: {
      well: {
        images: [
          {
            path: "invalid/path/format",
          },
        ],
      },
      version: "0.5",
    },
  };

  const result = WellSchemaV05.safeParse(wellWithInvalidPath);
  assertEquals(result.success, false);
});

Deno.test("Label Schema V0.4 - basic validation", () => {
  const labelV04 = {
    "image-label": {
      colors: [
        {
          "label-value": 1,
          rgba: [255, 0, 0, 255],
        },
        {
          "label-value": 2,
          rgba: [0, 255, 0, 255],
        },
      ],
      properties: [
        {
          "label-value": 1,
        },
      ],
      source: {
        image: "../../source",
      },
      version: "0.4",
    },
  };

  const result = LabelSchemaV04.safeParse(labelV04);
  assertEquals(result.success, true);
});

Deno.test("Label Schema V0.4 - validates without optional fields", () => {
  const minimalLabel = {
    "image-label": {},
  };

  const result = LabelSchemaV04.safeParse(minimalLabel);
  assertEquals(result.success, true);
});

Deno.test("Label Schema V0.4 - rejects invalid RGBA values", () => {
  const labelWithInvalidRGBA = {
    "image-label": {
      colors: [
        {
          "label-value": 1,
          rgba: [256, 0, 0, 255], // Invalid: 256 > 255
        },
      ],
    },
  };

  const result = LabelSchemaV04.safeParse(labelWithInvalidRGBA);
  assertEquals(result.success, false);
});

Deno.test("OME Metadata Schema - basic validation", () => {
  const omeMetadata = {
    ome: {
      series: [{ image: "image1" }, { image: "image2" }],
      version: "0.5",
    },
  };

  const result = OmeMetadataSchema.safeParse(omeMetadata);
  assertEquals(result.success, true);
});

Deno.test(
  "Comprehensive OME-Zarr Metadata Schema - validates different types",
  () => {
    const testCases = [
      {
        name: "Image V0.1",
        data: {
          multiscales: [
            {
              datasets: [{ path: "0" }],
            },
          ],
        },
      },
      {
        name: "Image V0.5",
        data: {
          ome: {
            multiscales: [
              {
                datasets: [
                  {
                    path: "0",
                    coordinateTransformations: [
                      {
                        type: "scale",
                        scale: [1.0, 1.0],
                      },
                    ],
                  },
                ],
                axes: [
                  { name: "y", type: "space" },
                  { name: "x", type: "space" },
                ],
              },
            ],
            version: "0.5",
          },
        },
      },
      {
        name: "Plate V0.5",
        data: {
          ome: {
            plate: {
              columns: [{ name: "1" }],
              rows: [{ name: "A" }],
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
        },
      },
      {
        name: "Well V0.5",
        data: {
          ome: {
            well: {
              images: [{ path: "0" }],
            },
            version: "0.5",
          },
        },
      },
      {
        name: "Label V0.4",
        data: {
          "image-label": {
            colors: [
              {
                "label-value": 1,
                rgba: [255, 0, 0, 255],
              },
            ],
          },
        },
      },
      {
        name: "OME Metadata",
        data: {
          ome: {
            series: [{ image: "test" }],
            version: "0.4",
          },
        },
      },
    ];

    for (const testCase of testCases) {
      const result = OmeZarrMetadataSchema.safeParse(testCase.data);
      assertEquals(
        result.success,
        true,
        `${testCase.name} should be valid: ${
          result.success ? "" : result.error?.message
        }`,
      );
    }
  },
);

Deno.test("Error Handling - provides meaningful error messages", () => {
  const invalidImage = {
    ome: {
      multiscales: [
        {
          datasets: [], // Invalid: must have at least 1 dataset
          axes: [
            { name: "y", type: "space" },
            { name: "x", type: "space" },
          ],
        },
      ],
      version: "0.5",
    },
  };

  const result = ImageSchemaV05.safeParse(invalidImage);
  assertEquals(result.success, false);
  assertExists(result.error);
});
