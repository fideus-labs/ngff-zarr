// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { expect, test } from "@playwright/test";

test.describe("NGFF Zarr Browser Bundle Tests", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the bundle test page
    await page.goto("/bundle-test");

    // Wait for the page to fully load
    await page.waitForLoadState("networkidle");
  });

  test("should load the browser-compatible bundle successfully", async ({ page }) => {
    const loadResult = await page.evaluate(async () => {
      try {
        // Import the browser-compatible bundle
        const ngffZarr = await import("./ngff-zarr.bundle.js");
        return {
          success: true,
          exports: Object.keys(ngffZarr),
          hasSchemas: "AxisSchema" in ngffZarr,
          hasTypes: "SupportedDims" in ngffZarr,
          hasUtils: "isValidDimension" in ngffZarr,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(loadResult.success).toBeTruthy();
    expect(loadResult.exports).toBeDefined();
    expect(loadResult.hasSchemas).toBeTruthy();
    expect(loadResult.hasUtils).toBeTruthy();
  });

  test("should validate zarr metadata using schemas", async ({ page }) => {
    const validationResult = await page.evaluate(async () => {
      try {
        const { MetadataSchema } = await import("./ngff-zarr.bundle.js");

        // Test valid NGFF metadata
        const validMetadata = {
          axes: [
            { name: "y", type: "space", unit: "micrometer" },
            { name: "x", type: "space", unit: "micrometer" },
          ],
          datasets: [
            {
              path: "0",
              coordinateTransformations: [{ type: "scale", scale: [1.0, 1.0] }],
            },
          ],
          name: "test-image",
          version: "0.4",
        };

        const result = MetadataSchema.safeParse(validMetadata);
        return {
          success: result.success,
          data: result.success ? result.data : undefined,
          error: result.success ? undefined : result.error.message,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(validationResult.success).toBeTruthy();
    expect(validationResult.data).toBeDefined();
    expect(validationResult.error).toBeUndefined();
  });

  test("should validate units and dimensions", async ({ page }) => {
    const unitsResult = await page.evaluate(async () => {
      try {
        const { isValidDimension, isValidUnit } = await import(
          "./ngff-zarr.bundle.js"
        );

        return {
          validDimensions: {
            x: isValidDimension("x"),
            y: isValidDimension("y"),
            z: isValidDimension("z"),
            t: isValidDimension("t"),
            c: isValidDimension("c"),
            invalid: isValidDimension("invalid"),
          },
          validUnits: {
            micrometer: isValidUnit("micrometer"),
            millimeter: isValidUnit("millimeter"),
            second: isValidUnit("second"),
            invalid: isValidUnit("invalid"),
          },
        };
      } catch (error) {
        return {
          error: error.message,
        };
      }
    });

    expect(unitsResult.error).toBeUndefined();
    expect(unitsResult.validDimensions?.x).toBeTruthy();
    expect(unitsResult.validDimensions?.y).toBeTruthy();
    expect(unitsResult.validDimensions?.z).toBeTruthy();
    expect(unitsResult.validDimensions?.t).toBeTruthy();
    expect(unitsResult.validDimensions?.c).toBeTruthy();
    expect(unitsResult.validDimensions?.invalid).toBeFalsy();

    expect(unitsResult.validUnits?.micrometer).toBeTruthy();
    expect(unitsResult.validUnits?.millimeter).toBeTruthy();
    expect(unitsResult.validUnits?.second).toBeTruthy();
    expect(unitsResult.validUnits?.invalid).toBeFalsy();
  });

  test("should handle multiscale validation", async ({ page }) => {
    const multiscaleResult = await page.evaluate(async () => {
      try {
        const { MultiscalesOptionsSchema } = await import(
          "./ngff-zarr.bundle.js"
        );

        const validMultiscaleOptions = {
          images: [
            {
              data: {
                shape: [1000, 1000],
                dtype: "float64",
                chunks: [100, 100],
                name: "test-image",
              },
              dims: ["y", "x"],
              scale: { y: 1.0, x: 1.0 },
              translation: { y: 0.0, x: 0.0 },
              name: "test-image",
              axesUnits: { y: "micrometer", x: "micrometer" },
            },
          ],
          metadata: {
            axes: [
              { name: "y", type: "space", unit: "micrometer" },
              { name: "x", type: "space", unit: "micrometer" },
            ],
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
            name: "test-image",
            version: "0.4",
          },
        };

        const result = MultiscalesOptionsSchema.safeParse(
          validMultiscaleOptions,
        );
        return {
          success: result.success,
          data: result.success ? result.data : undefined,
          error: result.success ? undefined : result.error.message,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(multiscaleResult.success).toBeTruthy();
    expect(multiscaleResult.data).toBeDefined();
    expect(multiscaleResult.error).toBeUndefined();
  });

  test("should handle performance benchmarks", async ({ page }) => {
    const performanceResult = await page.evaluate(async () => {
      try {
        const { MetadataSchema } = await import("./ngff-zarr.bundle.js");

        const start = performance.now();

        // Run validation 1000 times
        for (let i = 0; i < 1000; i++) {
          const metadata = {
            zarr_format: 2,
            shape: [100, 100],
            chunks: [10, 10],
            dtype: "float64",
            compressor: null,
            fill_value: 0,
            order: "C",
            filters: null,
          };

          MetadataSchema.safeParse(metadata);
        }

        const end = performance.now();

        return {
          success: true,
          duration: end - start,
          validationsPerSecond: 1000 / ((end - start) / 1000),
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(performanceResult.success).toBeTruthy();
    expect(performanceResult.duration).toBeLessThan(1000); // Should complete in under 1 second
    expect(performanceResult.validationsPerSecond).toBeGreaterThan(1000); // Should be able to validate 1000+ times per second
  });
});
