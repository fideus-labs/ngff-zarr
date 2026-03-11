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

test.describe("fromNgffZarr Browser Tests", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the bundle test page (which has access to the bundle)
    await page.goto("/bundle-test");

    // Wait for the page to fully load
    await page.waitForLoadState("networkidle");
  });

  test("should load fromNgffZarr function from browser bundle", async ({ page }) => {
    const loadResult = await page.evaluate(async () => {
      try {
        // Import the browser bundle
        const ngffZarr = await import("./ngff-zarr.bundle.js");
        return {
          success: "fromNgffZarr" in ngffZarr,
          hasFromNgffZarr: "fromNgffZarr" in ngffZarr,
          type: typeof ngffZarr.fromNgffZarr,
          exports: Object.keys(ngffZarr).filter((k) =>
            k.includes("Zarr") || k.includes("Store") || k.includes("Ngff")
          ),
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(loadResult.success).toBeTruthy();
    expect(loadResult.hasFromNgffZarr).toBeTruthy();
    expect(loadResult.type).toBe("function");
  });

  test("should reject local file paths in browser", async ({ page }) => {
    const rejectLocalResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");

        // Try to use a local file path - should be rejected in browser
        try {
          await ngffZarr.fromNgffZarr("/tmp/some_file.zarr", {
            validate: false,
          });
          return {
            rejectedAsExpected: false,
            error: "Local file path was not rejected",
          };
        } catch (error) {
          // Check that the error message indicates browser incompatibility
          const rejectedCorrectly = error.message.includes("browser") ||
            error.message.includes("Local file") ||
            error.message.includes("HTTP") ||
            error.message.includes("MemoryStore");
          return {
            rejectedAsExpected: rejectedCorrectly,
            error: error.message,
          };
        }
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(rejectLocalResult.rejectedAsExpected).toBeTruthy();
  });

  test("should have related types available in bundle", async ({ page }) => {
    const typesResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");

        return {
          hasNgffImage: "NgffImage" in ngffZarr,
          hasMultiscales: "Multiscales" in ngffZarr,
          hasMetadataSchema: "MetadataSchema" in ngffZarr,
          fromNgffZarrParamCount: ngffZarr.fromNgffZarr.length,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(typesResult.hasNgffImage).toBeTruthy();
    expect(typesResult.hasMultiscales).toBeTruthy();
    // Function should accept (store, options?) - 1 required, 1 optional
    expect(typesResult.fromNgffZarrParamCount).toBeLessThanOrEqual(2);
  });
});

test.describe("RFC-9 Browser Tests", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the bundle test page (which has access to the bundle)
    await page.goto("/bundle-test");

    // Wait for the page to fully load
    await page.waitForLoadState("networkidle");
  });

  test("should have RFC-9 functions available in browser bundle", async ({ page }) => {
    const loadResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");
        return {
          success: true,
          hasIsOzxPath: "isOzxPath" in ngffZarr,
          hasToNgffZarrOzx: "toNgffZarrOzx" in ngffZarr,
          hasMemoryStoreToZip: "memoryStoreToZip" in ngffZarr,
          hasOrderFilesForRfc9: "orderFilesForRfc9" in ngffZarr,
          hasReadOzxVersion: "readOzxVersion" in ngffZarr,
          typeOfIsOzxPath: typeof ngffZarr.isOzxPath,
          typeOfToNgffZarrOzx: typeof ngffZarr.toNgffZarrOzx,
          typeOfMemoryStoreToZip: typeof ngffZarr.memoryStoreToZip,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(loadResult.success).toBeTruthy();
    expect(loadResult.hasIsOzxPath).toBeTruthy();
    expect(loadResult.hasToNgffZarrOzx).toBeTruthy();
    expect(loadResult.hasMemoryStoreToZip).toBeTruthy();
    expect(loadResult.hasOrderFilesForRfc9).toBeTruthy();
    expect(loadResult.hasReadOzxVersion).toBeTruthy();
    expect(loadResult.typeOfIsOzxPath).toBe("function");
    expect(loadResult.typeOfToNgffZarrOzx).toBe("function");
    expect(loadResult.typeOfMemoryStoreToZip).toBe("function");
  });

  test("isOzxPath should correctly detect .ozx paths", async ({ page }) => {
    const pathResult = await page.evaluate(async () => {
      try {
        const { isOzxPath } = await import("./ngff-zarr.bundle.js");
        return {
          success: true,
          ozxPath: isOzxPath("test.ozx"),
          ozxPathWithDir: isOzxPath("/path/to/file.ozx"),
          zarrPath: isOzxPath("test.zarr"),
          zipPath: isOzxPath("test.zip"),
          noExtension: isOzxPath("ozx"),
          upperCase: isOzxPath("test.OZX"),
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(pathResult.success).toBeTruthy();
    expect(pathResult.ozxPath).toBe(true);
    expect(pathResult.ozxPathWithDir).toBe(true);
    expect(pathResult.zarrPath).toBe(false);
    expect(pathResult.zipPath).toBe(false);
    expect(pathResult.noExtension).toBe(false);
    expect(pathResult.upperCase).toBe(false); // Case sensitive
  });

  test("memoryStoreToZip should create valid ZIP data", async ({ page }) => {
    const zipResult = await page.evaluate(async () => {
      try {
        const { memoryStoreToZip, readOzxVersion } = await import(
          "./ngff-zarr.bundle.js"
        );

        // Create a simple memory store
        const store = new Map();
        store.set("zarr.json", new TextEncoder().encode('{"zarr_format": 3}'));
        store.set(
          "scale0/zarr.json",
          new TextEncoder().encode('{"zarr_format": 3}'),
        );

        // Convert to ZIP
        const zipData = memoryStoreToZip(store, { version: "0.5" });

        // Verify it's a Uint8Array
        const isUint8Array = zipData instanceof Uint8Array;

        // Verify ZIP magic number (PK\x03\x04)
        const hasZipMagic = zipData[0] === 0x50 && zipData[1] === 0x4b &&
          zipData[2] === 0x03 && zipData[3] === 0x04;

        // Read back the version from the ZIP comment
        const version = readOzxVersion(zipData);

        return {
          success: true,
          isUint8Array,
          hasZipMagic,
          zipSize: zipData.length,
          version,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(zipResult.success).toBeTruthy();
    expect(zipResult.isUint8Array).toBe(true);
    expect(zipResult.hasZipMagic).toBe(true);
    expect(zipResult.zipSize).toBeGreaterThan(0);
    expect(zipResult.version).toBe("0.5");
  });

  test("orderFilesForRfc9 should order files correctly", async ({ page }) => {
    const orderResult = await page.evaluate(async () => {
      try {
        const { orderFilesForRfc9 } = await import("./ngff-zarr.bundle.js");

        // Test with files in wrong order
        const files = [
          "c/0/0",
          "scale0/zarr.json",
          "zarr.json",
          "scale1/zarr.json",
        ];
        const ordered = orderFilesForRfc9(files);

        return {
          success: true,
          ordered,
          firstIsRootZarrJson: ordered[0] === "zarr.json",
          zarrJsonsBeforeData: ordered.indexOf("c/0/0") >
            ordered.indexOf("scale1/zarr.json"),
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(orderResult.success).toBeTruthy();
    expect(orderResult.firstIsRootZarrJson).toBe(true);
    expect(orderResult.zarrJsonsBeforeData).toBe(true);
  });

  // Note: Version validation test is covered in Deno tests (rfc9_test.ts)
  // Browser tests cannot easily create zarr arrays without zarrita import
});

test.describe("OMERO WebWorker Computation Tests", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the bundle test page
    await page.goto("/bundle-test");

    // Wait for the page to fully load
    await page.waitForLoadState("networkidle");
  });

  test("should have OMERO computation functions available", async ({ page }) => {
    const loadResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");
        return {
          success: true,
          hasComputeOmeroFromNgffImage: "computeOmeroFromNgffImage" in ngffZarr,
          hasComputeOmeroFromMultiscales: "computeOmeroFromMultiscales" in
            ngffZarr,
          hasTerminateOmeroWorkerPool: "terminateOmeroWorkerPool" in ngffZarr,
          hasGetDefaultColors: "getDefaultColors" in ngffZarr,
          hasGLASBEY_COLORS: "GLASBEY_COLORS" in ngffZarr,
          typeOfComputeOmero: typeof ngffZarr.computeOmeroFromNgffImage,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(loadResult.success).toBeTruthy();
    expect(loadResult.hasComputeOmeroFromNgffImage).toBeTruthy();
    expect(loadResult.hasComputeOmeroFromMultiscales).toBeTruthy();
    expect(loadResult.hasTerminateOmeroWorkerPool).toBeTruthy();
    expect(loadResult.hasGetDefaultColors).toBeTruthy();
    expect(loadResult.hasGLASBEY_COLORS).toBeTruthy();
    expect(loadResult.typeOfComputeOmero).toBe("function");
  });

  test("getDefaultColors should return correct colors", async ({ page }) => {
    const colorsResult = await page.evaluate(async () => {
      try {
        const { getDefaultColors, GLASBEY_COLORS } = await import(
          "./ngff-zarr.bundle.js"
        );

        return {
          success: true,
          singleChannel: getDefaultColors(1),
          twoChannels: getDefaultColors(2),
          threeChannels: getDefaultColors(3),
          glasbeyLength: GLASBEY_COLORS.length,
          firstGlasbey: GLASBEY_COLORS[0],
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(colorsResult.success).toBeTruthy();
    // Single channel should be white
    expect(colorsResult.singleChannel).toEqual(["FFFFFF"]);
    // Multiple channels use Glasbey colors
    expect(colorsResult.twoChannels.length).toBe(2);
    expect(colorsResult.threeChannels.length).toBe(3);
    expect(colorsResult.glasbeyLength).toBe(256);
    expect(colorsResult.firstGlasbey).toBe("30A2DA");
  });

  test("terminateOmeroWorkerPool should not throw", async ({ page }) => {
    const terminateResult = await page.evaluate(async () => {
      try {
        const { terminateOmeroWorkerPool } = await import(
          "./ngff-zarr.bundle.js"
        );

        // Should not throw even if worker pool was never created
        terminateOmeroWorkerPool();
        terminateOmeroWorkerPool(); // Call twice to ensure idempotent

        return { success: true };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(terminateResult.success).toBeTruthy();
  });

  test("should support Worker API and worker pool lifecycle", async ({ page }) => {
    // This test verifies that the browser environment supports Workers
    // and that the worker pool lifecycle (terminate, re-create) works.
    // Actual OMERO computation is tested in Deno unit tests since it
    // requires zarrita arrays which can't be dynamically imported in
    // browser tests due to CORS/security restrictions.
    const workerResult = await page.evaluate(async () => {
      try {
        const { terminateOmeroWorkerPool } = await import(
          "./ngff-zarr.bundle.js"
        );

        // Browser should have Worker support
        const hasWorkerApi = typeof Worker !== "undefined";

        // Terminate should not throw even if pool was never created
        terminateOmeroWorkerPool();

        // Terminate again — should be idempotent
        terminateOmeroWorkerPool();

        return {
          success: true,
          hasWorkerApi,
          typeOfTerminate: typeof terminateOmeroWorkerPool,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
          stack: error.stack,
        };
      }
    });

    if (!workerResult.success) {
      console.log("Worker test error:", workerResult.error);
      console.log("Stack:", workerResult.stack);
    }
    expect(workerResult.success).toBeTruthy();
    // In browsers, Worker API should be available
    expect(workerResult.hasWorkerApi).toBeTruthy();
    // terminateOmeroWorkerPool should be a function
    expect(workerResult.typeOfTerminate).toBe("function");
  });

  test("should have shared OMERO computation functions for benchmarking", async ({ page }) => {
    const benchmarkFunctionsResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");

        // Check for benchmark-related exports
        const exports = {
          hasCreateAccumulator: "createAccumulator" in ngffZarr,
          hasUpdateAccumulator: "updateAccumulator" in ngffZarr,
          hasFinalizeStatistics: "finalizeStatistics" in ngffZarr,
          hasComputeChannelStatistics: "computeChannelStatistics" in ngffZarr,
          hasExtractChannel: "extractChannel" in ngffZarr,
          hasBuildOmeroFromAccumulators: "buildOmeroFromAccumulators" in
            ngffZarr,
          hasValidateQuantiles: "validateQuantiles" in ngffZarr,
          hasQUANTILE_SAMPLE_SIZE: "QUANTILE_SAMPLE_SIZE" in ngffZarr,
        };

        // Test that the functions work correctly with synthetic data
        const testData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Test createAccumulator
        const acc = ngffZarr.createAccumulator();
        const accValid = acc &&
          typeof acc.min === "number" &&
          typeof acc.max === "number" &&
          Array.isArray(acc.sample);

        // Test updateAccumulator
        ngffZarr.updateAccumulator(acc, testData);
        const updateValid = acc.min === 1 && acc.max === 10 && acc.count === 10;

        // Test finalizeStatistics
        const stats = ngffZarr.finalizeStatistics(acc, [0.1, 0.9]);
        const statsValid = stats.min === 1 && stats.max === 10 &&
          typeof stats.qLow === "number";

        // Test computeChannelStatistics (convenience function)
        const directStats = ngffZarr.computeChannelStatistics(testData, [
          0.1,
          0.9,
        ]);
        const directStatsValid = directStats.min === 1 &&
          directStats.max === 10;

        return {
          success: true,
          exports,
          accValid,
          updateValid,
          statsValid,
          directStatsValid,
          quantileSampleSize: ngffZarr.QUANTILE_SAMPLE_SIZE,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
          stack: error.stack,
        };
      }
    });

    if (!benchmarkFunctionsResult.success) {
      console.log(
        "Benchmark functions test error:",
        benchmarkFunctionsResult.error,
      );
      console.log("Stack:", benchmarkFunctionsResult.stack);
    }

    expect(benchmarkFunctionsResult.success).toBeTruthy();
    // Verify all exports are available
    expect(benchmarkFunctionsResult.exports.hasCreateAccumulator).toBeTruthy();
    expect(benchmarkFunctionsResult.exports.hasUpdateAccumulator).toBeTruthy();
    expect(benchmarkFunctionsResult.exports.hasFinalizeStatistics).toBeTruthy();
    expect(
      benchmarkFunctionsResult.exports.hasComputeChannelStatistics,
    ).toBeTruthy();
    expect(benchmarkFunctionsResult.exports.hasExtractChannel).toBeTruthy();
    expect(
      benchmarkFunctionsResult.exports.hasBuildOmeroFromAccumulators,
    ).toBeTruthy();
    expect(benchmarkFunctionsResult.exports.hasValidateQuantiles).toBeTruthy();
    expect(
      benchmarkFunctionsResult.exports.hasQUANTILE_SAMPLE_SIZE,
    ).toBeTruthy();
    // Verify the functions work correctly
    expect(benchmarkFunctionsResult.accValid).toBeTruthy();
    expect(benchmarkFunctionsResult.updateValid).toBeTruthy();
    expect(benchmarkFunctionsResult.statsValid).toBeTruthy();
    expect(benchmarkFunctionsResult.directStatsValid).toBeTruthy();
    expect(benchmarkFunctionsResult.quantileSampleSize).toBe(10000);
  });
});

test.describe("toNgffZarr Browser Tests", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the bundle test page (which has access to the bundle)
    await page.goto("/bundle-test");

    // Wait for the page to fully load
    await page.waitForLoadState("networkidle");
  });

  test("should load toNgffZarr function from browser bundle", async ({ page }) => {
    const loadResult = await page.evaluate(async () => {
      try {
        // Import the browser bundle
        const ngffZarr = await import("./ngff-zarr.bundle.js");
        return {
          success: "toNgffZarr" in ngffZarr,
          hasToNgffZarr: "toNgffZarr" in ngffZarr,
          type: typeof ngffZarr.toNgffZarr,
          exports: Object.keys(ngffZarr).filter((k) =>
            k.includes("Zarr") || k.includes("Store") || k.includes("Ngff")
          ),
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(loadResult.success).toBeTruthy();
    expect(loadResult.hasToNgffZarr).toBeTruthy();
    expect(loadResult.type).toBe("function");
  });

  test("should reject local file paths in browser for writing", async ({ page }) => {
    const rejectLocalResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");

        // Create a minimal multiscales object for testing
        const mockMultiscales = new ngffZarr.Multiscales({
          images: [],
          metadata: {
            axes: [
              { name: "y", type: "space" },
              { name: "x", type: "space" },
            ],
            datasets: [],
            name: "test",
          },
        });

        // Try to use a local file path - should be rejected in browser
        try {
          await ngffZarr.toNgffZarr("/tmp/output.zarr", mockMultiscales, {});
          return {
            rejectedAsExpected: false,
            error: "Local file path was not rejected",
          };
        } catch (error) {
          // Check that the error message indicates browser incompatibility
          const rejectedCorrectly = error.message.includes("browser") ||
            error.message.includes("Local file") ||
            error.message.includes("MemoryStore");
          return {
            rejectedAsExpected: rejectedCorrectly,
            error: error.message,
          };
        }
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(rejectLocalResult.rejectedAsExpected).toBeTruthy();
  });

  test("should reject HTTP URLs for writing", async ({ page }) => {
    const rejectHttpResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");

        // Create a minimal multiscales object for testing
        const mockMultiscales = new ngffZarr.Multiscales({
          images: [],
          metadata: {
            axes: [
              { name: "y", type: "space" },
              { name: "x", type: "space" },
            ],
            datasets: [],
            name: "test",
          },
        });

        // Try to use HTTP URL - should be rejected (read-only)
        try {
          await ngffZarr.toNgffZarr(
            "https://example.com/output.zarr",
            mockMultiscales,
            {},
          );
          return {
            rejectedAsExpected: false,
            error: "HTTP URL was not rejected",
          };
        } catch (error) {
          // Check that the error message indicates read-only
          const rejectedCorrectly = error.message.includes("read-only") ||
            error.message.includes("HTTP") ||
            error.message.includes("MemoryStore");
          return {
            rejectedAsExpected: rejectedCorrectly,
            error: error.message,
          };
        }
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(rejectHttpResult.rejectedAsExpected).toBeTruthy();
  });

  test("should have related types available for toNgffZarr", async ({ page }) => {
    const typesResult = await page.evaluate(async () => {
      try {
        const ngffZarr = await import("./ngff-zarr.bundle.js");

        return {
          hasToNgffZarr: "toNgffZarr" in ngffZarr,
          hasNgffImage: "NgffImage" in ngffZarr,
          hasMultiscales: "Multiscales" in ngffZarr,
          hasFromNgffZarr: "fromNgffZarr" in ngffZarr,
          toNgffZarrParamCount: ngffZarr.toNgffZarr.length,
        };
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(typesResult.hasToNgffZarr).toBeTruthy();
    expect(typesResult.hasNgffImage).toBeTruthy();
    expect(typesResult.hasMultiscales).toBeTruthy();
    expect(typesResult.hasFromNgffZarr).toBeTruthy();
    // Function should accept (store, multiscales, options?) - 2-3 params
    expect(typesResult.toNgffZarrParamCount).toBeLessThanOrEqual(3);
  });
});
