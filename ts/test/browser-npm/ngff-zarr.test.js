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

  test("toNgffZarrOzx should reject conflicting metadata version", async ({ page }) => {
    const versionResult = await page.evaluate(async () => {
      try {
        const { toNgffZarrOzx, NgffImage, createAxis, createDataset, createMetadata, createMultiscales } = await import(
          "./ngff-zarr.bundle.js"
        );
        const zarr = await import("npm:zarrita@0.1.0-next.19");

        // Create a simple test image
        const store = new Map();
        const root = zarr.root(store);
        const array = await zarr.create(root.resolve("data"), {
          shape: [4, 4],
          chunk_shape: [4, 4],
          data_type: "uint8",
          fill_value: 0,
        });

        const image = new NgffImage({
          data: array,
          dims: ["y", "x"],
          scale: { y: 1.0, x: 1.0 },
          translation: { y: 0.0, x: 0.0 },
          name: "test",
          axesUnits: undefined,
          computedCallbacks: undefined,
        });

        const axes = [createAxis("y", "space"), createAxis("x", "space")];
        const datasets = [createDataset("scale0/image", [1.0, 1.0], [0.0, 0.0])];
        // Create metadata with version "0.4" - this should conflict with RFC-9
        const metadata = createMetadata(axes, datasets, "test", "0.4");
        const multiscales = createMultiscales([image], metadata);

        // Try to create .ozx with conflicting version
        try {
          await toNgffZarrOzx(multiscales);
          return { success: false, error: "Expected error was not thrown" };
        } catch (error) {
          return {
            success: true,
            errorMessage: error.message,
            hasExpectedError: error.message.includes("Incompatible multiscales.metadata.version"),
          };
        }
      } catch (error) {
        return {
          success: false,
          error: error.message,
        };
      }
    });

    expect(versionResult.success).toBeTruthy();
    expect(versionResult.hasExpectedError).toBe(true);
    expect(versionResult.errorMessage).toContain("0.5");
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
