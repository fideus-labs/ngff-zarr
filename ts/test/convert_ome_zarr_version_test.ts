// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assertEquals, assertExists } from "@std/assert";
import { fromNgffZarr, type MemoryStore, toNgffZarr } from "../src/mod.ts";

Deno.test("convert v0.4 to v0.5", async () => {
  // Use cross-platform path handling to access the Python test data
  const testStorePath = new URL(
    "../../py/test/data/input/v04/6001240.zarr",
    import.meta.url,
  );
  const resolvedPath = testStorePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");

  // Load the v0.4 zarr file
  const multiscales = await fromNgffZarr(resolvedPath, {
    validate: true,
    version: "0.4",
  });

  // Verify that we loaded valid multiscales data
  assertExists(multiscales);
  assertExists(multiscales.images);
  assertEquals(Array.isArray(multiscales.images), true);
  assertEquals(multiscales.images.length > 0, true);

  // Verify metadata exists and has the expected structure
  assertExists(multiscales.metadata);
  assertExists(multiscales.metadata.axes);
  assertExists(multiscales.metadata.datasets);

  // Convert to v0.5 format in memory store
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();
  const version = "0.5";
  await toNgffZarr(memoryStore, multiscales, { version });

  // Verify we can read back the v0.5 data and it validates correctly
  const v05Multiscales = await fromNgffZarr(memoryStore, {
    validate: true,
    version: version,
  });

  // Verify the loaded v0.5 data has the same structure
  assertExists(v05Multiscales);
  assertExists(v05Multiscales.images);
  assertEquals(v05Multiscales.images.length, multiscales.images.length);

  console.log(
    `Successfully converted v0.4 to v0.5 with ${v05Multiscales.images.length} images`,
  );
});

Deno.test("convert v0.4 to v0.5 to v0.4", async () => {
  // Use cross-platform path handling to access the Python test data
  const testStorePath = new URL(
    "../../py/test/data/input/v04/6001240.zarr",
    import.meta.url,
  );
  const resolvedPath = testStorePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");

  // Load the original v0.4 zarr file
  const originalMultiscales = await fromNgffZarr(resolvedPath, {
    validate: true,
    version: "0.4",
  });

  // Convert to v0.5 format in memory store
  const v05Store: MemoryStore = new Map<string, Uint8Array>();
  const v05Version = "0.5";
  await toNgffZarr(v05Store, originalMultiscales, { version: v05Version });

  // Read the v0.5 data
  const v05Multiscales = await fromNgffZarr(v05Store, {
    validate: true,
    version: v05Version,
  });

  // Convert back to v0.4 format in a new memory store
  const v04Store: MemoryStore = new Map<string, Uint8Array>();
  const v04Version = "0.4";
  await toNgffZarr(v04Store, v05Multiscales, { version: v04Version });

  // Verify we can read the final v0.4 data and it validates correctly
  const finalMultiscales = await fromNgffZarr(v04Store, {
    validate: true,
    version: v04Version,
  });

  // Verify the final data has the same structure as the original
  assertExists(finalMultiscales);
  assertExists(finalMultiscales.images);
  assertEquals(
    finalMultiscales.images.length,
    originalMultiscales.images.length,
  );

  // Verify basic metadata structure is preserved
  assertEquals(
    finalMultiscales.metadata.axes.length,
    originalMultiscales.metadata.axes.length,
  );
  assertEquals(
    finalMultiscales.metadata.datasets.length,
    originalMultiscales.metadata.datasets.length,
  );

  console.log(
    `Successfully round-trip converted v0.4 → v0.5 → v0.4 with ${finalMultiscales.images.length} images`,
  );
});
