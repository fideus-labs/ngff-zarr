// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { test } from "vite-plus/test";
import { assertEquals, assertExists, assertRejects } from "./_assert.ts";
import { fromNgffZarr, type MemoryStore, toNgffZarr } from "../src/mod.ts";

test("convert v0.4 to v0.5", async () => {
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

test("writing back to v0.4 is refused", async () => {
  // OME-Zarr 0.4 is Zarr v2, which this package does not write.
  const testStorePath = new URL(
    "../../py/test/data/input/v04/6001240.zarr",
    import.meta.url,
  );
  const resolvedPath = testStorePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");
  const original = await fromNgffZarr(resolvedPath, {
    validate: true,
    version: "0.4",
  });

  const v05Store: MemoryStore = new Map<string, Uint8Array>();
  await toNgffZarr(v05Store, original, { version: "0.5" });
  const v05 = await fromNgffZarr(v05Store, { validate: true, version: "0.5" });

  await assertRejects(
    () => toNgffZarr(new Map<string, Uint8Array>(), v05, { version: "0.4" }),
    Error,
    "Writing OME-Zarr 0.4 is not supported",
  );
});
