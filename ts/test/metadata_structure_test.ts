#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * Test to verify that 0.5 format metadata is written correctly
 *
 * This test writes data in 0.5 format and verifies the metadata structure
 * has the correct format with metadata under the "ome" property
 */

import { assertEquals, assertExists } from "@std/assert";
import { fromNgffZarr } from "../src/io/from_ngff_zarr.ts";
import { toNgffZarr } from "../src/io/to_ngff_zarr.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";
import * as zarr from "zarrita";
import { readImageNode } from "@itk-wasm/image-io";
import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";

Deno.test("Verify 0.5 metadata structure", async () => {
  // Create a simple test image
  const storePath = new URL(
    "../../py/test/data/input/cthead1.png",
    import.meta.url,
  );
  const resolvedPath = storePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");

  const itkImage = await readImageNode(resolvedPath);
  const ngffImage = await itkImageToNgffImage(itkImage);
  const multiscales = await toMultiscales(ngffImage);

  // Write as 0.5 format
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();
  await toNgffZarr(memoryStore, multiscales, { version: "0.5" });

  // Inspect the raw metadata structure
  const root = zarr.root(memoryStore);
  const rootGroup = await zarr.open(root, { kind: "group" });
  const attrs = rootGroup.attrs as Record<string, unknown>;

  console.log("Root attributes keys:", Object.keys(attrs));

  // Verify 0.5 structure: metadata should be under "ome"
  assertExists(attrs.ome, "Should have 'ome' property in root attributes");
  assertEquals(
    "multiscales" in (attrs.ome as Record<string, unknown>),
    true,
    "Should have 'multiscales' under 'ome'",
  );
  assertEquals(
    "version" in (attrs.ome as Record<string, unknown>),
    true,
    "Should have 'version' under 'ome'",
  );

  const omeAttrs = attrs.ome as Record<string, unknown>;
  assertEquals(omeAttrs.version, "0.5", "Version should be 0.5");

  // Verify multiscales structure
  const multiscalesArray = omeAttrs.multiscales as unknown[];
  assertExists(multiscalesArray, "Multiscales array should exist");
  assertEquals(Array.isArray(multiscalesArray), true, "Should be an array");
  assertEquals(
    multiscalesArray.length > 0,
    true,
    "Should have at least one element",
  );

  console.log("✓ 0.5 metadata structure is correct");

  // Now verify we can read it back
  const multiscalesRead = await fromNgffZarr(memoryStore, {
    version: "0.5",
    validate: true,
  });
  assertExists(multiscalesRead, "Should be able to read back 0.5 format");
  assertEquals(multiscalesRead.images.length > 0, true, "Should have images");

  console.log("✓ 0.5 format can be read back successfully");
});

Deno.test("Verify 0.4 metadata structure", async () => {
  // Create a simple test image
  const storePath = new URL(
    "../../py/test/data/input/cthead1.png",
    import.meta.url,
  );
  const resolvedPath = storePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");

  const itkImage = await readImageNode(resolvedPath);
  const ngffImage = await itkImageToNgffImage(itkImage);
  const multiscales = await toMultiscales(ngffImage);

  // Write as 0.4 format
  const memoryStore: MemoryStore = new Map<string, Uint8Array>();
  await toNgffZarr(memoryStore, multiscales, { version: "0.4" });

  // Inspect the raw metadata structure
  const root = zarr.root(memoryStore);
  const rootGroup = await zarr.open(root, { kind: "group" });
  const attrs = rootGroup.attrs as Record<string, unknown>;

  console.log("Root attributes keys:", Object.keys(attrs));

  // Verify 0.4 structure: metadata should be directly at root
  assertExists(
    attrs.multiscales,
    "Should have 'multiscales' property in root attributes",
  );
  assertEquals(
    "ome" in attrs,
    false,
    "Should NOT have 'ome' property in 0.4 format",
  );

  // Verify multiscales structure
  const multiscalesArray = attrs.multiscales as unknown[];
  assertExists(multiscalesArray, "Multiscales array should exist");
  assertEquals(Array.isArray(multiscalesArray), true, "Should be an array");
  assertEquals(
    multiscalesArray.length > 0,
    true,
    "Should have at least one element",
  );

  const firstMultiscale = multiscalesArray[0] as Record<string, unknown>;
  assertEquals(firstMultiscale.version, "0.4", "Version should be 0.4");

  console.log("✓ 0.4 metadata structure is correct");

  // Now verify we can read it back
  const multiscalesRead = await fromNgffZarr(memoryStore, {
    version: "0.4",
    validate: true,
  });
  assertExists(multiscalesRead, "Should be able to read back 0.4 format");
  assertEquals(multiscalesRead.images.length > 0, true, "Should have images");

  console.log("✓ 0.4 format can be read back successfully");
});
