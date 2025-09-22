// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Example demonstrating zarrita v0.5.2 API improvements
 *
 * This example shows the key improvements in zarrita v0.5.2:
 * - Better TypeScript support with type inference
 * - Location-based API with zarr.root()
 * - Consolidated metadata support for better performance
 * - Version-specific openers (v2/v3)
 */

import * as zarr from "zarrita";

// Example 1: Basic usage with improved API
async function _basicExample() {
  console.log("=== Basic zarrita v0.5.2 Example ===");

  // Create a Location using zarr.root()
  const store = new zarr.FetchStore(
    "https://raw.githubusercontent.com/zarr-developers/zarr_implementations/5dc998ac72/examples/zarr.zr/blosc",
  );
  const root = zarr.root(store);

  // Open array with automatic version detection
  const arr = await zarr.open(root, { kind: "array" });

  console.log("Array properties:");
  console.log(`- Shape: ${JSON.stringify(arr.shape)}`);
  console.log(`- Chunks: ${JSON.stringify(arr.chunks)}`);
  console.log(`- Data type: ${arr.dtype}`);

  // Read a region using the improved get API
  const region = await zarr.get(arr, [null, null, 0]);
  console.log(`- Region shape: ${JSON.stringify(region.shape)}`);
  console.log(`- Data type: ${region.data.constructor.name}`);
}

// Example 2: Using consolidated metadata for better performance
async function _consolidatedExample() {
  console.log("\n=== Consolidated Metadata Example ===");

  const store = new zarr.FetchStore("https://example.com/data.zarr");

  try {
    // Try to use consolidated metadata (reduces network requests)
    const consolidatedStore = await zarr.tryWithConsolidated(store);
    const root = zarr.root(consolidatedStore);

    // These operations won't require additional metadata requests
    const _group = await zarr.open(root, { kind: "group" });
    console.log("Successfully opened with consolidated metadata");

    // List known contents without network requests
    if ("contents" in consolidatedStore) {
      const contents = (consolidatedStore as { contents: () => unknown })
        .contents();
      console.log("Known contents:", contents);
    }
  } catch (_error) {
    console.log(
      "Consolidated metadata not available, falling back to regular store",
    );
    const root = zarr.root(store);
    const _group = await zarr.open(root, { kind: "group" });
  }
}

// Example 3: Version-specific opening
async function _versionSpecificExample() {
  console.log("\n=== Version-Specific Opening Example ===");

  const store = new zarr.FetchStore("https://example.com/data.zarr");
  const root = zarr.root(store);

  try {
    // Explicitly open as Zarr v2
    const _v2Array = await zarr.open.v2(root, { kind: "array" });
    console.log("Opened as Zarr v2 array");
  } catch (_error) {
    console.log("Not a Zarr v2 array");
  }

  try {
    // Explicitly open as Zarr v3
    const _v3Array = await zarr.open.v3(root, { kind: "array" });
    console.log("Opened as Zarr v3 array");
  } catch (_error) {
    console.log("Not a Zarr v3 array");
  }
}

// Example 4: Type inference and type guards
async function _typeInferenceExample() {
  console.log("\n=== Type Inference Example ===");

  const store = new zarr.FetchStore("https://example.com/data.zarr");
  const root = zarr.root(store);
  const arr = await zarr.open(root, { kind: "array" });

  // Use type guards for better TypeScript support
  if (arr.is("int64")) {
    const _chunk = await arr.getChunk([0, 0]);
    // TypeScript now knows chunk.data is BigInt64Array
    console.log("Working with BigInt64Array data");
  }

  if (arr.is("number")) {
    const _chunk = await arr.getChunk([0, 0]);
    // TypeScript knows this is a numeric typed array
    console.log("Working with numeric typed array");
  }

  // Primitive type guards
  if (arr.is("bigint")) {
    console.log("Array contains bigint data");
  }

  if (arr.is("string")) {
    console.log("Array contains string data");
  }
}

// Example 5: Slicing with improved API
async function _slicingExample() {
  console.log("\n=== Slicing Example ===");

  const store = new zarr.FetchStore("https://example.com/3d-data.zarr");
  const root = zarr.root(store);
  const arr = await zarr.open(root, { kind: "array" });

  // Slice using zarr.slice() utility
  const region = await zarr.get(arr, [zarr.slice(10, 20), null, 0]);
  console.log(`Sliced region shape: ${JSON.stringify(region.shape)}`);

  // This is equivalent to Python: arr[10:20, :, 0]
}

// Run examples (commented out to avoid network requests in this demo)
if (import.meta.main) {
  console.log("Zarrita v0.5.2 API Examples");
  console.log("===========================");

  // Uncomment to run examples:
  // await basicExample();
  // await consolidatedExample();
  // await versionSpecificExample();
  // await typeInferenceExample();
  // await slicingExample();

  console.log("\nExamples are commented out to avoid network requests.");
  console.log("Uncomment the function calls above to run them.");
}
