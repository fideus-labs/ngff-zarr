// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Tests for mergeAccumulators in compute_omero-shared.ts.
 * Focuses on proportional sampling and unbiased reservoir merging.
 */

import { assertEquals, assertExists } from "@std/assert";
import {
  createAccumulator,
  finalizeStatistics,
  mergeAccumulators,
  updateAccumulator,
} from "../src/utils/compute_omero-shared.ts";

// ============================================================================
// Basic merging tests
// ============================================================================

Deno.test("mergeAccumulators: two empty accumulators", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  const merged = mergeAccumulators(a, b);

  assertEquals(merged.count, 0);
  assertEquals(merged.sample.length, 0);
  assertEquals(merged.min, Math.min(a.min, b.min));
  assertEquals(merged.max, Math.max(a.max, b.max));
});

Deno.test("mergeAccumulators: one empty, one with data", () => {
  const empty = createAccumulator();
  const withData = createAccumulator();
  const data = [1, 2, 3, 4, 5];
  updateAccumulator(withData, data);

  const merged = mergeAccumulators(empty, withData);

  assertEquals(merged.count, 5);
  assertEquals(merged.min, 1);
  assertEquals(merged.max, 5);
  // Should have all samples from withData since empty contributes nothing
  assertEquals(merged.sample.length, 5);
});

Deno.test("mergeAccumulators: preserves min/max from both", () => {
  const a = createAccumulator();
  const b = createAccumulator();
  updateAccumulator(a, [5, 10, 15]);
  updateAccumulator(b, [1, 20, 25]);

  const merged = mergeAccumulators(a, b);

  assertEquals(merged.min, 1); // From b
  assertEquals(merged.max, 25); // From b
});

// ============================================================================
// Proportional sampling tests
// ============================================================================

Deno.test("mergeAccumulators: equal-sized populations get equal representation", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  // Both have 100 elements
  updateAccumulator(a, Array(100).fill(1));
  updateAccumulator(b, Array(100).fill(2));

  const merged = mergeAccumulators(a, b);

  assertEquals(merged.count, 200);

  // Each accumulator should contribute roughly half of the samples
  // Count how many 1s vs 2s in the merged sample
  const count1s = merged.sample.filter((x) => x === 1).length;
  const count2s = merged.sample.filter((x) => x === 2).length;

  // With equal populations, expect roughly 50/50 split
  // Allow some variance due to random sampling
  const ratio = count1s / (count1s + count2s);
  assertEquals(ratio > 0.4 && ratio < 0.6, true, `ratio=${ratio}`);
});

Deno.test("mergeAccumulators: 10:1 population ratio", () => {
  const large = createAccumulator();
  const small = createAccumulator();

  // Large chunk: 1000 elements
  updateAccumulator(large, Array(1000).fill(100));
  // Small chunk: 100 elements
  updateAccumulator(small, Array(100).fill(200));

  const merged = mergeAccumulators(large, small);

  assertEquals(merged.count, 1100);

  // Count representation in sample
  const countLarge = merged.sample.filter((x) => x === 100).length;
  const countSmall = merged.sample.filter((x) => x === 200).length;

  // Large should have ~10x more samples than small
  // Expected ratio: 1000/1100 ≈ 0.909
  const ratio = countLarge / (countLarge + countSmall);
  assertEquals(
    ratio > 0.85 && ratio < 0.95,
    true,
    `ratio=${ratio}, should be ~0.909`,
  );
});

Deno.test("mergeAccumulators: 100:1 population ratio (edge chunk scenario)", () => {
  const fullChunk = createAccumulator();
  const edgeChunk = createAccumulator();

  // Full chunk: 10000 voxels
  updateAccumulator(fullChunk, Array(10000).fill(50));
  // Edge chunk: 100 voxels
  updateAccumulator(edgeChunk, Array(100).fill(150));

  const merged = mergeAccumulators(fullChunk, edgeChunk);

  assertEquals(merged.count, 10100);

  const countFull = merged.sample.filter((x) => x === 50).length;
  const countEdge = merged.sample.filter((x) => x === 150).length;

  // Expected ratio: 10000/10100 ≈ 0.99
  const ratio = countFull / (countFull + countEdge);
  assertEquals(
    ratio > 0.97 && ratio < 1.0,
    true,
    `ratio=${ratio}, should be ~0.99`,
  );
});

// ============================================================================
// Tests for correct quantile estimation after merging
// ============================================================================

Deno.test("mergeAccumulators: quantiles remain accurate after merge", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  // a: values 0-499
  updateAccumulator(a, Array.from({ length: 500 }, (_, i) => i));
  // b: values 500-999
  updateAccumulator(b, Array.from({ length: 500 }, (_, i) => i + 500));

  const merged = mergeAccumulators(a, b);

  // Finalize with 2% and 98% quantiles
  const stats = finalizeStatistics(merged, [0.02, 0.98]);

  assertEquals(merged.count, 1000);
  assertEquals(stats.min, 0);
  assertEquals(stats.max, 999);

  // 2% of 1000 = 20, 98% = 980
  // Allow some sampling error
  assertEquals(
    stats.qLow >= 0 && stats.qLow <= 40,
    true,
    `qLow=${stats.qLow}`,
  );
  assertEquals(
    stats.qHigh >= 960 && stats.qHigh <= 999,
    true,
    `qHigh=${stats.qHigh}`,
  );
});

Deno.test("mergeAccumulators: bimodal distribution", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  // a: all zeros (background)
  updateAccumulator(a, Array(1000).fill(0));
  // b: all 255s (foreground)
  updateAccumulator(b, Array(1000).fill(255));

  const merged = mergeAccumulators(a, b);
  const stats = finalizeStatistics(merged, [0.02, 0.98]);

  assertEquals(merged.count, 2000);
  assertEquals(stats.min, 0);
  assertEquals(stats.max, 255);

  // With equal populations, both values should be well represented
  const count0s = merged.sample.filter((x) => x === 0).length;
  const count255s = merged.sample.filter((x) => x === 255).length;

  // Expect roughly 50/50 split
  const ratio = count0s / (count0s + count255s);
  assertEquals(ratio > 0.4 && ratio < 0.6, true, `ratio=${ratio}`);
});

// ============================================================================
// Tests for multiple sequential merges
// ============================================================================

Deno.test("mergeAccumulators: chaining multiple merges", () => {
  const acc1 = createAccumulator();
  const acc2 = createAccumulator();
  const acc3 = createAccumulator();
  const acc4 = createAccumulator();

  // Each chunk has different size and values
  updateAccumulator(acc1, Array(100).fill(10)); // Small
  updateAccumulator(acc2, Array(1000).fill(20)); // Large
  updateAccumulator(acc3, Array(100).fill(30)); // Small
  updateAccumulator(acc4, Array(1000).fill(40)); // Large

  // Merge in sequence (simulating parallel reduction)
  const merged12 = mergeAccumulators(acc1, acc2);
  const merged34 = mergeAccumulators(acc3, acc4);
  const final = mergeAccumulators(merged12, merged34);

  assertEquals(final.count, 2200);
  assertEquals(final.min, 10);
  assertEquals(final.max, 40);

  // Count representation
  const counts = {
    10: final.sample.filter((x) => x === 10).length,
    20: final.sample.filter((x) => x === 20).length,
    30: final.sample.filter((x) => x === 30).length,
    40: final.sample.filter((x) => x === 40).length,
  };

  // Large chunks (20, 40) should dominate with ~1000/2200 ≈ 0.45 each
  // Small chunks (10, 30) should each have ~100/2200 ≈ 0.045
  const total = counts[10] + counts[20] + counts[30] + counts[40];
  const ratio20 = counts[20] / total;
  const ratio40 = counts[40] / total;
  const ratio10 = counts[10] / total;
  const ratio30 = counts[30] / total;

  // Large chunks: expect ~0.45 each (allow 0.35-0.55)
  assertEquals(
    ratio20 > 0.35 && ratio20 < 0.55,
    true,
    `ratio20=${ratio20}`,
  );
  assertEquals(
    ratio40 > 0.35 && ratio40 < 0.55,
    true,
    `ratio40=${ratio40}`,
  );

  // Small chunks: expect ~0.045 each (allow 0.01-0.15)
  assertEquals(
    ratio10 > 0.01 && ratio10 < 0.15,
    true,
    `ratio10=${ratio10}`,
  );
  assertEquals(
    ratio30 > 0.01 && ratio30 < 0.15,
    true,
    `ratio30=${ratio30}`,
  );
});

// ============================================================================
// Tests for Fisher-Yates partial shuffle
// ============================================================================

Deno.test("mergeAccumulators: randomness in subsampling", () => {
  // Run the same merge multiple times and verify different samples
  const createTestAccumulator = () => {
    const acc = createAccumulator();
    updateAccumulator(acc, Array.from({ length: 100 }, (_, i) => i));
    return acc;
  };

  const a1 = createTestAccumulator();
  const b1 = createTestAccumulator();
  const merged1 = mergeAccumulators(a1, b1);

  const a2 = createTestAccumulator();
  const b2 = createTestAccumulator();
  const merged2 = mergeAccumulators(a2, b2);

  // The samples should be different due to random subsampling
  // (very unlikely to be identical)
  const samplesEqual = JSON.stringify(merged1.sample.sort()) ===
    JSON.stringify(merged2.sample.sort());

  assertEquals(samplesEqual, false, "samples should differ due to randomness");
});

// ============================================================================
// Tests for reservoir size limits
// ============================================================================

Deno.test("mergeAccumulators: respects QUANTILE_SAMPLE_SIZE limit", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  // Each accumulator has large sample (will be at QUANTILE_SAMPLE_SIZE=10000)
  updateAccumulator(a, Array.from({ length: 20000 }, (_, i) => i));
  updateAccumulator(b, Array.from({ length: 20000 }, (_, i) => i + 20000));

  const merged = mergeAccumulators(a, b);

  // Merged sample should not exceed 10000
  assertEquals(
    merged.sample.length <= 10000,
    true,
    `sample.length=${merged.sample.length}`,
  );
  // Should be exactly 10000 (sum of proportional targets)
  assertEquals(merged.sample.length, 10000);
});

Deno.test("mergeAccumulators: small accumulators preserve all samples", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  updateAccumulator(a, [1, 2, 3]);
  updateAccumulator(b, [4, 5, 6]);

  const merged = mergeAccumulators(a, b);

  // With only 6 total samples, all should be preserved
  assertEquals(merged.sample.length, 6);
  assertEquals(merged.count, 6);

  // Verify all values present
  const sortedSample = merged.sample.sort((x, y) => x - y);
  assertEquals(sortedSample, [1, 2, 3, 4, 5, 6]);
});

// ============================================================================
// Tests for edge cases
// ============================================================================

Deno.test("mergeAccumulators: one accumulator has full reservoir", () => {
  const full = createAccumulator();
  const small = createAccumulator();

  // Fill one accumulator to reservoir limit
  updateAccumulator(full, Array.from({ length: 15000 }, () => 100));
  // Other has few samples
  updateAccumulator(small, Array(100).fill(200));

  const merged = mergeAccumulators(full, small);

  assertEquals(merged.count, 15100);
  assertEquals(merged.sample.length, 10000);

  // The large accumulator should dominate (~99% of samples)
  const countFull = merged.sample.filter((x) => x === 100).length;
  const countSmall = merged.sample.filter((x) => x === 200).length;

  const ratio = countFull / (countFull + countSmall);
  assertEquals(
    ratio > 0.98,
    true,
    `ratio=${ratio}, should be ~0.993`,
  );
});

Deno.test("mergeAccumulators: handles NaN gracefully", () => {
  const a = createAccumulator();
  const b = createAccumulator();

  // Accumulators with valid data (NaN should have been filtered in updateAccumulator)
  updateAccumulator(a, [1, 2, 3]);
  updateAccumulator(b, [4, 5, 6]);

  const merged = mergeAccumulators(a, b);

  // Verify no NaN in results
  const hasNaN = merged.sample.some((x) => Number.isNaN(x));
  assertEquals(hasNaN, false);
  assertEquals(Number.isNaN(merged.min), false);
  assertEquals(Number.isNaN(merged.max), false);
});

Deno.test("mergeAccumulators: commutative property", () => {
  const a1 = createAccumulator();
  const b1 = createAccumulator();
  const a2 = createAccumulator();
  const b2 = createAccumulator();

  const data_a = Array.from({ length: 100 }, (_, i) => i);
  const data_b = Array.from({ length: 200 }, (_, i) => i + 100);

  updateAccumulator(a1, data_a);
  updateAccumulator(b1, data_b);
  updateAccumulator(a2, data_a);
  updateAccumulator(b2, data_b);

  const merged1 = mergeAccumulators(a1, b1);
  const merged2 = mergeAccumulators(b2, a2); // Swapped order

  // Basic properties should be identical
  assertEquals(merged1.count, merged2.count);
  assertEquals(merged1.min, merged2.min);
  assertEquals(merged1.max, merged2.max);
  assertEquals(merged1.sample.length, merged2.sample.length);

  // Samples may differ due to randomness, but statistics should be similar
  const stats1 = finalizeStatistics(merged1, [0.02, 0.98]);
  const stats2 = finalizeStatistics(merged2, [0.02, 0.98]);

  // Allow small differences due to sampling
  assertEquals(
    Math.abs(stats1.qLow - stats2.qLow) < 20,
    true,
    `qLow diff=${Math.abs(stats1.qLow - stats2.qLow)}`,
  );
  assertEquals(
    Math.abs(stats1.qHigh - stats2.qHigh) < 20,
    true,
    `qHigh diff=${Math.abs(stats1.qHigh - stats2.qHigh)}`,
  );
});

// ============================================================================
// Real-world scenario tests
// ============================================================================

Deno.test("mergeAccumulators: typical OME-Zarr chunk scenario", () => {
  // Simulate real-world OME-Zarr chunking scenario:
  // - Most chunks are full size (256x256 = 65536 voxels)
  // - Edge chunks are smaller (e.g., 128x256 = 32768 voxels)

  const fullChunks = [];
  const edgeChunks = [];

  // Create 8 full chunks
  for (let i = 0; i < 8; i++) {
    const acc = createAccumulator();
    // Full chunk with values around 100
    updateAccumulator(acc, Array(65536).fill(100 + i * 10));
    fullChunks.push(acc);
  }

  // Create 2 edge chunks
  for (let i = 0; i < 2; i++) {
    const acc = createAccumulator();
    // Edge chunk with values around 200
    updateAccumulator(acc, Array(32768).fill(200 + i * 10));
    edgeChunks.push(acc);
  }

  // Merge all chunks
  let merged = fullChunks[0];
  for (let i = 1; i < fullChunks.length; i++) {
    merged = mergeAccumulators(merged, fullChunks[i]);
  }
  for (const edge of edgeChunks) {
    merged = mergeAccumulators(merged, edge);
  }

  // Total voxels: 8*65536 + 2*32768 = 524288 + 65536 = 589824
  assertEquals(merged.count, 589824);
  assertExists(merged.sample);
  assertEquals(merged.sample.length, 10000);

  // Full chunks should dominate: 524288/589824 ≈ 0.889
  const countFull = merged.sample.filter((x) => x >= 100 && x < 200).length;
  const countEdge = merged.sample.filter((x) => x >= 200).length;

  const ratio = countFull / (countFull + countEdge);
  assertEquals(
    ratio > 0.85 && ratio < 0.92,
    true,
    `ratio=${ratio}, expected ~0.889`,
  );
});
