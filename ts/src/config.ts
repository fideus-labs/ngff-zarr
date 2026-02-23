// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Runtime configuration for @fideus-labs/ngff-zarr.
 *
 * Settings can be changed at any time before the first pool usage.
 * To change pool size after pools have already been created, use
 * {@link setWorkerPoolSize} which terminates existing pools so they
 * are lazily recreated with the new size.
 */

/** Default pool size: `Math.min(navigator.hardwareConcurrency || 4, 16)`. */
const DEFAULT_POOL_SIZE = Math.min(
  (typeof navigator !== "undefined" && navigator?.hardwareConcurrency) || 4,
  16,
);

/**
 * Global configuration object.
 *
 * @example
 * ```ts
 * import { config } from "@fideus-labs/ngff-zarr";
 *
 * // Set before first pool usage — no teardown needed.
 * config.workerPoolSize = 8;
 * ```
 */
export const config = {
  /**
   * Number of Web Workers used by each worker pool (codec, write, and
   * omero).  Defaults to `Math.min(navigator.hardwareConcurrency || 4, 16)`.
   *
   * Changing this value only affects pools created **after** the change.
   * Use {@link setWorkerPoolSize} to also terminate existing pools.
   */
  workerPoolSize: DEFAULT_POOL_SIZE,
};

/**
 * Set the worker pool size and terminate all existing pools.
 *
 * This is a convenience wrapper that updates {@link config.workerPoolSize}
 * and tears down the codec, write-scheduling, and omero worker pools so
 * they are lazily recreated with the new size on next use.
 *
 * @param size - Positive integer for the new pool size.
 *
 * @example
 * ```ts
 * import { setWorkerPoolSize } from "@fideus-labs/ngff-zarr";
 *
 * // Change pool size at runtime (terminates existing workers).
 * await setWorkerPoolSize(4);
 * ```
 */
export async function setWorkerPoolSize(size: number): Promise<void> {
  if (!Number.isInteger(size) || size < 1) {
    throw new Error(`workerPoolSize must be a positive integer, got ${size}`);
  }
  config.workerPoolSize = size;

  // Dynamic imports to avoid circular dependencies — config.ts is imported
  // by both worker_pool.ts and compute_omero.ts.
  const { terminateWorkerPool } = await import("./utils/worker_pool.ts");
  const { terminateOmeroWorkerPool } = await import(
    "./utils/compute_omero.ts"
  );
  terminateWorkerPool();
  terminateOmeroWorkerPool();
}
