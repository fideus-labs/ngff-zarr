// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Centralized worker pool for zarrita get/set operations.
 *
 * Provides two pools:
 * - **Codec pool**: Used internally by `getWorker`/`setWorker` from fizarrita
 *   for offloading codec encode/decode to Web Workers.
 * - **Write scheduling pool**: Used by `createWriteQueue()` for bounding
 *   concurrent chunk-level write pipelines (get + transform + set).
 *
 * Both pools are lazily initialized singletons with pool size
 * `Math.min(navigator?.hardwareConcurrency || 4, 128)`.
 */

import type {
  GetWorkerOptions,
  SetWorkerOptions,
} from "@fideus-labs/fizarrita";
import { getWorker, setWorker } from "@fideus-labs/fizarrita";
import type { WorkerPoolTask } from "@fideus-labs/worker-pool";
import { WorkerPool } from "@fideus-labs/worker-pool";
import type {
  Array as ZarrArray,
  Chunk,
  DataType,
  Mutable,
  Readable,
  Scalar,
  Slice,
} from "zarrita";
import * as zarr from "zarrita";

export type { ChunkCache } from "@fideus-labs/fizarrita";

// fizarrita uses npm:zarrita while this project uses jsr:@zarrita/zarrita.
// The types are structurally identical but TypeScript sees them as
// incompatible due to differing #private class fields. We bridge with
// explicit `as any` casts when calling into fizarrita.
// deno-lint-ignore no-explicit-any
type AnyZarrArray = any;

/** Pool size for both codec and write scheduling pools. */
const POOL_SIZE = Math.min(
  (typeof navigator !== "undefined" && navigator?.hardwareConcurrency) || 4,
  128,
);

/** Whether SharedArrayBuffer is available in this environment. */
const USE_SHARED_ARRAY_BUFFER = typeof SharedArrayBuffer !== "undefined";

// ---------------------------------------------------------------------------
// Codec pool — used by getWorker/setWorker for codec operations
// ---------------------------------------------------------------------------

let _codecPool: WorkerPool | null = null;

function getCodecPool(): WorkerPool {
  if (!_codecPool) {
    _codecPool = new WorkerPool(POOL_SIZE);
  }
  return _codecPool;
}

// ---------------------------------------------------------------------------
// Write scheduling pool — bounds concurrent chunk write pipelines
// ---------------------------------------------------------------------------

let _writePool: WorkerPool | null = null;

function getWritePool(): WorkerPool {
  if (!_writePool) {
    _writePool = new WorkerPool(POOL_SIZE);
  }
  return _writePool;
}

// ---------------------------------------------------------------------------
// Early codec detection — skip workers for unsupported codecs
// ---------------------------------------------------------------------------

/** Codec names that fizarrita workers cannot handle. */
const UNSUPPORTED_CODEC_NAMES = new Set(["sharding_indexed"]);

const _textDecoder = new TextDecoder();

/**
 * Cache: maps `arr.path + store identity` → whether the array uses
 * unsupported codecs.  Uses a WeakMap keyed on the store so entries are
 * GC'd when the store is collected, with a nested Map keyed on the array
 * path string.
 */
// deno-lint-ignore no-explicit-any
const _codecCheckCache = new WeakMap<any, Map<string, boolean>>();

/**
 * Check whether a zarr array uses codecs that fizarrita workers cannot
 * handle (e.g. `sharding_indexed`).  Result is cached per store + path.
 *
 * Only inspects Zarr v3 `zarr.json`; v2 arrays always return `false`
 * (fizarrita handles all v2 codecs).
 */
async function hasUnsupportedCodecs<Store extends Readable>(
  arr: ZarrArray<DataType, Store>,
): Promise<boolean> {
  // Check cache first
  let pathMap = _codecCheckCache.get(arr.store);
  if (pathMap?.has(arr.path)) {
    return pathMap.get(arr.path)!;
  }

  const zarrJsonPath = (
    arr.path === "/" ? "/zarr.json" : `${arr.path}/zarr.json`
  ) as `/${string}`;
  const bytes = await arr.store.get(zarrJsonPath);

  let result = false;
  if (bytes) {
    try {
      const meta = JSON.parse(_textDecoder.decode(bytes));
      const codecs: Array<{ name: string }> = meta.codecs ?? [];
      result = codecs.some((c) => UNSUPPORTED_CODEC_NAMES.has(c.name));
    } catch {
      // Malformed metadata — let fizarrita/zarrita deal with it
      result = false;
    }
  }

  // Populate cache
  if (!pathMap) {
    pathMap = new Map();
    _codecCheckCache.set(arr.store, pathMap);
  }
  pathMap.set(arr.path, result);

  return result;
}

// ---------------------------------------------------------------------------
// Public API: zarrGet / zarrSet
// ---------------------------------------------------------------------------

// Return type alias for zarrGet — avoids repeating the conditional type.
type GetResult<
  D extends DataType,
  Sel extends (null | Slice | number)[],
> = null extends Sel[number] ? Chunk<D>
  : Slice extends Sel[number] ? Chunk<D>
  : Scalar<D>;

/**
 * Worker-accelerated zarr array read.
 *
 * Drop-in replacement for `zarr.get()` that offloads codec decode to Web
 * Workers via a shared WorkerPool. Uses SharedArrayBuffer when available.
 *
 * Falls back to `zarr.get()` when the array uses codecs not supported by
 * fizarrita (e.g. sharding_indexed).
 */
export async function zarrGet<
  D extends DataType,
  Store extends Readable,
  Sel extends (null | Slice | number)[],
>(
  arr: ZarrArray<D, Store>,
  selection?: Sel | null,
  opts?: Partial<GetWorkerOptions<unknown>>,
): Promise<GetResult<D, Sel>> {
  // Fast-path: skip workers entirely for arrays with unsupported codecs
  // (e.g. sharding_indexed). Avoids Worker creation that can hang on
  // some platforms (Windows CI).
  if (await hasUnsupportedCodecs(arr)) {
    return (await zarr.get(arr, selection)) as GetResult<D, Sel>;
  }

  try {
    const mergedOpts: GetWorkerOptions<unknown> = {
      pool: getCodecPool(),
      useSharedArrayBuffer: USE_SHARED_ARRAY_BUFFER,
      ...opts,
    };
    // Cast through AnyZarrArray to bridge jsr:zarrita ↔ npm:zarrita types
    const result = await getWorker(
      arr as AnyZarrArray,
      selection ?? null,
      mergedOpts,
    );
    return result as GetResult<D, Sel>;
  } catch (err) {
    // Fallback to zarr.get() for unsupported codecs (e.g. sharding_indexed)
    if (
      err instanceof Error &&
      /(?:unsupported|unknown) codec/i.test(err.message)
    ) {
      return (await zarr.get(arr, selection)) as GetResult<D, Sel>;
    }
    throw err;
  }
}

/**
 * Worker-accelerated zarr array write.
 *
 * Drop-in replacement for `zarr.set()` that offloads codec encode/decode to
 * Web Workers via a shared WorkerPool. Uses SharedArrayBuffer when available.
 *
 * Falls back to `zarr.set()` when the array uses codecs not supported by
 * fizarrita (e.g. sharding_indexed).
 */
export async function zarrSet<D extends DataType>(
  arr: ZarrArray<D, Mutable>,
  selection: (number | Slice | null)[] | null,
  value: Scalar<D> | Chunk<D>,
  opts?: Partial<SetWorkerOptions>,
): Promise<void> {
  // Fast-path: skip workers entirely for arrays with unsupported codecs
  if (await hasUnsupportedCodecs(arr)) {
    await zarr.set(arr, selection, value);
    return;
  }

  try {
    const mergedOpts: SetWorkerOptions = {
      pool: getCodecPool(),
      useSharedArrayBuffer: USE_SHARED_ARRAY_BUFFER,
      ...opts,
    };
    // Cast through AnyZarrArray to bridge jsr:zarrita ↔ npm:zarrita types
    await setWorker(
      arr as AnyZarrArray,
      selection,
      value as AnyZarrArray,
      mergedOpts,
    );
  } catch (err) {
    const isUnsupportedCodecError = err instanceof Error &&
      /(?:unsupported|unknown) codec/i.test(err.message);
    if (isUnsupportedCodecError) {
      // Fallback to zarr.set() for unsupported codecs (e.g. sharding_indexed)
      await zarr.set(arr, selection, value);
      return;
    }
    // Re-throw other errors so real failures aren't masked.
    throw new Error("zarrSet worker-accelerated write failed", { cause: err });
  }
}

// ---------------------------------------------------------------------------
// Public API: write queue (replaces PQueue-based create_queue)
// ---------------------------------------------------------------------------

/** Progress callback for chunk-level progress reporting. */
export type ChunkProgressCallback = (
  completedChunks: number,
  totalChunks: number,
) => void;

/** Interface for chunk write scheduling queue. */
export type ChunkQueue = {
  add(fn: () => Promise<void>): void;
  onIdle(onProgress?: ChunkProgressCallback | null): Promise<void>;
};

/**
 * Create a bounded-concurrency queue for chunk write scheduling.
 *
 * Uses a dedicated WorkerPool (separate from the codec pool) so that
 * outer scheduling tasks do not compete with inner codec worker tasks.
 *
 * Each queued function runs with one pool slot held; the pool bounds
 * concurrency to `Math.min(navigator?.hardwareConcurrency || 4, 128)`.
 */
export function createWriteQueue(): ChunkQueue {
  const pool = getWritePool();
  const tasks: WorkerPoolTask<void>[] = [];

  return {
    add(fn: () => Promise<void>) {
      tasks.push(async (worker: Worker | null) => {
        await fn();
        // Return the worker (or a dummy) — the write pool does not
        // actually use workers, only the concurrency slots matter.
        return {
          worker: worker ?? (null as unknown as Worker),
          result: undefined,
        };
      });
    },
    async onIdle(onProgress?: ChunkProgressCallback | null) {
      if (tasks.length === 0) return;
      const batch = tasks.splice(0, tasks.length);
      const { promise } = pool.runTasks(batch, onProgress ?? null);
      await promise;
    },
  };
}

// ---------------------------------------------------------------------------
// Public API: cleanup
// ---------------------------------------------------------------------------

/**
 * Terminate all workers in both the codec and write scheduling pools.
 *
 * Call this when the application is done with zarr operations to free
 * Web Worker resources.
 */
export function terminateWorkerPool(): void {
  if (_codecPool) {
    _codecPool.terminateWorkers();
    _codecPool = null;
  }
  if (_writePool) {
    _writePool.terminateWorkers();
    _writePool = null;
  }
}
