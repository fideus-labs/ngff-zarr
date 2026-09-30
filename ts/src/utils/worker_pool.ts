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
 * Both pools are lazily initialized singletons whose size is controlled
 * by {@link config.workerPoolSize} (default:
 * `Math.min(navigator?.hardwareConcurrency || 4, 16)`).
 */

import type {
  GetWorkerOptions,
  SetWorkerOptions,
} from "@fideus-labs/fizarrita";
import { getWorker, setWorker } from "@fideus-labs/fizarrita";
import type { WorkerLike, WorkerPoolTask } from "@fideus-labs/worker-pool";
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
import { registry } from "zarrita";

import { installBloscShuffleNormalization } from "./blosc_registry.ts";
import type { CodecRegistry } from "./blosc_registry.ts";
import { config } from "../config.ts";

// Chunk encoding normally happens in a codec worker (which installs this
// itself — see ../workers/omero_codec_worker.ts), but the fallbacks below drop
// to zarrita's main-thread pipeline, so the main-thread registry needs it too.
installBloscShuffleNormalization(registry as unknown as CodecRegistry);

export type { ChunkCache } from "@fideus-labs/fizarrita";

// fizarrita uses npm:zarrita while this project uses jsr:@zarrita/zarrita.
// The types are structurally identical but TypeScript sees them as
// incompatible due to differing #private class fields. We bridge with
// explicit `as any` casts when calling into fizarrita.
// deno-lint-ignore no-explicit-any
type AnyZarrArray = any;

/** Whether SharedArrayBuffer is available in this environment. */
const USE_SHARED_ARRAY_BUFFER = typeof SharedArrayBuffer !== "undefined";

/**
 * Codec worker script backing the codec pool.
 *
 * This project's own worker rather than fizarrita's bundled one: it
 * speaks the same protocol but additionally normalizes Zarr v3 string
 * blosc shuffle modes — see {@link ./blosc_registry.ts}. Kept as a
 * single `new URL(..., import.meta.url)` expression so bundlers can
 * trace it, and so `scripts/inline_worker.ts` can swap in a blob URL for
 * the self-contained browser bundle.
 */
const CODEC_WORKER_URL = new URL(
  "../workers/omero_codec_worker.ts",
  import.meta.url,
);

// ---------------------------------------------------------------------------
// Codec pool — used by getWorker/setWorker for codec operations
// ---------------------------------------------------------------------------

let _codecPool: WorkerPool | null = null;

function getCodecPool(): WorkerPool {
  if (!_codecPool) {
    _codecPool = new WorkerPool(config.workerPoolSize);
  }
  return _codecPool;
}

// ---------------------------------------------------------------------------
// Write scheduling pool — bounds concurrent chunk write pipelines
// ---------------------------------------------------------------------------

let _writePool: WorkerPool | null = null;

function getWritePool(): WorkerPool {
  if (!_writePool) {
    _writePool = new WorkerPool(config.workerPoolSize);
  }
  return _writePool;
}

// ---------------------------------------------------------------------------
// Main-thread fallback
// ---------------------------------------------------------------------------

/**
 * Whether `err` is one of zarrita's structured errors carrying one of `tags`.
 *
 * Matched on `_tag` rather than with `zarr.isZarritaError`, whose
 * `instanceof` test fails across module copies: fizarrita raises its errors
 * from npm:zarrita, while under Deno this package imports jsr:@zarrita/zarrita.
 */
function isZarritaErrorTagged(err: unknown, ...tags: string[]): boolean {
  return err instanceof Error &&
    tags.includes((err as { _tag?: unknown })._tag as string);
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
 * Sharded (`sharding_indexed`) arrays decode their inner chunks on the
 * workers too.
 *
 * Falls back to `zarr.get()` when the workers cannot decode the array: a
 * codec registered only on the main thread, or a capability fizarrita lacks.
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
  try {
    const mergedOpts: GetWorkerOptions<unknown> = {
      pool: getCodecPool(),
      workerUrl: CODEC_WORKER_URL,
      useSharedArrayBuffer: USE_SHARED_ARRAY_BUFFER,
      ...opts,
    };
    // Cast through AnyZarrArray to bridge jsr:zarrita ↔ npm:zarrita types;
    // the store options are typed from the (npm) store, so they cross too.
    const result = await getWorker(
      arr as AnyZarrArray,
      selection ?? null,
      mergedOpts as AnyZarrArray,
    );
    return result as GetResult<D, Sel>;
  } catch (err) {
    if (isZarritaErrorTagged(err, "UnknownCodecError", "UnsupportedError")) {
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
 * Unlike `zarr.set()`, this writes sharded (`sharding_indexed`) arrays: each
 * touched shard is read (only when part of it is kept), reassembled, and
 * written whole. Concurrent writes to one shard race, the last one winning,
 * so callers must not write two regions of the same shard at once; writing
 * whole shards also spares the read.
 *
 * Falls back to `zarr.set()` when the workers cannot write the array: a
 * codec registered only on the main thread, or a data type such as `bool`
 * that the worker codecs do not handle. When `zarr.set()` cannot either, the
 * `AggregateError` thrown carries both errors.
 */
export async function zarrSet<D extends DataType>(
  arr: ZarrArray<D, Mutable>,
  selection: (number | Slice | null)[] | null,
  value: Scalar<D> | Chunk<D>,
  opts?: Partial<SetWorkerOptions>,
): Promise<void> {
  try {
    const mergedOpts: SetWorkerOptions = {
      pool: getCodecPool(),
      workerUrl: CODEC_WORKER_URL,
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
    if (isZarritaErrorTagged(err, "UnknownCodecError", "UnsupportedError")) {
      try {
        await zarr.set(arr, selection, value);
      } catch (fallbackErr) {
        // zarr.set() refuses every sharded array, so its error alone would
        // hide what the workers were actually missing.
        throw new AggregateError(
          [err, fallbackErr],
          "zarrSet: neither the codec workers nor zarr.set() can write the " +
            "array",
        );
      }
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
 * concurrency to {@link config.workerPoolSize}.
 */
export function createWriteQueue(): ChunkQueue {
  const pool = getWritePool();
  const tasks: WorkerPoolTask<void>[] = [];

  return {
    add(fn: () => Promise<void>) {
      tasks.push(async (worker: WorkerLike | null) => {
        await fn();
        // The write pool is a concurrency limiter, not a worker pool: `fn`
        // runs on this thread and never touches a worker, so none is created
        // and an empty slot is handed straight back.
        //
        // `WorkerPoolTask` types this field as non-null, hence the cast, but
        // an empty slot is the pool's own representation: `workerQueue` is
        // `Array<WorkerLike | null>`, starts out filled with `null`, and
        // `terminateWorkers()` skips null entries. Returning null recycles
        // the slot without spawning a thread that would do nothing.
        return {
          worker: worker ?? (null as unknown as WorkerLike),
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
