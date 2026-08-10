// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Main-thread RPC helpers for communicating with the omero codec worker.
 *
 * Follows the same persistent-dispatcher pattern as fizarrita's worker-rpc.ts:
 *   - One message listener per worker, routed by request ID
 *   - Meta-init protocol: codec metadata sent once per worker per array config
 *   - Zero-copy buffer transfer where possible
 *
 * Adds the `workerDecodeAndStats` function for the decode_and_stats message.
 */

import type { Chunk, DataType, TypedArrayConstructor } from "zarrita";
import type { CodecChunkMeta } from "@fideus-labs/fizarrita";
import type {
  WorkerErrorEventLike,
  WorkerLike,
  WorkerMessageEventLike,
} from "@fideus-labs/worker-pool";
import type { ChannelStatisticsAccumulator } from "./compute_omero-shared.ts";

// Local typed array constructor lookup (avoids depending on fizarrita internals
// from the main thread — the worker uses fizarrita's version)
function get_ctr<D extends DataType>(
  data_type: D,
): TypedArrayConstructor<D> {
  const ctr: TypedArrayConstructor<D> | undefined = ({
    int8: Int8Array,
    int16: Int16Array,
    int32: Int32Array,
    int64: globalThis.BigInt64Array,
    uint8: Uint8Array,
    uint16: Uint16Array,
    uint32: Uint32Array,
    uint64: globalThis.BigUint64Array,
    float32: Float32Array,
    float64: Float64Array,
  } as Record<string, unknown>)[data_type as string] as
    | TypedArrayConstructor<D>
    | undefined;
  if (!ctr) {
    throw new Error(`Unsupported data_type for worker codec: ${data_type}`);
  }
  return ctr;
}

// ---------------------------------------------------------------------------
// Persistent message dispatcher (same pattern as fizarrita's worker-rpc.ts)
// ---------------------------------------------------------------------------

interface PendingRequest {
  resolve: (data: unknown) => void;
  reject: (err: Error) => void;
}

class WorkerDispatcher {
  private pending = new Map<number, PendingRequest>();
  /** Tracks which metaIds have been sent to this worker. */
  private sentMetas = new Set<number>();

  constructor(private worker: WorkerLike) {
    worker.addEventListener("message", this.onMessage);
    worker.addEventListener("error", this.onError);
  }

  private onMessage = (event: WorkerMessageEventLike): void => {
    const { id } = event.data;
    const req = this.pending.get(id);
    if (!req) return;
    this.pending.delete(id);

    if (event.data.error) {
      req.reject(new Error(event.data.error));
    } else {
      req.resolve(event.data);
    }
  };

  private onError = (err: WorkerErrorEventLike): void => {
    const error = new Error(err.message ?? "Worker error");
    for (const req of this.pending.values()) {
      req.reject(error);
    }
    this.pending.clear();
  };

  send(
    id: number,
    message: unknown,
    transfer: Transferable[],
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(message, transfer);
    });
  }

  hasMeta(metaId: number): boolean {
    return this.sentMetas.has(metaId);
  }

  markMeta(metaId: number): void {
    this.sentMetas.add(metaId);
  }
}

/** Map from Worker to its dispatcher. WeakMap so dispatchers are GC'd with workers. */
const dispatchers = new WeakMap<WorkerLike, WorkerDispatcher>();

function getDispatcher(worker: WorkerLike): WorkerDispatcher {
  let d = dispatchers.get(worker);
  if (!d) {
    d = new WorkerDispatcher(worker);
    dispatchers.set(worker, d);
  }
  return d;
}

// ---------------------------------------------------------------------------
// Meta ID registry — assigns stable IDs to unique codec metadata
// ---------------------------------------------------------------------------

let nextMetaId = 0;
const metaKeyToId = new Map<string, number>();
const metaIdToMeta = new Map<number, CodecChunkMeta>();

/**
 * Get or create a stable metaId for the given codec metadata.
 * Uses JSON.stringify as the dedup key — called once per unique array config.
 */
export function getMetaId(meta: CodecChunkMeta): number {
  const key = JSON.stringify(meta);
  let id = metaKeyToId.get(key);
  if (id === undefined) {
    id = nextMetaId++;
    metaKeyToId.set(key, id);
    metaIdToMeta.set(id, meta);
  }
  return id;
}

// ---------------------------------------------------------------------------
// Request ID
// ---------------------------------------------------------------------------

let nextRequestId = 0;

// ---------------------------------------------------------------------------
// Buffer transfer helpers
// ---------------------------------------------------------------------------

function prepareTransferBuffer(
  buffer: ArrayBuffer,
  byteOffset: number,
  byteLength: number,
): ArrayBuffer {
  // Always copy: store-fetched bytes may be cached/shared
  return buffer.slice(byteOffset, byteOffset + byteLength);
}

// ---------------------------------------------------------------------------
// Ensure meta is initialized on the worker
// ---------------------------------------------------------------------------

async function ensureMeta(
  dispatcher: WorkerDispatcher,
  metaId: number,
): Promise<void> {
  if (dispatcher.hasMeta(metaId)) return;
  const meta = metaIdToMeta.get(metaId)!;
  const id = nextRequestId++;
  await dispatcher.send(id, { type: "init", id, metaId, meta }, []);
  dispatcher.markMeta(metaId);
}

// ---------------------------------------------------------------------------
// workerDecodeAndStats
// ---------------------------------------------------------------------------

interface DecodeAndStatsResponse {
  data: ArrayBuffer;
  shape: number[];
  stride: number[];
  accumulators: ChannelStatisticsAccumulator[];
}

/**
 * Send raw bytes to the omero codec worker for decoding AND per-channel
 * statistics computation. Returns both the decoded chunk (for caching)
 * and partial accumulators (for merging).
 *
 * @param worker - The Web Worker to send the message to
 * @param bytes - Raw encoded chunk bytes from the store
 * @param metaId - Stable ID for the codec metadata
 * @param meta - Codec metadata for this array
 * @param nChannels - Number of channels in the image
 * @param cIndex - Index of the channel dimension (-1 if no channel dim)
 * @param actualChunkShape - Edge chunk shape correction (optional)
 * @returns The decoded chunk and per-channel accumulators
 */
export async function workerDecodeAndStats<D extends DataType>(
  worker: WorkerLike,
  bytes: Uint8Array,
  metaId: number,
  meta: CodecChunkMeta,
  nChannels: number,
  cIndex: number,
  actualChunkShape?: number[],
): Promise<{
  chunk: Chunk<D>;
  accumulators: ChannelStatisticsAccumulator[];
}> {
  const dispatcher = getDispatcher(worker);
  await ensureMeta(dispatcher, metaId);

  const id = nextRequestId++;
  const transferBuffer = prepareTransferBuffer(
    bytes.buffer as ArrayBuffer,
    bytes.byteOffset,
    bytes.byteLength,
  );

  const response = (await dispatcher.send(
    id,
    {
      type: "decode_and_stats",
      id,
      bytes: transferBuffer,
      metaId,
      actualChunkShape,
      nChannels,
      cIndex,
    },
    [transferBuffer],
  )) as DecodeAndStatsResponse;

  // Reconstruct the TypedArray from the transferred buffer
  const Ctr = get_ctr(meta.data_type) as unknown as {
    new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ): Chunk<D>["data"];
    BYTES_PER_ELEMENT: number;
  };
  const data = new Ctr(
    response.data,
    0,
    response.data.byteLength / Ctr.BYTES_PER_ELEMENT,
  );
  const chunk: Chunk<D> = {
    data,
    shape: response.shape,
    stride: response.stride,
  };

  return { chunk, accumulators: response.accumulators };
}
