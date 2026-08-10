// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * This project's codec Web Worker. Backs both the codec pool used by
 * `zarrGet`/`zarrSet` and the OMERO statistics pool.
 *
 * Runs in a browser `DedicatedWorkerGlobalScope` and in a
 * `node:worker_threads` thread. The two runtimes differ only in how messages
 * arrive and leave, so the plumbing is selected at startup and everything
 * below it is shared.
 *
 * The standard codec protocol (`init`, `decode`, `decode_into`, `encode`) is
 * delegated to fizarrita's exported {@link handleCodecMessage}, which owns the
 * pipeline cache and the edge-chunk correction. This worker adds:
 *
 * - `decode_and_stats` — decode a chunk *and* compute per-channel statistics
 *   in a single round-trip, saving a second transfer of the decoded data.
 * - The blosc shuffle registry patch, so Zarr v3's string shuffle modes reach
 *   numcodecs in the integer form it expects (see
 *   {@link ../utils/blosc_registry.ts}). It has to be installed here because
 *   chunks are encoded in this module graph, not the main thread's.
 *
 * Message protocol (superset of fizarrita's codec worker):
 *   init:             → init_ok              (register codec pipeline metadata)
 *   decode:           → decoded              (decode raw bytes, return chunk)
 *   decode_into:      → decode_into_ok       (decode into SharedArrayBuffer)
 *   encode:           → encoded              (encode chunk data)
 *   decode_and_stats: → decoded_and_stats    (decode + compute channel stats)
 */

import type { CodecChunkMeta } from "@fideus-labs/fizarrita";
import { handleCodecMessage } from "@fideus-labs/fizarrita";
import type { CodecWorkerMessage } from "@fideus-labs/fizarrita/internals/codec-worker-core";
import { get_ctr } from "@fideus-labs/fizarrita/internals/util";
import { isNodeRuntime } from "@fideus-labs/worker-pool";
import type { DataType } from "zarrita";
import { registry } from "zarrita";
// `handleCodecMessage` resolves its registry through fizarrita, an npm
// package, so under Deno that is npm:zarrita rather than the
// jsr:@zarrita/zarrita this package imports as "zarrita" — separate module
// instances with separate registries. Patch both; in the npm build the two
// specifiers collapse to one module and the installer's idempotency guard
// makes the second call a no-op.
import { registry as npmRegistry } from "npm:zarrita@^0.6.1";

import { installBloscShuffleNormalization } from "../utils/blosc_registry.ts";
import type { CodecRegistry } from "../utils/blosc_registry.ts";
import {
  createAccumulator,
  extractChannel,
  updateAccumulator,
} from "../utils/compute_omero-shared.ts";

installBloscShuffleNormalization(registry as unknown as CodecRegistry);
installBloscShuffleNormalization(npmRegistry as unknown as CodecRegistry);

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/** Decode a chunk and compute its per-channel statistics in one round-trip. */
interface DecodeAndStatsMessage {
  type: "decode_and_stats";
  id: number;
  bytes: ArrayBuffer;
  metaId: number;
  actualChunkShape?: number[];
  nChannels: number;
  cIndex: number;
}

type WorkerMessage = CodecWorkerMessage | DecodeAndStatsMessage;

/** Accumulator in the plain-object form that survives structured clone. */
interface SerializedAccumulator {
  min: number;
  max: number;
  sample: number[];
  count: number;
}

/** A reply to post back, with the buffers to hand over rather than copy. */
interface Reply {
  response: Record<string, unknown>;
  transfer: ArrayBuffer[];
}

// ---------------------------------------------------------------------------
// decode_and_stats
// ---------------------------------------------------------------------------

/**
 * Element type per `metaId`, recorded from the `init` messages on their way
 * through to {@link handleCodecMessage}.
 *
 * `decode_and_stats` reuses the core `decode` path, which hands back a raw
 * `ArrayBuffer`; reading statistics out of it needs the array's data type to
 * pick a typed-array view.
 */
const dataTypeByMetaId = new Map<number, DataType>();

function rememberDataType(metaId: number, meta: CodecChunkMeta): void {
  dataTypeByMetaId.set(metaId, meta.data_type);
}

/**
 * Handle `decode_and_stats` by delegating the decode to the shared core and
 * computing statistics over the result.
 *
 * Reusing the core's `decode` keeps a single codec pipeline cache and a single
 * implementation of the edge-chunk correction.
 */
async function handleDecodeAndStats(
  msg: DecodeAndStatsMessage,
): Promise<Reply> {
  const dataType = dataTypeByMetaId.get(msg.metaId);
  if (dataType === undefined) {
    return {
      response: {
        type: "decoded_and_stats",
        id: msg.id,
        error: `No metadata for metaId ${msg.metaId}. Send an 'init' first.`,
      },
      transfer: [],
    };
  }

  const decoded = await handleCodecMessage({
    type: "decode",
    id: msg.id,
    bytes: msg.bytes,
    metaId: msg.metaId,
    // Spread rather than assign: `exactOptionalPropertyTypes` rejects an
    // explicit `undefined` for an optional property.
    ...(msg.actualChunkShape ? { actualChunkShape: msg.actualChunkShape } : {}),
  });

  // A decode failure comes back as a `decoded` response carrying `error`.
  // Relabel it so the caller's `decoded_and_stats` request settles.
  if (decoded === null || decoded.response.error !== undefined) {
    return {
      response: {
        type: "decoded_and_stats",
        id: msg.id,
        error: decoded?.response.error ??
          "Codec worker did not handle the decode request",
      },
      transfer: [],
    };
  }

  const { data, shape, stride } = decoded.response as {
    data: ArrayBuffer;
    shape: number[];
    stride: number[];
  };

  const Ctr = get_ctr(dataType);
  const chunkData = new Ctr(data) as unknown as ArrayLike<number>;

  const accumulators: SerializedAccumulator[] = [];
  if (msg.cIndex >= 0) {
    // Multi-channel: extract each channel and accumulate separately.
    for (let ch = 0; ch < msg.nChannels; ch++) {
      const channelData = extractChannel(chunkData, shape, ch, msg.cIndex);
      const acc = createAccumulator();
      updateAccumulator(acc, channelData);
      accumulators.push(acc);
    }
  } else {
    const acc = createAccumulator();
    updateAccumulator(acc, chunkData);
    accumulators.push(acc);
  }

  // Hand the decoded buffer back too, so the caller can cache the chunk
  // instead of decoding it a second time.
  return {
    response: {
      type: "decoded_and_stats",
      id: msg.id,
      data,
      shape,
      stride,
      accumulators,
    },
    transfer: decoded.transfer,
  };
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/** Response type to report a failure under, per request type. */
const ERROR_RESPONSE_TYPE: Record<string, string> = {
  init: "init_ok",
  decode: "decoded",
  decode_into: "decode_into_ok",
  encode: "encoded",
  decode_and_stats: "decoded_and_stats",
};

/**
 * Route one request. Never rejects — a failure is reported as the request's
 * normal response type carrying an `error` string, which is what the
 * main-thread dispatcher routes back to the waiting caller.
 */
async function handleMessage(msg: WorkerMessage): Promise<Reply | null> {
  try {
    if (msg.type === "init") {
      rememberDataType(msg.metaId, msg.meta);
      return await handleCodecMessage(msg);
    }
    if (msg.type === "decode_and_stats") {
      return await handleDecodeAndStats(msg);
    }
    return await handleCodecMessage(msg);
  } catch (error) {
    return {
      response: {
        type: ERROR_RESPONSE_TYPE[msg.type] ?? "init_ok",
        id: (msg as { id: number }).id,
        error: error instanceof Error ? error.message : String(error),
      },
      transfer: [],
    };
  }
}

/** Post a reply, falling back to an error-only reply if posting fails. */
function postReply(
  post: (response: unknown, transfer: ArrayBuffer[]) => void,
  reply: Reply,
): void {
  try {
    post(reply.response, reply.transfer);
  } catch (error) {
    // Posting can still fail on its own — an unclonable value, a buffer that
    // is no longer transferable. Retry without the payload so the caller gets
    // a rejection instead of a request that never settles.
    post(
      {
        type: reply.response.type,
        id: reply.response.id,
        error: error instanceof Error ? error.message : String(error),
      },
      [],
    );
  }
}

// ---------------------------------------------------------------------------
// Runtime plumbing
// ---------------------------------------------------------------------------

/** The worker scope, when this module is running as a web-style worker. */
interface WorkerScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<WorkerMessage>) => void,
  ): void;
  postMessage(message: unknown, transfer: ArrayBuffer[]): void;
}

// A browser or Deno worker scope exposes `self`; a Node worker thread does not
// and uses `parentPort` instead. `node:worker_threads` is imported through a
// variable specifier so bundlers building for the browser never try to resolve
// it.
//
// Order matters: Deno sets `process.versions.node` for npm compatibility, so
// `isNodeRuntime()` is true there as well. Testing `self` first keeps Deno
// workers on the web branch, which is the one they can actually use.
const workerScope = (globalThis as { self?: unknown }).self as
  | WorkerScope
  | undefined;

if (workerScope !== undefined) {
  workerScope.addEventListener("message", async (event) => {
    const reply = await handleMessage(event.data);
    if (!reply) return;
    postReply(
      (response, transfer) => workerScope.postMessage(response, transfer),
      reply,
    );
  });
} else if (isNodeRuntime()) {
  const specifier = "node:worker_threads";
  const { parentPort } = await import(
    /* @vite-ignore */ /* webpackIgnore: true */ specifier
  );

  if (parentPort === null) {
    throw new Error(
      "omero_codec_worker must be run as a worker thread, not imported on " +
        "the main thread.",
    );
  }

  // A MessagePort queues incoming messages until the first 'message' listener
  // is attached, so nothing posted before this module finishes evaluating is
  // lost. Awaiting here rather than chaining `.then()` keeps a failed startup
  // a rejection of the module's own evaluation, instead of an unhandled
  // rejection beside a worker that silently answers nothing.
  parentPort.on("message", async (msg: WorkerMessage) => {
    const reply = await handleMessage(msg);
    if (!reply) return;
    postReply(
      (response, transfer) => parentPort.postMessage(response, transfer),
      reply,
    );
  });
} else {
  throw new Error(
    "omero_codec_worker: no worker message channel available — this runtime " +
      "has no `self` and is not Node.",
  );
}
