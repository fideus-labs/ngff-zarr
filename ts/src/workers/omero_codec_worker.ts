// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Custom Web Worker that extends fizarrita's codec worker with OMERO
 * statistics computation.
 *
 * Handles all standard fizarrita codec messages (init, decode, decode_into,
 * encode) PLUS a new `decode_and_stats` message that decodes a chunk AND
 * computes per-channel statistics in a single round-trip.
 *
 * Message protocol (superset of fizarrita's codec-worker):
 *   init:             → init_ok              (register codec pipeline metadata)
 *   decode:           → decoded              (decode raw bytes, return chunk)
 *   decode_into:      → decode_into_ok       (decode into SharedArrayBuffer)
 *   encode:           → encoded              (encode chunk data)
 *   decode_and_stats: → decoded_and_stats    (decode + compute channel stats)
 */

// fizarrita internals for codec pipeline
import { create_codec_pipeline } from "@fideus-labs/fizarrita/internals/codec-pipeline";
import {
  compat_chunk,
  set_from_chunk_binary,
} from "@fideus-labs/fizarrita/internals/setter";
import { get_ctr, get_strides } from "@fideus-labs/fizarrita/internals/util";

import type { Chunk, DataType } from "zarrita";
import type { CodecChunkMeta } from "@fideus-labs/fizarrita";

// Projection type — plain serializable data describing chunk→output mapping.
// Mirrors fizarrita's Projection type for worker messages.
type Indices = [start: number, stop: number, step: number];
type Projection =
  | { from: null; to: number }
  | { from: number; to: null }
  | { from: Indices; to: Indices };

// ngff-zarr stats functions
import {
  createAccumulator,
  extractChannel,
  updateAccumulator,
} from "../utils/compute_omero-shared.ts";

// Use typeof self for cross-environment compatibility (Deno, Node.js, Browser workers)
// deno-lint-ignore no-explicit-any
const ctx = self as any;

// ---------------------------------------------------------------------------
// Edge chunk shape correction (same as fizarrita's codec-worker)
// ---------------------------------------------------------------------------

function fixEdgeChunkShapeStride<D extends DataType>(
  chunk: Chunk<D>,
  actualChunkShape?: number[],
): Chunk<D> {
  if (actualChunkShape) {
    const expectedElements = actualChunkShape.reduce((a, b) => a * b, 1);
    const actualElements = (chunk.data as unknown as ArrayLike<unknown>).length;
    if (actualElements === expectedElements) {
      return {
        data: chunk.data,
        shape: actualChunkShape,
        stride: get_strides(actualChunkShape, "C"),
      };
    }
  }
  return chunk;
}

// ---------------------------------------------------------------------------
// Codec pipeline cache — keyed by metaId
// ---------------------------------------------------------------------------

const pipelineByMetaId = new Map<
  number,
  ReturnType<typeof create_codec_pipeline>
>();

const metaByMetaId = new Map<number, CodecChunkMeta>();

const pipelineByKey = new Map<
  string,
  ReturnType<typeof create_codec_pipeline>
>();

function getPipeline(
  metaId: number,
): ReturnType<typeof create_codec_pipeline> {
  const pipeline = pipelineByMetaId.get(metaId);
  if (!pipeline) {
    throw new Error(
      `No pipeline for metaId ${metaId}. Send an 'init' message first.`,
    );
  }
  return pipeline;
}

function getOrCreatePipelineLegacy(meta: CodecChunkMeta) {
  const key = JSON.stringify(meta);
  let pipeline = pipelineByKey.get(key);
  if (!pipeline) {
    pipeline = create_codec_pipeline({
      data_type: meta.data_type,
      shape: meta.chunk_shape,
      codecs: meta.codecs,
    });
    pipelineByKey.set(key, pipeline);
  }
  return pipeline;
}

// ---------------------------------------------------------------------------
// Serialized accumulator type for transfer
// ---------------------------------------------------------------------------

interface SerializedAccumulator {
  min: number;
  max: number;
  sample: number[];
  count: number;
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

type WorkerMessage =
  | {
    type: "init";
    id: number;
    metaId: number;
    meta: CodecChunkMeta;
  }
  | {
    type: "decode";
    id: number;
    bytes: ArrayBuffer;
    metaId?: number;
    meta?: CodecChunkMeta;
    actualChunkShape?: number[];
  }
  | {
    type: "decode_into";
    id: number;
    bytes: ArrayBuffer;
    metaId: number;
    output: SharedArrayBuffer;
    outputByteLength: number;
    outputStride: number[];
    projections: Projection[];
    bytesPerElement: number;
    actualChunkShape?: number[];
  }
  | {
    type: "encode";
    id: number;
    data: ArrayBuffer;
    metaId?: number;
    meta?: CodecChunkMeta;
  }
  | {
    type: "decode_and_stats";
    id: number;
    bytes: ArrayBuffer;
    metaId: number;
    actualChunkShape?: number[];
    nChannels: number;
    cIndex: number;
  };

ctx.addEventListener(
  "message",
  async (event: MessageEvent<WorkerMessage>) => {
    const msg = event.data;

    try {
      if (msg.type === "init") {
        metaByMetaId.set(msg.metaId, msg.meta);
        const pipeline = create_codec_pipeline({
          data_type: msg.meta.data_type,
          shape: msg.meta.chunk_shape,
          codecs: msg.meta.codecs,
        });
        pipelineByMetaId.set(msg.metaId, pipeline);
        ctx.postMessage({ type: "init_ok", id: msg.id });
        return;
      }

      if (msg.type === "decode") {
        const pipeline =
          msg.metaId !== undefined && pipelineByMetaId.has(msg.metaId)
            ? getPipeline(msg.metaId)
            : getOrCreatePipelineLegacy(msg.meta!);

        const bytes = new Uint8Array(msg.bytes);
        let chunk = (await pipeline.decode(bytes)) as Chunk<DataType>;
        chunk = fixEdgeChunkShapeStride(chunk, msg.actualChunkShape);

        const dataView = chunk.data as unknown as {
          buffer: ArrayBuffer;
          byteOffset: number;
          byteLength: number;
        };
        const buffer = dataView.buffer;
        const byteOffset = dataView.byteOffset;
        const byteLength = dataView.byteLength;

        let transferBuffer: ArrayBuffer;
        if (byteOffset === 0 && byteLength === buffer.byteLength) {
          transferBuffer = buffer;
        } else {
          transferBuffer = buffer.slice(byteOffset, byteOffset + byteLength);
        }

        ctx.postMessage(
          {
            type: "decoded" as const,
            id: msg.id,
            data: transferBuffer,
            shape: chunk.shape,
            stride: chunk.stride,
          },
          [transferBuffer],
        );
      } else if (msg.type === "decode_into") {
        const pipeline = getPipeline(msg.metaId);
        const bytes = new Uint8Array(msg.bytes);
        let chunk = (await pipeline.decode(bytes)) as Chunk<DataType>;
        chunk = fixEdgeChunkShapeStride(chunk, msg.actualChunkShape);

        const destView = new Uint8Array(msg.output, 0, msg.outputByteLength);
        // deno-lint-ignore no-explicit-any
        const src = compat_chunk(chunk as any);

        set_from_chunk_binary(
          { data: destView, stride: msg.outputStride },
          src,
          msg.bytesPerElement,
          msg.projections,
        );

        ctx.postMessage({ type: "decode_into_ok", id: msg.id });
      } else if (msg.type === "encode") {
        let pipeline: ReturnType<typeof create_codec_pipeline>;
        let meta: CodecChunkMeta;
        if (
          msg.metaId !== undefined && pipelineByMetaId.has(msg.metaId)
        ) {
          pipeline = getPipeline(msg.metaId);
          meta = metaByMetaId.get(msg.metaId)!;
        } else {
          meta = msg.meta!;
          pipeline = getOrCreatePipelineLegacy(meta);
        }

        const Ctr = get_ctr(meta.data_type) as unknown as {
          new (buf: ArrayBuffer, off: number, len: number): unknown;
          BYTES_PER_ELEMENT: number;
        };
        const data = new Ctr(
          msg.data,
          0,
          msg.data.byteLength / Ctr.BYTES_PER_ELEMENT,
        );
        const shape = meta.chunk_shape;
        const stride = get_strides(shape, "C");

        const chunk = { data, shape, stride } as Chunk<DataType>;
        // deno-lint-ignore no-explicit-any
        const encoded = await pipeline.encode(chunk as any);

        const transferBuffer = encoded.byteOffset === 0 &&
            encoded.byteLength === encoded.buffer.byteLength
          ? (encoded.buffer as ArrayBuffer)
          : encoded.buffer.slice(
            encoded.byteOffset,
            encoded.byteOffset + encoded.byteLength,
          );

        ctx.postMessage(
          {
            type: "encoded" as const,
            id: msg.id,
            bytes: transferBuffer,
          },
          [transferBuffer],
        );
      } else if (msg.type === "decode_and_stats") {
        // NEW: Decode chunk AND compute per-channel statistics in one pass
        const pipeline = getPipeline(msg.metaId);
        const bytes = new Uint8Array(msg.bytes);
        let chunk = (await pipeline.decode(bytes)) as Chunk<DataType>;
        chunk = fixEdgeChunkShapeStride(chunk, msg.actualChunkShape);

        const chunkData = chunk.data as unknown as ArrayLike<number>;
        const chunkShape = chunk.shape;
        const { nChannels, cIndex } = msg;

        // Compute per-channel statistics
        const accumulators: SerializedAccumulator[] = [];
        if (cIndex >= 0) {
          // Multi-channel: extract each channel and compute stats
          for (let ch = 0; ch < nChannels; ch++) {
            const channelData = extractChannel(
              chunkData,
              chunkShape,
              ch,
              cIndex,
            );
            const acc = createAccumulator();
            updateAccumulator(acc, channelData);
            accumulators.push(acc);
          }
        } else {
          // Single channel: compute stats on full chunk data
          const acc = createAccumulator();
          updateAccumulator(acc, chunkData);
          accumulators.push(acc);
        }

        // Transfer the decoded data buffer back for caching
        const dataView = chunk.data as unknown as {
          buffer: ArrayBuffer;
          byteOffset: number;
          byteLength: number;
        };
        const buffer = dataView.buffer;
        const byteOffset = dataView.byteOffset;
        const byteLength = dataView.byteLength;

        let transferBuffer: ArrayBuffer;
        if (byteOffset === 0 && byteLength === buffer.byteLength) {
          transferBuffer = buffer;
        } else {
          transferBuffer = buffer.slice(byteOffset, byteOffset + byteLength);
        }

        ctx.postMessage(
          {
            type: "decoded_and_stats" as const,
            id: msg.id,
            data: transferBuffer,
            shape: chunk.shape,
            stride: chunk.stride,
            accumulators,
          },
          [transferBuffer],
        );
      }
    } catch (error) {
      // Send error back to main thread
      const errorType = msg.type === "decode"
        ? "decoded"
        : msg.type === "encode"
        ? "encoded"
        : msg.type === "decode_into"
        ? "decode_into_ok"
        : msg.type === "decode_and_stats"
        ? "decoded_and_stats"
        : "init_ok";
      ctx.postMessage({
        type: errorType,
        id: msg.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
);
