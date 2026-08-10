// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Node.js worker tests, run against the built npm package.
 *
 * Plain Node has no global `Worker`, so every pooled codec task used to throw
 * "Worker is not defined" and `zarrSet`/`zarrGet` were unusable outside a
 * browser. @fideus-labs/worker-pool 2.0 added a `node:worker_threads` adapter
 * and this package's codec worker gained a Node branch; these tests exercise
 * that path end to end, in the runtime it was broken in.
 *
 * Run with `deno task test:node`, which rebuilds `npm/` first. Use the task
 * rather than invoking `node --test` directly: `deno task build:bundle` leaves
 * `npm/esm` in its browser form, where `scripts/inline_worker.ts` has replaced
 * the codec worker URL with a `blob:` URL that Node cannot load.
 */

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import { after, describe, it } from "node:test";

import {
  terminateWorkerPool,
  zarrGet,
  zarrSet,
} from "../../npm/esm/utils/worker_pool.js";

// Load zarrita from the built package's own dependency tree rather than this
// directory's. A bare `import "zarrita"` resolves to the copy Deno installed
// for the source tree, which is a different version from the one npm installs
// for the package — the test and the code under test would then hold two
// separate zarrita instances.
//
// The entry comes from the package's own `exports` map (zarrita publishes only
// an `import` condition, so CJS-style `require.resolve` cannot find it).
const zarritaPkgUrl = new URL(
  "../../npm/node_modules/zarrita/package.json",
  import.meta.url,
);
const zarritaPkg = JSON.parse(await readFile(zarritaPkgUrl, "utf8"));
const zarritaEntryUrl =
  new URL(zarritaPkg.exports["."].import, zarritaPkgUrl).href;
const zarr = await import(zarritaEntryUrl);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Shuffle mode recorded in a blosc frame's header flags (byte 2). */
function frameShuffle(chunk) {
  const flags = chunk[2];
  if (flags & 0x4) return "bitshuffle";
  if (flags & 0x1) return "shuffle";
  return "noshuffle";
}

/** Deterministic image-like data — smooth, so shuffle mode actually matters. */
function sampleData(Ctor, count) {
  const data = new Ctor(count);
  for (let i = 0; i < count; i++) {
    data[i] = Math.round(900 + 400 * Math.sin(i / 419) + (i % 13));
  }
  return data;
}

function strides(shape) {
  const out = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    out[i] = acc;
    acc *= shape[i];
  }
  return out;
}

function bloscCodecs(shuffle, typesize) {
  return [
    { name: "bytes", configuration: { endian: "little" } },
    {
      name: "blosc",
      configuration: {
        cname: "zstd",
        clevel: 5,
        shuffle,
        typesize,
        blocksize: 0,
      },
    },
  ];
}

async function createArray(store, { shape, chunkShape, dataType, codecs }) {
  return await zarr.create(zarr.root(store).resolve("/img"), {
    shape,
    chunk_shape: chunkShape,
    data_type: dataType,
    fill_value: 0,
    codecs,
  });
}

function encodedChunks(store) {
  return [...store.entries()]
    .filter(([k]) => !k.endsWith("zarr.json"))
    .map(([, v]) => v);
}

// Worker threads hold the Node event loop open — without this the test
// process would hang after the last assertion rather than exiting.
after(() => {
  terminateWorkerPool();
});

// ---------------------------------------------------------------------------
// The regression: workers usable at all under Node
// ---------------------------------------------------------------------------

describe("worker-accelerated IO under Node", () => {
  it("writes and reads back through worker threads", async () => {
    const shape = [16, 32, 32];
    const data = sampleData(Uint16Array, 16 * 32 * 32);
    const store = new Map();
    const arr = await createArray(store, {
      shape,
      chunkShape: [8, 16, 16],
      dataType: "uint16",
      codecs: bloscCodecs("shuffle", 2),
    });

    // Before worker-pool 2.0 this threw "Worker is not defined".
    await zarrSet(arr, null, { data, shape, stride: strides(shape) });

    assert.ok(encodedChunks(store).length > 0, "no chunks were written");

    const back = await zarrGet(arr, null);
    assert.deepEqual(Array.from(back.data), Array.from(data));
  });

  it("round-trips every supported element width", async () => {
    const shape = [8, 16, 16];
    const count = 8 * 16 * 16;
    const cases = [
      ["uint8", Uint8Array, 1],
      ["int16", Int16Array, 2],
      ["uint16", Uint16Array, 2],
      ["int32", Int32Array, 4],
      ["float32", Float32Array, 4],
      ["float64", Float64Array, 8],
    ];

    for (const [dataType, Ctor, typesize] of cases) {
      const data = sampleData(Ctor, count);
      const store = new Map();
      const arr = await createArray(store, {
        shape,
        chunkShape: shape,
        dataType,
        codecs: bloscCodecs("shuffle", typesize),
      });

      await zarrSet(arr, null, { data, shape, stride: strides(shape) });
      const back = await zarrGet(arr, null);

      assert.deepEqual(
        Array.from(back.data),
        Array.from(data),
        `${dataType} did not round-trip`,
      );
    }
  });

  it("handles edge chunks that do not divide the shape evenly", async () => {
    // Exercises the actualChunkShape correction inside the codec worker,
    // which this package delegates to fizarrita's shared core.
    const shape = [5, 13, 17];
    const data = sampleData(Uint16Array, 5 * 13 * 17);
    const store = new Map();
    const arr = await createArray(store, {
      shape,
      chunkShape: [4, 8, 8],
      dataType: "uint16",
      codecs: bloscCodecs("shuffle", 2),
    });

    await zarrSet(arr, null, { data, shape, stride: strides(shape) });
    const back = await zarrGet(arr, null);

    assert.deepEqual(Array.from(back.data), Array.from(data));
  });

  it("reads a partial selection", async () => {
    const shape = [16, 16];
    const data = sampleData(Int32Array, 256);
    const store = new Map();
    const arr = await createArray(store, {
      shape,
      chunkShape: [8, 8],
      dataType: "int32",
      codecs: bloscCodecs("shuffle", 4),
    });

    await zarrSet(arr, null, { data, shape, stride: strides(shape) });

    const row = await zarrGet(arr, [3, null]);
    assert.deepEqual(
      Array.from(row.data),
      Array.from(data.subarray(3 * 16, 4 * 16)),
    );
  });
});

// ---------------------------------------------------------------------------
// The blosc shuffle patch has to reach the Node worker too
// ---------------------------------------------------------------------------

describe("blosc shuffle under Node", () => {
  it("writes chunks matching the declared shuffle mode", async () => {
    // The registry patch is installed inside the worker, so it has to survive
    // the node:worker_threads entry path as well as the browser one.
    const shape = [16, 16, 16];
    const data = sampleData(Uint16Array, 16 * 16 * 16);

    for (const shuffle of ["noshuffle", "shuffle", "bitshuffle"]) {
      const store = new Map();
      const arr = await createArray(store, {
        shape,
        chunkShape: shape,
        dataType: "uint16",
        codecs: bloscCodecs(shuffle, 2),
      });

      await zarrSet(arr, null, { data, shape, stride: strides(shape) });

      for (const chunk of encodedChunks(store)) {
        assert.equal(
          frameShuffle(chunk),
          shuffle,
          `frame disagrees with declared ${shuffle}`,
        );
      }

      const back = await zarrGet(arr, null);
      assert.deepEqual(Array.from(back.data), Array.from(data));
    }
  });

  it("produces distinct bytes per shuffle mode", async () => {
    // Before the string→integer normalization every mode silently encoded as
    // noshuffle, making all three byte-identical.
    const shape = [16, 16, 16];
    const data = sampleData(Uint16Array, 16 * 16 * 16);
    const sizes = new Map();

    for (const shuffle of ["noshuffle", "shuffle", "bitshuffle"]) {
      const store = new Map();
      const arr = await createArray(store, {
        shape,
        chunkShape: shape,
        dataType: "uint16",
        codecs: bloscCodecs(shuffle, 2),
      });
      await zarrSet(arr, null, { data, shape, stride: strides(shape) });
      sizes.set(
        shuffle,
        encodedChunks(store).reduce((sum, c) => sum + c.byteLength, 0),
      );
    }

    assert.notEqual(sizes.get("shuffle"), sizes.get("noshuffle"));
    assert.notEqual(sizes.get("bitshuffle"), sizes.get("noshuffle"));
  });
});

// ---------------------------------------------------------------------------
// The work really happens off the main thread
// ---------------------------------------------------------------------------

describe("worker threads are actually used", () => {
  it("refuses to run the codec worker on the main thread", async () => {
    // The Node branch of the codec worker is only meaningful inside a worker
    // thread, and says so rather than silently attaching to nothing. That the
    // IO tests above pass while this holds is what shows the codec work is
    // genuinely being offloaded: `zarrSet` has no main-thread fallback for
    // blosc, so it either reaches a worker or throws.
    const workerUrl = new URL(
      "../../npm/esm/workers/omero_codec_worker.js",
      import.meta.url,
    ).href;

    await assert.rejects(
      () => import(workerUrl),
      /must be run as a worker thread/,
    );
  });

  it("exits cleanly once the pool is terminated", async () => {
    // Worker threads keep the Node event loop alive, so a consumer that never
    // calls terminateWorkerPool() gets a process that hangs. Run the whole
    // cycle in a child process and require it to exit on its own.
    //
    // The script goes to a file rather than `node -e`: `--input-type=module`
    // is inherited by worker threads, where it collides with their
    // file-based entry ("--input-type can only be used with string input").
    const workerPoolUrl = new URL(
      "../../npm/esm/utils/worker_pool.js",
      import.meta.url,
    ).href;
    const dir = await mkdtemp(join(tmpdir(), "ngff-zarr-node-exit-"));
    const scriptPath = join(dir, "exit_check.mjs");

    try {
      await writeFile(
        scriptPath,
        `
import { zarrSet, terminateWorkerPool } from ${JSON.stringify(workerPoolUrl)};
import * as zarr from ${JSON.stringify(zarritaEntryUrl)};

const shape = [4, 8, 8];
const store = new Map();
const arr = await zarr.create(zarr.root(store).resolve("/img"), {
  shape, chunk_shape: shape, data_type: "uint16", fill_value: 0,
  codecs: [
    { name: "bytes", configuration: { endian: "little" } },
    { name: "blosc", configuration: { cname: "zstd", clevel: 5,
      shuffle: "shuffle", typesize: 2, blocksize: 0 } },
  ],
});
await zarrSet(arr, null, {
  data: new Uint16Array(4 * 8 * 8), shape, stride: [64, 8, 1],
});
terminateWorkerPool();
`,
      );

      const child = spawn(process.execPath, [scriptPath], {
        stdio: ["ignore", "ignore", "pipe"],
      });
      let stderr = "";
      child.stderr.on("data", (d) => {
        stderr += d;
      });

      const exitCode = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          child.kill("SIGKILL");
          reject(new Error("process did not exit within 60s"));
        }, 60_000);
        child.on("error", reject);
        child.on("exit", (code) => {
          clearTimeout(timer);
          resolve(code);
        });
      });

      assert.equal(exitCode, 0, `child did not exit cleanly:\n${stderr}`);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

describe("pool scheduling under Node", () => {
  it("serves concurrent reads from one pool", async () => {
    // More concurrent operations than the pool has worker slots, so tasks
    // queue and workers get recycled rather than each call getting its own.
    const shape = [8, 32, 32];
    const data = sampleData(Uint16Array, 8 * 32 * 32);
    const store = new Map();
    const arr = await createArray(store, {
      shape,
      chunkShape: [4, 16, 16],
      dataType: "uint16",
      codecs: bloscCodecs("shuffle", 2),
    });

    await zarrSet(arr, null, { data, shape, stride: strides(shape) });

    const reads = await Promise.all(
      Array.from({ length: 12 }, () => zarrGet(arr, null)),
    );

    for (const read of reads) {
      assert.deepEqual(Array.from(read.data), Array.from(data));
    }
  });
});
