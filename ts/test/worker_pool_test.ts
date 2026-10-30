// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/** Who terminates the codec worker when a pooled task fails. */

import type { WorkerLike } from "@fideus-labs/worker-pool";
import { test } from "vite-plus/test";

import { assert, assertFalse, assertRejects } from "./_assert.ts";
import { withCodecWorker } from "../src/utils/worker_pool.ts";

test("withCodecWorker terminates the worker it spawned for a rejected task", async () => {
  let terminated = false;
  await assertRejects(
    () =>
      withCodecWorker(null, (worker) => {
        const terminate = worker.terminate.bind(worker);
        worker.terminate = () => {
          terminated = true;
          terminate();
        };
        return Promise.reject(new Error("decode failed"));
      }),
    Error,
    "decode failed",
  );
  assert(terminated, "the spawned worker was not terminated");
});

test("withCodecWorker leaves a lent worker to the pool", async () => {
  let terminated = false;
  const lent: WorkerLike = {
    postMessage() {},
    terminate() {
      terminated = true;
    },
    addEventListener() {},
    removeEventListener() {},
  };
  await assertRejects(
    () =>
      withCodecWorker(lent, () => Promise.reject(new Error("decode failed"))),
    Error,
    "decode failed",
  );
  assertFalse(terminated, "the pool's worker was terminated by the task");
});
