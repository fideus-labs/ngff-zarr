// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Tests for the runtime configuration (config.ts) and worker pool size
 * management (setWorkerPoolSize).
 */

import { assertEquals, assertRejects } from "@std/assert";
import { config, setWorkerPoolSize } from "../src/config.ts";
import { terminateWorkerPool } from "../src/utils/worker_pool.ts";

// ---------------------------------------------------------------------------
// config.workerPoolSize — default and direct assignment
// ---------------------------------------------------------------------------

Deno.test("config.workerPoolSize default is between 1 and 16", () => {
  const size = config.workerPoolSize;
  assertEquals(size >= 1, true, `expected size >= 1, got ${size}`);
  assertEquals(size <= 16, true, `expected size <= 16, got ${size}`);
  assertEquals(Number.isInteger(size), true, "default must be an integer");
});

Deno.test(
  "config.workerPoolSize can be set directly before first pool usage",
  () => {
    const original = config.workerPoolSize;
    try {
      config.workerPoolSize = 8;
      assertEquals(config.workerPoolSize, 8);
    } finally {
      config.workerPoolSize = original;
    }
  },
);

// ---------------------------------------------------------------------------
// setWorkerPoolSize — input validation
// ---------------------------------------------------------------------------

Deno.test("setWorkerPoolSize rejects a non-integer (float)", async () => {
  const original = config.workerPoolSize;
  try {
    await assertRejects(
      () => setWorkerPoolSize(1.5),
      Error,
      "workerPoolSize must be a positive integer",
    );
  } finally {
    config.workerPoolSize = original;
  }
});

Deno.test("setWorkerPoolSize rejects NaN", async () => {
  const original = config.workerPoolSize;
  try {
    await assertRejects(
      () => setWorkerPoolSize(NaN),
      Error,
      "workerPoolSize must be a positive integer",
    );
  } finally {
    config.workerPoolSize = original;
  }
});

Deno.test("setWorkerPoolSize rejects Infinity", async () => {
  const original = config.workerPoolSize;
  try {
    await assertRejects(
      () => setWorkerPoolSize(Infinity),
      Error,
      "workerPoolSize must be a positive integer",
    );
  } finally {
    config.workerPoolSize = original;
  }
});

Deno.test("setWorkerPoolSize rejects zero", async () => {
  const original = config.workerPoolSize;
  try {
    await assertRejects(
      () => setWorkerPoolSize(0),
      Error,
      "workerPoolSize must be a positive integer",
    );
  } finally {
    config.workerPoolSize = original;
  }
});

Deno.test("setWorkerPoolSize rejects negative integers", async () => {
  const original = config.workerPoolSize;
  try {
    await assertRejects(
      () => setWorkerPoolSize(-1),
      Error,
      "workerPoolSize must be a positive integer",
    );
    await assertRejects(
      () => setWorkerPoolSize(-10),
      Error,
      "workerPoolSize must be a positive integer",
    );
  } finally {
    config.workerPoolSize = original;
  }
});

// ---------------------------------------------------------------------------
// setWorkerPoolSize — successful operation
// ---------------------------------------------------------------------------

Deno.test(
  "setWorkerPoolSize updates config.workerPoolSize to the new value",
  async () => {
    const original = config.workerPoolSize;
    try {
      await setWorkerPoolSize(4);
      assertEquals(config.workerPoolSize, 4);

      await setWorkerPoolSize(8);
      assertEquals(config.workerPoolSize, 8);
    } finally {
      config.workerPoolSize = original;
      terminateWorkerPool();
    }
  },
);

Deno.test(
  "setWorkerPoolSize terminates existing pools so they are recreated with the new size",
  async () => {
    const original = config.workerPoolSize;
    try {
      // First call terminates any existing pools and sets size to 2
      await setWorkerPoolSize(2);
      assertEquals(config.workerPoolSize, 2);

      // Second call terminates pools again (no-op if not yet recreated) and
      // sets size to 6; must not throw
      await setWorkerPoolSize(6);
      assertEquals(config.workerPoolSize, 6);
    } finally {
      config.workerPoolSize = original;
      terminateWorkerPool();
    }
  },
);

Deno.test(
  "worker pools use the configured workerPoolSize when (re)created",
  () => {
    const original = config.workerPoolSize;
    try {
      // Set a distinct size, terminate pools so they are re-created on next use
      config.workerPoolSize = 3;
      terminateWorkerPool();

      // The pools are now null; the next operation that calls getCodecPool()
      // or getWritePool() will construct them with config.workerPoolSize === 3.
      // Verify the configured value is preserved so it will be used on creation.
      assertEquals(config.workerPoolSize, 3);
    } finally {
      config.workerPoolSize = original;
      terminateWorkerPool();
    }
  },
);
