// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import PQueue from "p-queue";

export type ChunkQueue = {
  add(fn: () => Promise<void>): void;
  onIdle(): Promise<void>;
};

export function createQueue(): ChunkQueue {
  const concurrency = Math.min(navigator?.hardwareConcurrency || 4, 128);
  const queue = new PQueue({ concurrency });
  return {
    add: (fn: () => Promise<void>) => queue.add(fn),
    onIdle: queue.onIdle.bind(queue),
  };
}
