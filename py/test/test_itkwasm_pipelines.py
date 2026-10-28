# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A pipeline function's first call runs alone; its later calls run in parallel."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ngff_zarr._itkwasm_pipelines import call_itkwasm


def test_first_calls_are_serialized_and_later_calls_are_not():
    # Concurrent first calls of an itkwasm_downsample function truncated the
    # compiled-module cache another thread had mapped: SIGBUS.
    guard = threading.Lock()
    finished = 0
    started_before_any_finished = 0
    running = 0
    overlap = 0

    def pipeline(index):
        nonlocal finished, started_before_any_finished, running, overlap
        with guard:
            started_before_any_finished += finished == 0
            running += 1
            overlap = max(overlap, running)
        time.sleep(0.05)
        with guard:
            running -= 1
            finished += 1
        return index

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: call_itkwasm(pipeline, i), range(8)))
    assert results == list(range(8))
    assert started_before_any_finished == 1  # the first call ran alone
    assert overlap > 1  # the calls after it overlapped
