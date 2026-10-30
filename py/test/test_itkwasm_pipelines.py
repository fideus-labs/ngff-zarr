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


def test_every_itkwasm_downsample_call_goes_through_the_lock():
    # resample_bounding_box and gaussian_kernel_radius ran their pipelines
    # directly: a first call in two threads at once raced on the module cache.
    import ast
    from pathlib import Path

    import ngff_zarr

    direct = []
    for path in Path(ngff_zarr.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        pipelines = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "itkwasm_downsample"
            for alias in node.names
        }
        direct += [
            f"{path.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in pipelines
        ]
    assert not direct, f"call these through call_itkwasm: {direct}"
